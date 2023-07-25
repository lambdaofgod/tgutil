import logging
import ast
import json
import logging
from pathlib import Path as P
from typing import Dict, List, Tuple

import itertools
from tgutil.context_loader import ContextLoader
import clearml
import fire
import jsonlines
import numpy as np
import pandas as pd
import tqdm
from clearml import Dataset, PipelineController, Task
from pydantic.dataclasses import dataclass

from tgutil.prompting import *
from tgutil.prompting_utils import (
    MinichainHFConfig,
    MinichainPrompterWrapper,
    MinichainRWKVConfig,
    PrompterWrapper,
)
from tgutil.configs import (
    PipelineConfig,
    SamplingConfig,
    TextGenerationConfig,
    PromptConfig,
)
from tgutil.prompting import ContextPromptInfo
from tgutil.prompting_utils import PrompterWrapper
import json
from returns.result import Success

np.random.seed(seed=0)


logging.basicConfig(level="INFO")


class DocumentExpander(BaseModel):
    text_generation_config: TextGenerationConfig
    prompts_config: PromptConfig

    def expand_documents(
        self,
        prompt_infos: List[ContextPromptInfo],
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        expand documents by generating tasks using PrompterWrapper
        with prompt template et c specified in TextGenerationConfig
        """
        print(f"loading data from {self.prompts_config.data_path}...")
        model_nm = P(
            self.text_generation_config.model_name.replace("/", "-")
        ).parent.name
        out_path = P(self.text_generation_config.out_dir) / (model_nm + ".jsonl")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        writer_kwargs = {"file_path": out_path}
        print(
            f"expanding data using text generator model: {self.text_generation_config.model_name}..."
        )

        generated_texts_df, failures = self._get_generation_results(prompt_infos)
        return generated_texts_df, failures

    @staticmethod
    def sample_data(
        prompts_config: PromptConfig,
        sampling_config: SamplingConfig,
    ):
        prompt_infos = ContextLoader(
            data_path=sampling_config.pq_data_path,
            field_mapping=prompts_config.field_mapping,
        ).load_prompt_infos(n_samples=sampling_config.n_samples)
        logging.info("created samples")
        return list(prompt_infos)

    def _expand_documents_single_step(
        self,
        prompt_infos: List[ContextPromptInfo],
        generation: int = 0,
    ) -> Tuple[pd.DataFrame, dict]:
        prompter_wrapper = MinichainPrompterWrapper.create(
            self.text_generation_config,
            None,
            self.prompts_config.templates_path,
            self.prompts_config.prompt_template_name,
        )

        results = [
            prompter_wrapper.get_dict_with_generated_text(pinfo)
            for pinfo in tqdm.tqdm(prompt_infos)
        ]
        records = [result.unwrap() for result in results if type(result) is Success]
        failures = [
            {"id": pinfo.id, "failure": str(result.failure())}
            for (pinfo, result) in zip(prompt_infos, results)
            if not type(result) is Success
        ]

        return (
            pd.DataFrame.from_records(records).assign(generation=generation),
            failures,
        )

    def _get_generation_result_pairs(self, prompt_infos):
        for generation in tqdm.tqdm(
            range(self.text_generation_config.n_generations), desc="generating"
        ):
            yield self._expand_documents_single_step(
                prompt_infos, generation=generation
            )

    def _get_generation_results(self, prompt_infos):
        pair_results = self._get_generation_result_pairs(prompt_infos)
        [pair_results_copy1, pair_results_copy2] = itertools.tee(pair_results)
        result_dfs = [df for (df, _) in pair_results_copy1]
        failures = [failure for (_, failure) in pair_results_copy2]
        return pd.concat(result_dfs), failures
