import logging
import ast
import json
import logging
from pathlib import Path as P
from typing import Dict, List

from tgutil.context_loader import ContextLoader
import clearml
import fire
import jsonlines
import numpy as np
import pandas as pd
import tqdm
from clearml import Dataset, PipelineController, Task
from mlutil.text import rwkv_utils
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
from tgutil.experiment_managers import ClearMLExperimentManager
from tgutil.prompting import PromptInfo
from tgutil.prompting_utils import PrompterWrapper
import json

np.random.seed(seed=0)


logging.basicConfig(level="INFO")


def get_experiment_manager(project_name, task_name, config=dict()):
    return ClearMLExperimentManager(
        project_name=project_name, task_name=task_name, config=config
    )


def sample_data(
    sampling_config: SamplingConfig,
):
    prompt_infos = ContextLoader(
        data_path=sampling_config.pq_data_path
    ).load_prompt_infos(n_samples=sampling_config.n_samples)
    logging.info("created samples")
    return list(prompt_infos)


def _expand_documents_single_step(
    text_generation_config: TextGenerationConfig,
    prompts_config: PromptConfig,
    prompt_infos: List[PromptInfo],
):
    prompter_wrapper = MinichainPrompterWrapper.create(
        text_generation_config,
        None,
        prompts_config.templates_path,
        prompts_config.prompt_template_name,
    )

    records = [
        prompter_wrapper.get_dict_with_generated_text(pinfo)
        for pinfo in tqdm.tqdm(prompt_infos)
    ]
    return pd.DataFrame.from_records(records)


def expand_documents(
    text_generation_config: TextGenerationConfig,
    prompts_config: PromptConfig,
    prompt_infos: List[PromptInfo],
):
    """
    expand documents by generating tasks using PrompterWrapper
    with prompt template et c specified in TextGenerationConfig
    """
    print(f"loading data from {prompts_config.data_path}...")
    model_nm = P(text_generation_config.model_name.replace("/", "-")).parent.name
    out_path = P(text_generation_config.out_dir) / (model_nm + ".jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    writer_kwargs = {"file_path": out_path}
    print(
        f"expanding data using text generator model: {text_generation_config.model_name}..."
    )

    generated_texts_dfs = [
        _expand_documents_single_step(
            text_generation_config, prompts_config, prompt_infos
        ).assign(generation_id=i)
        for i in tqdm.tqdm(
            range(text_generation_config.n_generations), desc="generating"
        )
    ]
    return pd.concat(generated_texts_dfs)
