import abc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any

from minichain import prompt
from minichain.backend import Backend
from mlutil import minichain_utils
from pydantic import BaseModel, Field
import jinja2
from tgutil.prompting import ContextPromptInfo
import requests
from typing import Iterator
from tgutil.configs import TextGenerationConfig, APIConfig
from returns.result import Result, Success, Failure

try:
    from promptify import Prompter
except ImportError:
    Prompter = Any


class APIParams:
    @classmethod
    def get_api_params(cls, input_text: str, flavor: str):
        api_params_mapping = {
            "text-generation-inference": cls._get_text_generation_inference_default_api_params,
            "lmserver": cls._get_lmserver_default_api_params,
            "vllm": cls._get_vllm_default_api_params,
            "openai": cls._get_openai_default_api_params,
        }
        api_text_field_mapping = {
            "text-generation-inference": "inputs",
            "lmserver": "prompt",
            "vllm": "prompt",
            "openai": "prompt",
        }
        assert flavor in api_params_mapping.keys()
        params = api_params_mapping[flavor]()
        params[api_text_field_mapping[flavor]] = input_text
        return params

    @classmethod
    def _get_text_generation_inference_default_api_params(cls):
        return {
            "inputs": "My name is Olivier and I",
            "parameters": {
                "best_of": 1,
                "decoder_input_details": True,
                "details": True,
                "do_sample": True,
                "max_new_tokens": 20,
                "repetition_penalty": 1.03,
                "return_full_text": False,
                "seed": None,
                "stop": ["photographer"],
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.95,
                "truncate": None,
                "typical_p": 0.95,
                "watermark": True,
            },
        }

    @classmethod
    def _get_lmserver_default_api_params(cls):
        return {
            "prompt": "string",
            "max_length": 512,
            "max_new_tokens": 50,
            "n": 1,
            "stop": "string",
            "stream": False,
            "sampling_parameters": {
                "temperature": 0.5,
                "top_k": None,
                "top_p": 0.9,
                "logit_bias": {},
                "presence_penalty": 0,
                "frequency_penalty": 1.0,
                "repetition_penalty": 1.8,
                "typical_p": 0.9,
            },
        }

    @classmethod
    def _get_vllm_default_api_params(cls):
        return {
            "prompt": "string",
            "sampling_parameters": {
                "use_beam_search": False,
                "max_tokens": 50,
                "temperature": 0.5,
                "top_k": -1,
                "top_p": 0.9,
                "presence_penalty": 0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        }

    @classmethod
    def _get_openai_default_api_params(cls):
        return {
            "model": "gpt-3.5-turbo",
            "prompt": "string",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "stop": None
        }


class APIBackend(Backend, BaseModel):
    endpoint_url: str
    flavor: str

    @property
    def description(self) -> str:
        return ""

    def run(self, request: str) -> str:
        params = APIParams.get_api_params(request, self.flavor)

        response = requests.post(self.endpoint_url, json=params).json()
        return self._get_text(response)

    def _get_text(self, response):
        if self.flavor == "lmserver":
            return response["texts"]
        elif self.flavor == "text-generation-inference":
            return [response["generated_text"]]
        elif self.flavor == "vllm":
            return response["text"]
        elif self.flavor == "openai":
            return response["choices"][0]["text"]
        else:
            raise ValueError(f"Unknown flavor: {self.flavor}")

    def run_stream(self, request: str) -> Iterator[str]:
        yield self.run(request)

    async def arun(self, request: str) -> str:
        return self.run(request)

    def _block_input(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)

    def _block_output(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)


class PrompterWrapper(abc.ABC):
    @abc.abstractmethod
    def get_generation_results(
        self, pinfo: ContextPromptInfo
    ) -> Result[dict, Exception]:
        pass

    def get_dict_with_generated_text(
        self, pinfo: ContextPromptInfo
    ) -> Result[dict, Exception]:
        out_record = dict(pinfo)
        out_record["input_text"] = pinfo.format_jinja_template(self.prompt_template)

        generated_result = self.get_generation_results(pinfo)
        return generated_result.map(lambda text: self._add_to_record(text, out_record))

    @classmethod
    def _add_to_record(cls, text, record):
        record["generated_text"] = text
        return record


class MinichainHFConfig(BaseModel):
    model_name_or_path: str
    device: int = Field(default=dict(device=0))


class MinichainRWKVConfig(BaseModel):
    model_name_or_path: str


class MinichainPrompterWrapper(PrompterWrapper, BaseModel):
    generate_text_fn: Callable[[ContextPromptInfo], str]
    prompt_template: jinja2.Template

    def get_generation_results(
        self, pinfo: ContextPromptInfo
    ) -> Result[str, Exception]:
        try:
            generation_result = self.generate_text_fn(pinfo).run()
            return Success(generation_result)
        except Exception as e:
            return Failure(e)

    @classmethod
    def create(
        cls,
        text_generation_config: TextGenerationConfig,
        prompt_template: Optional[str],
        prompt_templates_path: Optional[str],
        prompt_template_name: Optional[str],
        max_length: int = 512,
        max_new_tokens: int = 50,
    ):
        if prompt_template is not None:
            prompt_template = prompt_template
        else:
            prompt_template_path = str(
                Path(prompt_templates_path) / prompt_template_name
            )
            with open(prompt_template_path) as f:
                prompt_template = f.read().strip()

        minichain_kwargs = cls.make_minichain_template_kwargs(
            prompt_template_path=prompt_template_path,
            prompt_template=prompt_template if prompt_template_path is None else None,
        )

        model = cls.load_model(
            text_generation_config,
            max_length,
            max_new_tokens,
        )

        @prompt(model, **minichain_kwargs)
        def generate_text_fn(model, prompt_info: ContextPromptInfo):
            """
            calling model on a dict takes care of
            formatting the prompt from str template or template loaded from file
            """

            return model(dict(prompt_info))

        def format_promptinfo(pinfo: ContextPromptInfo):
            return pinfo.format_prompt()

        return MinichainPrompterWrapper(
            generate_text_fn=generate_text_fn,
            prompt_template=jinja2.Template(prompt_template),
        )

    @classmethod
    def make_minichain_template_kwargs(
        cls,
        prompt_template: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
    ):
        if prompt_template_path is not None:
            return dict(template_file=prompt_template_path)
        else:
            assert prompt_template is None
            return dict(template=prompt_template)

    @classmethod
    def load_model(
        cls,
        config: Union[APIConfig, MinichainHFConfig, MinichainRWKVConfig],
        max_length,
        max_new_tokens,
    ):
        if type(config) is APIConfig:
            return APIBackend(endpoint_url=config.endpoint_url, flavor=config.flavor)
        if type(config) is MinichainRWKVConfig:
            return minichain_utils.RWKVModel.load(config.model_name_or_path)
        else:
            return minichain_utils.HuggingFaceLocalModel.load(
                config.model_name_or_path,
                max_new_tokens=max_new_tokens,
            )

    class Config:
        arbitrary_types_allowed = True


class PromptFormatter(BaseModel):
    content_text_field: str = Field(default="dependencies")
    record_name_field: str = Field(default="repo")
    true_text_field: str = Field(default="tasks")

    def get_repo_args(self, record, use_tasks=True):
        return [
            record[self.record_name_field],
            ", ".join(record[self.repo_text_field]),
            record[self.true_text_field],
        ]

    def get_prompt_args(self):
        repo_with_tags_args = [
            self.get_repo_args(record) for record in self.repo_records
        ]
        predicted_repo_args = self.get_repo_args(
            self.predicted_repo_record, use_tasks=False
        )
        return [
            arg
            for rec_args in repo_with_tags_args + [predicted_repo_args]
            for arg in rec_args
        ]

    def get_prompter_input_dict(self):
        args_dict = {}
        args_dict["input_record_info"] = [
            (
                rec[self.record_name_field],
                rec[self.content_text_field],
                rec[self.true_text_field],
            )
            for rec in self.repo_records
        ]
        args_dict["predicted_record"] = self.predicted_repo_record[
            self.record_name_field
        ]
        args_dict["predicted_content"] = self.predicted_repo_record[
            self.content_text_field
        ]
        return args_dict


class PromptifyPrompterWrapper(PrompterWrapper):
    prompter: Prompter
    template_name: str
    max_tokens: int = Field(default=20)
    formatter: PromptFormatter

    def get_generation_results(self, pinfo: ContextPromptInfo):
        promptify_args = self.formatter.get_prompter_input_dict(pinfo)
        return self.prompter.fit(
            template_name=self.template_name,
            max_tokens=self.max_tokens,
            **promptify_args,
        )

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def create(
        model_path, templates_path, template_name, max_tokens=20, use_fake_model=False
    ):
        if use_fake_model:
            model = FakeModel()
        else:
            model = load_model(model_path)
        nlp_prompter = Prompter(model, templates_path=templates_path)
        return PrompterWrapper(
            prompter=nlp_prompter, template_name=template_name, max_tokens=max_tokens
        )
