import abc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from promptify import Prompter
from minichain import prompt
from minichain.backend import Backend
from mlutil import minichain_utils
from pydantic import BaseModel, Field
import jinja2
from tgutil.prompting import PromptInfo
import requests
from typing import Iterator
from tgutil.configs import TextGenerationConfig, APIConfig


def get_default_api_params():
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


class APIBackend(Backend, BaseModel):
    endpoint_url: str
    default_parameters: dict = Field(default_factory=get_default_api_params)

    @property
    def description(self) -> str:
        return ""

    def run(self, request: str) -> str:
        api_json = {**self.default_parameters, "prompt": request}
        return requests.post(self.endpoint_url, json=api_json).json()["text"]

    def run_stream(self, request: str) -> Iterator[str]:
        yield self.run(request)

    async def arun(self, request: str) -> str:
        return self.run(request)

    def _block_input(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)

    def _block_output(self, gr):  # type: ignore
        return gr.Textbox(show_label=False)


class PrompterWrapper(abc.ABC):
    prompter: Prompter
    template_name: str

    @abc.abstractmethod
    def get_generation_results(self, pinfo: PromptInfo) -> dict:
        pass

    def get_dict_with_generated_text(self, pinfo: PromptInfo):
        out_record = dict(pinfo)
        generated_text = self.get_generation_results(pinfo)
        out_record["input_text"] = pinfo.format_jinja_template(self.prompt_template)
        out_record["generated_text"] = generated_text
        return out_record


class MinichainHFConfig(BaseModel):
    model_name_or_path: str
    device: int = Field(default=dict(device=0))


class MinichainRWKVConfig(BaseModel):
    model_name_or_path: str


class MinichainPrompterWrapper(PrompterWrapper, BaseModel):
    generate_text_fn: Callable[[PromptInfo], str]
    prompt_template: jinja2.Template

    def get_generation_results(self, pinfo: PromptInfo):
        return self.generate_text_fn(pinfo).run()

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
        def generate_text_fn(model, prompt_info: PromptInfo):
            """
            calling model on a dict takes care of
            formatting the prompt from str template or template loaded from file
            """
            return model(dict(prompt_info))

        def format_promptinfo(pinfo: PromptInfo):
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
            return APIBackend(endpoint_url=config.endpoint_url)
        if type(config) is MinichainRWKVConfig:
            return minichain_utils.RWKVModel.load(config.model_name_or_path)
        else:
            return minichain_utils.HuggingFaceLocalModel.load(
                config.model_name_or_path,
                max_new_tokens=max_new_tokens,
            )

    class Config:
        arbitrary_types_allowed = True


class PromptifyPrompterWrapper(PrompterWrapper):
    prompter: Prompter
    template_name: str
    max_tokens: int = Field(default=20)

    def get_generation_results(self, pinfo: PromptInfo):
        promptify_args = pinfo.get_promptify_input_dict()
        return self.prompter.fit(
            template_name=self.template_name,
            max_tokens=self.max_tokens,
            **promptify_args
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
