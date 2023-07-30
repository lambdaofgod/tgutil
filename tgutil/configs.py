from pydantic import Field, BaseModel
from pathlib import Path as P
import yaml
from typing import Optional, Dict, Any


class YamlModel(BaseModel):
    @classmethod
    def parse_file(cls, path: str):
        with open(path) as f:
            return cls(**yaml.safe_load(f))


class SamplingConfig(YamlModel):
    pq_data_path: str
    out_data_path: str
    n_samples: Optional[int] = 100
    dset_kwargs: Dict[str, Any] = dict(
        dataset_project="tgutil_llms", dataset_name="prompt_info_sample"
    )


class PromptConfig(YamlModel):
    prompt_template_name: str
    templates_path: str
    data_path: str
    field_mapping: dict


class TextGenerationConfig(YamlModel):
    out_dir: str
    n_generations: int


class LocalModelConfig(TextGenerationConfig):
    model_name: str


class APIConfig(TextGenerationConfig):
    endpoint_url: str
    model_name: str
    flavor: str = Field(default="text-generation-inference")


def load_config_from_dict(config_dict: dict):
    if "endpoint_url" in config_dict.keys():
        return APIConfig(**config_dict)
    else:
        return LocalModelConfig(**config_dict)


class ConfigPaths(YamlModel):
    sampling: str
    generation: str
    prompts: str
    name: str
    project: str
    root_dir: str

    @staticmethod
    def load(config_path: str):
        root_dir = P(config_path).parent
        with open(config_path) as f:
            paths = yaml.safe_load(f)
        return ConfigPaths(root_dir=str(root_dir), **paths)


class PipelineConfig(YamlModel):
    sampling_config: SamplingConfig
    prompt_config: PromptConfig
    generation_config: TextGenerationConfig
    name: str
    project: str
    paperswithcode_path: str

    @staticmethod
    def load_from_config_paths(cfg_paths: ConfigPaths):
        root_dir = cfg_paths.root_dir
        sampling_config = SamplingConfig.parse_file(
            f"{str(root_dir)}/sampling/{cfg_paths.sampling}.yaml"
        )
        generation_config = TextGenerationConfig.parse_file(
            f"{str(root_dir)}/generation/{cfg_paths.generation}.yaml"
        )
        prompt_config = PromptConfig.parse_file(
            f"{str(root_dir)}/prompts/{cfg_paths.prompts}.yaml"
        )
        return PipelineConfig(
            sampling_config=sampling_config,
            generation_config=generation_config,
            prompt_config=prompt_config,
            name=cfg_paths.name,
            project=cfg_paths.project,
        )

    @staticmethod
    def load(config_path: str):
        p = P(config_path)
        if p.is_dir():
            config_path = p / "config.yaml"
        else:
            config_path = p
        cfg_paths = ConfigPaths.load(config_path)
        return PipelineConfig.load_from_config_paths(cfg_paths)
