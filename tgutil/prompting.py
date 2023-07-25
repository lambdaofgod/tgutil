from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import sys
import re
from pathlib import Path
import ast
from typing import List, Optional, Dict, Any
from pathlib import Path as P
from pydantic import BaseModel, Field, validator
from pydantic_core.core_schema import FieldValidationInfo


import pandas as pd
from tgutil import utils


def preprocess_dep(dep):
    return P(dep).name


def select_deps(deps, n_deps):
    return [preprocess_dep(dep) for dep in deps if not "__init__" in dep][:n_deps]


def preprocess_dep(dep):
    return P(dep).name


def select_deps(deps, n_deps):
    return [preprocess_dep(dep) for dep in deps if not "__init__" in dep][:n_deps]


def get_records_by_index(
    data_df, indices, fields=["repo", "dependencies", "tasks"], n_deps=10
):
    records_df = data_df.iloc[indices].copy()
    raw_deps = records_df["dependencies"].str.split()
    records_df["dependencies"] = raw_deps.apply(lambda dep: select_deps(dep, n_deps))
    return records_df[fields].to_dict(orient="records")


def get_prompt_template(repo_prompt, prefix="", n_repos=2):
    return "\n\n".join([prefix] + [repo_prompt.strip()] * (n_repos + 1)).strip()


class PromptInfo(BaseModel):
    content: str
    id: str
    name: str = Field(default="")
    true_text: str = Field(default="")

    @classmethod
    def from_predicted_record(cls, record, field_mapping):
        return cls(
            id=utils.dict_hash(record),
            content=record[field_mapping["content"]],
            name=record[field_mapping["name"]],
            true_text=record[field_mapping["true_text"]],
        )


class ContextPromptInfo(BaseModel):
    """
    information about sample repositories passed to prompt
    """

    prompt_info: PromptInfo
    context_prompt_infos: List[PromptInfo]
    id: str = Field(default="")

    @validator("id", always=True)
    @classmethod
    def set_id(cls, v, values):
        if v == "":
            return utils.dict_hash(dict(values))
        else:
            return v

    def format_jinja_template(self, template):
        return template.render(dict(self))

    def format_prompt(self, prompt_template):
        n_placeholders = len(re.findall(r"\{\}", prompt_template))
        expected_n_placeholders = 3 * (len(self.repo_records) + 1)
        assert (
            n_placeholders == expected_n_placeholders
        ), f"unexpected placeholders: {n_placeholders}"
        return prompt_template.format(*self.get_prompt_args())

    @classmethod
    def from_df(cls, data_df, pos_indices, pred_index, n_deps=10):
        predicted_record = get_records_by_index(data_df, [pred_index], n_deps=n_deps)[0]
        return ContextPromptInfo(
            context_records=get_records_by_index(data_df, pos_indices, n_deps=n_deps),
            prompt_info=PromptInfo(
                **predicted_record,
            ),
        )
