from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from tgutil.prompting import ContextPromptInfo, PromptInfo
import ast


class ContextLoader(BaseModel):
    data_path: str
    field_mapping: dict

    @property
    def used_cols(self):
        return list(self.field_mapping.values())

    def _convert_lists_to_str(self, df):
        str_df = df.copy()
        for col in self.used_cols:
            if type(str_df[col].iloc[0]) is list:
                str_df[col] = str_df[col].apply(" ".join)
        return str_df

    def _get_pandas_dicts(self, df):
        for _, row in df.iterrows():
            row_dict = dict(row)
            yield row_dict

    def _load_df(self, path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".json"):
            return pd.read_json(path, orient="records", lines=True)

    def load_contexts_df(self, n_samples=1000):
        raw_df = self._load_df(self.data_path)
        df = self._convert_lists_to_str(raw_df[self.used_cols])
        n_samples = df.shape[0] if n_samples is None else n_samples

        if n_samples is None:
            return df
        else:
            return df.sample(n_samples)

    def load_prompt_infos(self, n_samples, n_shots=2):
        predicted_df = self.load_contexts_df(n_samples=n_samples)
        predicted_records = list(self._get_pandas_dicts(predicted_df))
        contexts_per_shot = list(
            self._get_pandas_dicts(
                self.load_contexts_df(n_samples=n_samples or len(predicted_df))
            )
            for _ in range(n_shots)
        )
        contexts = zip(*contexts_per_shot)

        prompt_infos = [
            PromptInfo.from_predicted_record(predicted_record, self.field_mapping)
            for predicted_record in predicted_records
        ]
        prompt_infos_contexts = [
            [
                PromptInfo.from_predicted_record(record, self.field_mapping)
                for record in records
            ]
            for records in contexts
        ]

        ctx_prompt_infos = [
            ContextPromptInfo(
                prompt_info=prompt_info, context_prompt_infos=context_prompt_infos
            )
            for (prompt_info, context_prompt_infos) in zip(
                prompt_infos, prompt_infos_contexts
            )
        ]
        return ctx_prompt_infos

    def load_prompt_info(context, predicted_record):
        return ContextPromptInfo(
            repo_records=context,
            predicted_repo_record=dict(predicted_record),
            true_tasks=ast.literal_eval(predicted_record["tasks"]),
        )
