import pandas as pd
import re
from pydantic import BaseModel, Field
import ast
from typing import Optional


class EvalDFPreprocessor(BaseModel):
    """
    utils for loading data for generation evaluation
    """

    id_col: str
    reference_text_col: str
    sanitize_re: Optional[str] = Field(default="\[(.*)\]")

    def get_eval_df_from_raw_generated_text(
        self,
        generated_text_df: pd.DataFrame,
        reference_text_df: pd.DataFrame,
    ):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        # task_list_df = generated_text_df["predicted_text"].apply(get_task_list)
        evaluation_df = self.prepare_texts_for_scoring(
            generated_text_df.merge(reference_text_df, on=self.id_col)
        )
        return evaluation_df

    @classmethod
    def get_eval_dffrom_generated_text(cls, generated_text_df: pd.DataFrame):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        return cls.prepare_texts_for_scoring(generated_text_df)

    def prepare_texts_for_scoring(self, texts_df):
        raw_reference_texts = texts_df[self.reference_text_col]
        processed_reference_texts = self.preprocess_texts(
            texts_df[self.reference_text_col]
        )
        return texts_df.assign(
            raw_generated_text=texts_df["generated_text"],
            raw_reference_text=raw_reference_texts,
            reference_text=processed_reference_texts,
            generated_text=self.preprocess_texts(texts_df["generated_text"]),
        )

    @classmethod
    def get_task_list(cls, task_line):
        return task_line.split(", ")

    def preprocess_texts(self, texts):
        return texts.apply(self.sanitize_string)

    @classmethod
    def sanitize_string(cls, s):
        try:
            sanitize_re = "\[(.*)\]?"
            extracted_sanitized_part = next(re.finditer(sanitize_re, s)).group()
        except:
            extracted_sanitized_part = s
        punctuation_pattern = r"([{}])".format("\[\]\(\)'")
        cleaned_m = re.sub(punctuation_pattern, "", extracted_sanitized_part)
        deduplicated_m = set([part.strip() for part in cleaned_m.split(r",")])
        return " ".join(deduplicated_m)

    @classmethod
    def sanitize_str(cls, s, sanitize_re):
        """older version"""
        if type(s) is list and len(s) == 1:
            s = s[0]
        if sanitize_re is not None:
            return cls.sanitize_with_re(s, sanitize_re)
        else:
            return s

    @classmethod
    def sanitize_with_re(cls, s, sanitize_re):
        try:
            m = next(re.finditer(sanitize_re, s))
            return ", ".join(ast.literal_eval(m.group()))
        except Exception as e:
            return s

    @classmethod
    def postprocess_reference_texts(cls, texts_df, reference_text_col):
        texts_df[self.reference_text_col].str.join(", ")
