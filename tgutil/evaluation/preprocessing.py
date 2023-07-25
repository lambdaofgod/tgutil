import pandas as pd
import re
from pydantic import BaseModel


class EvalDFPreprocessor(BaseModel):
    """
    utils for loading data for generation evaluation
    """

    id_col: str
    reference_text_col: str

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
        pred_raw_texts = (
            texts_df["generated_text"].str.strip().apply(self.sanitize_list_str)
        )
        texts_df["predicted_text"] = pred_raw_texts.apply(
            EvalDFPreprocessor.sanitize_list_str
        )
        texts_df["reference_text"] = texts_df[self.reference_text_col].str.join(", ")
        return texts_df

    @classmethod
    def get_task_list(cls, task_line):
        return task_line.split(", ")

    @classmethod
    def sanitize_list_str(cls, s):
        return re.sub(r'[\[\]\'"]', "", s)

    @classmethod
    def postprocess_reference_texts(cls, texts_df, reference_text_col):
        texts_df["true_tasks"].str.join(", ")
