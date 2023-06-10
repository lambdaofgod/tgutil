import pandas as pd
import re


class EvalDFPreprocessor:
    """
    utils for loading data for generation evaluation
    """

    @classmethod
    def get_eval_df_from_raw_generated_text(
        cls, generated_text_df: pd.DataFrame, repo_tasks_df: pd.DataFrame
    ):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        # task_list_df = generated_text_df["predicted_text"].apply(get_task_list)
        repo_tasks_df = repo_tasks_df.rename({"tasks": "true_tasks"}, axis=1)[
            ["repo", "true_tasks"]
        ]
        evaluation_df = cls.prepare_texts_for_scoring(
            generated_text_df.merge(repo_tasks_df, on="repo")
        )
        return evaluation_df

    @classmethod
    def get_eval_dffrom_generated_text(cls, generated_text_df: pd.DataFrame):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        return cls.prepare_texts_for_scoring(generated_text_df)

    @classmethod
    def prepare_texts_for_scoring(cls, texts_df):
        pred_raw_texts = (
            texts_df["generated_text"].str.strip().apply(cls.sanitize_list_str)
        )
        texts_df["predicted_text"] = pred_raw_texts.apply(
            EvalDFPreprocessor.sanitize_list_str
        )
        texts_df["reference_text"] = texts_df["true_tasks"].str.join(", ")
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
