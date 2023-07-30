from typing import List

import pandas as pd
from pydantic import BaseModel
from tgutil.evaluation_metrics import TextGenerationMetric, load_metric


class TextGenerationEvaluator(BaseModel):
    evaluation_metrics: List[TextGenerationMetric]

    @classmethod
    def from_metric_names(
        cls,
        metric_names=["bleurt", "rouge", "wmd", "sentence_transformer_similarity"],
    ):
        metrics = [load_metric(metric_name) for metric_name in metric_names]
        return cls(evaluation_metrics=metrics)

    def get_scores(
        self,
        texts_df,
    ):
        generation_metrics_dfs = [
            metric.evaluate(texts_df, "reference_text", "predicted_text")
            for metric in self.evaluation_metrics
        ]
        return pd.concat(generation_metrics_dfs, axis=1)

    def get_df_with_added_scores(self, texts_df, stratify_val, stratify):
        group_df = texts_df[texts_df[stratify] == stratify_val]
        return pd.concat([group_df, self.get_scores(group_df)], axis=1)

    def get_evaluated_df(self, texts_df, stratify):
        stratifying_idxs = set(texts_df[stratify])
        return pd.concat(
            [
                self.get_df_with_added_scores(texts_df, stratify_val, stratify)
                for stratify_val in stratifying_idxs
            ],
            axis=1,
        )

    class Config:
        arbitrary_types_allowed = True
