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

    def get_evaluated_df(self, texts_df):
        return pd.concat(
            [texts_df, self.get_scores(texts_df)],
            axis=1,
        )

    class Config:
        arbitrary_types_allowed = True
