from typing import Any, List, Dict, Callable
import pandas as pd
import evaluate
from pydantic import BaseModel, Field
import string
import nltk
import torch
import sentence_transformers
import abc
from mlutil.feature_extraction.embeddings import load_gensim_embedding_model
from enum import Enum
from tgutil.utils import _strip_punctuation


class MetricsKwargs:
    kwargs = {
        "sentence_transformer_similarity": {
            "model_name": "paraphrase-distilroberta-base-v1"
        },
        "wmd": {"model_name": "glove-wiki-gigaword-100"},
        "rouge": {"use_aggregator": False},
        "bleurt": {},
    }


class HuggingfaceMetric(TextGenerationMetric, BaseModel):
    name: str
    hf_metric: evaluate.EvaluationModule
    metric_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def evaluate(
        self,
        texts_df: pd.DataFrame,
        reference_text_col: str,
        predicted_text_col: str,
    ):
        metric_kwargs = MetricsKwargs.kwargs.get(self.name, dict())
        reference_texts = texts_df[reference_text_col].apply(_strip_punctuation)
        predicted_texts = texts_df[predicted_text_col].apply(_strip_punctuation)
        scores = self.hf_metric.compute(
            predictions=predicted_texts.to_list(),
            references=reference_texts.to_list(),
            **metric_kwargs,
        )
        if len(scores.keys()) == 1:
            values = list(scores.values())[0]
            scores_pd = pd.Series(values, name=self.name)
        else:
            scores_pd = pd.DataFrame(scores)
        scores_pd.index = texts_df.index
        return scores_pd

    @staticmethod
    def load(metric_name, metric_kwargs):
        return HuggingfaceMetric(
            name=metric_name,
            hf_metric=evaluate.load(metric_name),
            metric_kwargs=metric_kwargs,
        )

    class Config:
        arbitrary_types_allowed = True


class SentenceTransformersMetric(TextGenerationMetric, BaseModel):
    sentence_transformer: sentence_transformers.SentenceTransformer

    def evaluate(self, texts_df, reference_text_col, predicted_text_col):
        reference_texts = texts_df[reference_text_col].tolist()
        predicted_texts = texts_df[predicted_text_col].tolist()
        reference_embeddings = self.sentence_transformer.encode(
            reference_texts, convert_to_tensor=True
        )
        predicted_embeddings = self.sentence_transformer.encode(
            predicted_texts, convert_to_tensor=True
        )
        with torch.no_grad():
            similarities = (
                torch.nn.CosineSimilarity()(reference_embeddings, predicted_embeddings)
                .cpu()
                .numpy()
            )
        return pd.DataFrame(
            {"sentence_transformer_similarity": similarities}, index=texts_df.index
        )

    @staticmethod
    def load(model_name):
        return SentenceTransformersMetric(
            sentence_transformer=sentence_transformers.SentenceTransformer(model_name)
        )

    class Config:
        arbitrary_types_allowed = True
