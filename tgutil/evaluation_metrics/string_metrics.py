from typing import Any, List, Dict, Callable
import pandas as pd
import evaluate
from pydantic import BaseModel, Field

import nltk
import torch
import sentence_transformers
import abc
from mlutil.feature_extraction.embeddings import load_gensim_embedding_model
from enum import Enum
from tgutil.evaluation_metrics.base import TextGenerationMetric
from tgutil.type_utils import StrEnum

StringComparisonFunction = Callable[[str, str], float]


class StringDistanceType(StrEnum):
    edit = "edit"
    jaccard = "jaccard"
    custom = "custom"


class SplitType(StrEnum):
    """whether to split on words or assume string is a comma separated list"""

    none = "none"
    word = "word"
    lst = "lst"


class StringMetricType(BaseModel):
    distance_type: StringDistanceType
    split_type: SplitType

    @classmethod
    def load_from_str(cls, metric_name):
        distance_type, split_type = metric_name.split("_")
        return StringMetricType(distance_type=distance_type, split_type=split_type)

    @classmethod
    def is_string_metric(cls, metric_name):
        splits = metric_name.split("_")
        is_twopart = len(splits) == 2
        if not is_twopart:
            return False
        is_distance_valid = StringDistanceType.contains(splits[0])
        is_split_valid = SplitType.contains(splits[1])
        return is_distance_valid and is_split_valid

    def get_normalization_factor(self, ref, pred):
        if self.distance_type == StringDistanceType.edit:
            return max(len(pred), len(ref))
        else:
            return 1


class StringBasedMetric(TextGenerationMetric, BaseModel):
    metric_type: StringMetricType
    comparison_function: StringComparisonFunction

    @classmethod
    def load(cls, metric_name, split_type=SplitType.word):
        metric_type = StringMetricType.load_from_str(metric_name)
        if metric_type.distance_type == StringDistanceType.edit:
            comparison_function = nltk.edit_distance
        elif metric_type.distance_type == StringDistanceType.jaccard:
            comparison_function = cls._jaccard
        return StringBasedMetric(
            metric_type=metric_type,
            comparison_function=comparison_function,
        )

    @property
    def name(self):
        return f"{self.metric_type.distance_type}_{self.metric_type.split_type}"

    def split_texts(self, texts):
        if self.metric_type.split_type == SplitType.word:
            return texts.str.replace(",", " ").str.split()
        elif self.metric_type.split_type == SplitType.lst:
            return texts.str.split(",").apply(lambda lst: [s.strip() for s in lst])
        else:
            return texts

    def evaluate(
        self,
        texts_df: pd.DataFrame,
        reference_text_col: str,
        predicted_text_col: str,
    ):
        reference_texts = self.split_texts(texts_df[reference_text_col])
        predicted_texts = self.split_texts(texts_df[predicted_text_col])

        scores = [
            self.get_score(ref, pred)
            for ref, pred in zip(reference_texts, predicted_texts)
        ]
        scores_pd = pd.Series(scores, name=self.name)
        scores_pd.index = texts_df.index

        return scores_pd

    def get_score(self, ref, pred):
        return min(
            1,
            self.comparison_function(ref, pred)
            / self.metric_type.get_normalization_factor(ref, pred),
        )

    @classmethod
    def _jaccard(cls, s1, s2):
        s1 = set(s1)
        s2 = set(s2)
        return len(s1.intersection(s2)) / len(s1.union(s2))
