from typing import Any
import pandas as pd
from pydantic import BaseModel, Field

import torch
import sentence_transformers
from mlutil.feature_extraction.embeddings import load_gensim_embedding_model
from tgutil.evaluation_metrics.base import TextGenerationMetric
from tgutil.utils import _strip_punctuation


class WMDMetric(TextGenerationMetric, BaseModel):
    gensim_model: Any  # Zeus only knows what will this model be in a new gensim version
    name: str = Field(default="wmd")

    def evaluate(self, texts_df, reference_text_col, predicted_text_col):
        wmdistances = []
        for pred_text, ref_text in zip(
            texts_df[reference_text_col], texts_df[predicted_text_col]
        ):
            pred_text = _strip_punctuation(pred_text).split()
            ref_text = _strip_punctuation(ref_text).split()
            distance = self.gensim_model.wmdistance(pred_text, ref_text)
            wmdistances.append(distance)
        return pd.DataFrame({"wmd": wmdistances}, index=texts_df.index)

    @staticmethod
    def load(model_name="fasttext-wiki-news-subwords-300"):
        model = load_gensim_embedding_model(model_name)
        return WMDMetric(gensim_model=model)

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
