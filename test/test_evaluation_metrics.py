from tgutil.evaluation.evaluators import TextGenerationEvaluator
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def texts_df():
    reference_texts = [
        "model selection",
        "collaborative filtering, model selection",
        "fairness, node classification, general classification",
    ]
    predicted_texts = [
        "model selection1",
        "collaborative filtering",
        "text classification, graph neural network, fairness, general classification",
    ]
    return pd.DataFrame(
        {"reference_text": reference_texts, "predicted_text": predicted_texts}
    )


def run_metrics_test(metric_names, texts_df):
    return TextGenerationEvaluator.from_metric_names(metric_names).get_evaluated_df(
        texts_df
    )


def test_string_metrics(texts_df):
    scores = run_metrics_test(["edit_none", "jaccard_word", "jaccard_lst"], texts_df)
    assert np.isclose(scores["edit_none"].iloc[0], 1 / len("model selections"))
    assert np.isclose(
        scores["jaccard_word"].values, np.array([1 / 3, 1 / 2, 3 / 8])
    ).all()
    assert np.isclose(scores["jaccard_lst"].values, np.array([0, 1 / 2, 0.4])).all()


def test_hf_metrics(texts_df):
    """
    there are two words and one is matched so rouge is 1/2
    """
    tested_df = texts_df.iloc[:1]
    scores = run_metrics_test(["rouge"], tested_df)
    assert np.isclose(scores["rouge1"].iloc[0], 1 / 2)


@pytest.mark.slow
def test_wmd_metric(texts_df):
    """
    wmd uses fasttext so "selection" "selection1" have the same embeddings
    """
    tested_df = texts_df.iloc[:1]
    scores = run_metrics_test(["wmd"], texts_df)
    assert np.isclose(scores["wmd"].iloc[0], 0)


@pytest.mark.slow
def test_sentence_transformer_metric(texts_df):
    scores = run_metrics_test(["sentence_transformer_similarity"], texts_df)
    assert np.isclose(
        scores["sentence_transformer_similarity"].iloc[0], 0.79, rtol=0.025
    )
