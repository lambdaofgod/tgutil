from tgutil.evaluation_metrics.base import TextGenerationMetric
from tgutil.evaluation_metrics.string_metrics import (
    StringBasedMetric,
    StringMetricType,
)
from tgutil.evaluation_metrics.other_metrics import (
    WMDMetric,
    SentenceTransformersMetric,
)
from tgutil.evaluation_metrics.hf_metrics import (
    HuggingfaceMetric,
    HuggingfaceMetricName,
)


class OtherMetricsKwargs:
    kwargs = {
        "sentence_transformer_similarity": {
            "model_name": "paraphrase-distilroberta-base-v1"
        },
        "wmd": {"model_name": "glove-twitter-25"},
    }


class MetricsKwargs:
    kwargs = {**OtherMetricsKwargs.kwargs, **HuggingfaceMetric.Config.kwargs}


def is_allowed_metric(metric_name):
    is_string_metric = StringMetricType.is_string_metric(metric_name)
    is_hf_metric = HuggingfaceMetricName.contains(metric_name)
    is_other_metric = metric_name in OtherMetricsKwargs.kwargs.keys()
    return is_string_metric or is_hf_metric or is_other_metric


def load_metric(metric_name, **metric_kwargs):
    assert is_allowed_metric(metric_name), f"unsupported metric: {metric_name}"
    if StringMetricType.is_string_metric(metric_name):
        return StringBasedMetric.load(metric_name)
    elif metric_name == "wmd":
        model_name = metric_kwargs.get(
            "model_name", OtherMetricsKwargs.kwargs["wmd"]["model_name"]
        )
        return WMDMetric.load(model_name=model_name)
    elif metric_name == "sentence_transformer_similarity":
        model_name = metric_kwargs.get(
            "model_name",
            OtherMetricsKwargs.kwargs["sentence_transformer_similarity"]["model_name"],
        )
        return SentenceTransformersMetric.load(model_name)
    else:
        return HuggingfaceMetric.load(metric_name, metric_kwargs)
