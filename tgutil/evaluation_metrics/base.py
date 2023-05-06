import abc
import pandas as pd


class TextGenerationMetric(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self, texts_df: pd.DataFrame, reference_text_col: str, predicted_text_col: str
    ):
        pass

    @abc.abstractstaticmethod
    def load(metric_name: str):
        pass
