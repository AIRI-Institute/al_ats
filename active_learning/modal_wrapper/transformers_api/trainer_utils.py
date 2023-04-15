from typing import NamedTuple, Union, Tuple
import numpy as np


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    inputs: np.ndarray
