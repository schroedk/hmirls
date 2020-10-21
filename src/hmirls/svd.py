from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy


class SVDEngine(ABC):
    @staticmethod
    @abstractmethod
    def svd(x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.array, np.ndarray]:
        pass


class ScipySVDEngine(SVDEngine):
    @staticmethod
    def svd(x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.array, np.ndarray]:
        left_sing_vec, sing_val, right_sing_vec_herm = scipy.linalg.svd(
            x, full_matrices=True
        )
        return left_sing_vec, sing_val, right_sing_vec_herm.conj().transpose()


class NumpySVDEngine(SVDEngine):
    @staticmethod
    def svd(x: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.array, np.ndarray]:
        pass
