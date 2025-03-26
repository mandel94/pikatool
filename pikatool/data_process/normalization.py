from abc import ABC, abstractmethod
import numpy as np


class NormalizationStrategy(ABC):
    """
    Abstract base class for normalization strategies.

    Any normalization strategy should implement the `normalize` method.
    """

    @abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the input NumPy array.

        :param data: Input array to be normalized.
        :type data: np.ndarray
        :return: Normalized array.
        :rtype: np.ndarray
        """
        pass


class MinMaxNormalizationStrategy(NormalizationStrategy):
    """
    Implements Min-Max normalization.

    Scales the input array into a specified range [min_value, max_value].

    **Example Usage**::

        import numpy as np
        data = np.array([10, 20, 30, 40, 50])
        normalizer = MinMaxNormalizationStrategy()
        normalized_data = normalizer.normalize(data, 0, 1)
        print(normalized_data)  # Output: [0. , 0.25, 0.5 , 0.75, 1. ]
    """

    def normalize(
        self, data: np.ndarray, min_value: int = 0, max_value: int = 1
    ) -> np.ndarray:
        """
        Apply Min-Max normalization to a NumPy array.

        :param data: Input array to be normalized.
        :type data: np.ndarray
        :param min_value: Desired minimum value after normalization, defaults to 0.
        :type min_value: int, optional
        :param max_value: Desired maximum value after normalization, defaults to 1.
        :type max_value: int, optional
        :return: Normalized array with values scaled between min_value and max_value.
        :rtype: np.ndarray

        **Edge Case Handling**:
            - If all values in the input array are identical, it returns an array filled with `min_value`.

        **Example Usage**::

            import numpy as np
            data = np.array([5, 15, 25, 35, 45])
            normalizer = MinMaxNormalizationStrategy()
            normalized_data = normalizer.normalize(data, 0, 1)
            print(normalized_data)  # Output: [0. , 0.25, 0.5 , 0.75, 1. ]

        """
        array_min = np.min(data)
        array_max = np.max(data)

        # Avoid division by zero in case of constant array
        if array_max == array_min:
            return np.full_like(data, min_value, dtype=float)

        return ((data - array_min) / (array_max - array_min)) * (
            max_value - min_value
        ) + min_value


class Normalizer:
    """
    Context class that applies a given normalization strategy.

    :param strategy: Normalization strategy to use.
    :type strategy: NormalizationStrategy

    **Example Usage**::

        import numpy as np
        data = np.array([5, 15, 25, 35, 45])
        normalizer = Normalizer(MinMaxNormalizationStrategy())
        normalized_data = normalizer.normalize(data)
        print(normalized_data)  # Output: [0. , 0.25, 0.5 , 0.75, 1. ]
    """

    def __init__(self, strategy: NormalizationStrategy):
        """
        Initialize the normalizer with a given strategy.

        :param strategy: Normalization strategy to be used.
        :type strategy: NormalizationStrategy
        """
        self.strategy = strategy

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using the selected strategy.

        :param data: Input array to be normalized.
        :type data: np.ndarray
        :return: Normalized data.
        :rtype: np.ndarray
        """
        return self.strategy.normalize(data)
