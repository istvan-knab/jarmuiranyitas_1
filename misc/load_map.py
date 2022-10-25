import scipy
import numpy as np
from abc import ABC, abstractmethod


class MapLoader(ABC):
    @abstractmethod
    def load(self, file_name) -> np.ndarray:
        """

        :param file_name:
        :return:
        """
        pass


class MatlabMapLoader(MapLoader):
    def load(self, file_name) -> np.ndarray:
        map_data = scipy.io.loadmat("maps/" + file_name + '.mat')
        map_data = map_data['xyz']
        map_data = np.delete(map_data, 2, 1)

        return map_data
