import scipy
import numpy as np
from abc import ABC, abstractmethod


class MapLoader(ABC):
    @abstractmethod
    def load(self, file_name: str) -> np.ndarray:
        """
        This function creates a standard numpy array from the map data given as an argument.
        :param file_name: name of the map file to be preprocessed, excluding extension
        :return: numpy array of size (number_of_coordinates, 2) where each row corresponds to an (x, y) coordinate pair
        """
        pass


class MatlabMapLoader(MapLoader):
    def load(self, file_name: str) -> np.ndarray:
        map_data = scipy.io.loadmat("maps/" + file_name + '.mat')
        map_data = map_data['xyz']
        map_data = np.delete(map_data, (2, 3), 1)

        return