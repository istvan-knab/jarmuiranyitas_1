import scipy
import numpy as np


class MapLoader:

    def __init__(self, file_name: str) -> None:
        """

        :param file_name:
        """
        self._file_name = file_name

    def load(self) -> np.ndarray:
        """

        :return:
        """
        map_data = scipy.io.loadmat("maps/" + self._file_name + '.mat')
        map_data = map_data['xyz']
        map_data = np.delete(map_data, 2, 1)

        return map_data
