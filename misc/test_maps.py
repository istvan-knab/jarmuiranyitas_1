from misc.load_map import MatlabMapLoader
import numpy as np


def test_map_load()->None:
    """
    :return: None
    """
    file_name = "xyz_palya"
    map = MatlabMapLoader()
    map_data = map.load(file_name = file_name)
    print(np.shape(map_data))


    return None

if __name__ == "__main__":
    test_map_load()