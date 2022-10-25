from misc import load_map
import numpy as np


def test_map_load()->None:
    """

    :return: None
    """
    file_name = "xyz_palya"
    map = load_map.MatlabMapLoader()
    map_data = map.load(file_name = file_name)
    print(np.shape(map_data))
    print("The refactored array is:  " + map_data)

    return None

if __name__ == "__main__":
    test_map_load()