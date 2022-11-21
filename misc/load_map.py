from scipy.io import loadmat
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


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
        map_data = loadmat(file_name + '.mat')
        map_data = map_data['xyz']
        map_data = np.delete(map_data, (2, 3), 1)

        return map_data


def make_offset_polygon(old_x, old_y, offset, outer_ccw=1):
    num_points = len(old_x)
    new_x = []
    new_y = []

    for curr in range(num_points):
        prev_point = (curr + num_points - 1) % num_points
        next_point = (curr + 1) % num_points

        vn_x = old_x[next_point] - old_x[curr]
        vn_y = old_y[next_point] - old_y[curr]

        norm = np.sqrt(vn_x**2+vn_y**2)
        vnn_x = vn_x / norm
        vnn_y = vn_y / norm
        nnn_x = vnn_y
        nnn_y = -vnn_x

        vp_x = old_x[curr] - old_x[prev_point]
        vp_y = old_y[curr] - old_y[prev_point]
        norm = np.sqrt(vp_x ** 2 + vp_y ** 2)
        vpn_x = vp_x / norm
        vpn_y = vp_y / norm
        npn_x = vpn_y * outer_ccw
        npn_y = -vpn_x * outer_ccw

        bis_x = (nnn_x + npn_x) * outer_ccw
        bis_y = (nnn_y + npn_y) * outer_ccw

        norm = np.sqrt(bis_x ** 2 + bis_y ** 2)
        bisn_x = bis_x / norm
        bisn_y = bis_y / norm
        bislen = offset / np.sqrt((1 + nnn_x*npn_x + nnn_y*npn_y)/2)

        new_x.append(old_x[curr] + bislen * bisn_x)
        new_y.append(old_y[curr] + bislen * bisn_y)

    return new_x, new_y


def main():
    matlab_map_loader = MatlabMapLoader()
    map_data = matlab_map_loader.load('maps/xyz_palya')

    x_centerline = np.reshape(map_data[:, 0], (-1, 1))
    y_centerline = np.reshape(map_data[:, 1], (-1, 1))

    plt.figure()
    x_inner_boundary, y_inner_boundary = make_offset_polygon(x_centerline, y_centerline, 0.5, 1)
    plt.plot(x_inner_boundary, y_inner_boundary, "-k")
    x_outer_boundary, y_outer_boundary = make_offset_polygon(x_centerline, y_centerline, -0.5, 1)
    plt.plot(x_outer_boundary, y_outer_boundary, "-k")
    plt.axis('off')
    plt.savefig('xyz_palya.png', dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
