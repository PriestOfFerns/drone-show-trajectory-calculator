import numpy as np

Position = np.ndarray


def get_center_of_array(array: np.ndarray) -> Position:
    x_min, y_min, z_min = np.min(array, 0)[0], np.min(array, 0)[1], np.min(array, 0)[2]
    x_max, y_max, z_max = np.max(array, 0)[0], np.max(array, 0)[1], np.max(array, 0)[2]

    center_x, center_y, center_z = (x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2

    return np.array((center_x, center_y, center_z))


def turn_csv_into_targets(csv_file_path: str, drone_radius: float, center_position: Position) -> np.ndarray:
    targets = np.loadtxt(csv_file_path, float, delimiter=",")
    np.multiply(targets, 2 * drone_radius)

    center = get_center_of_array(targets)
    for i, position in enumerate(targets):

        relative = center - position
        targets[i] = center_position + relative

    return targets


if __name__ == "__main__":
    targets = turn_csv_into_targets('../test_data/00.csv', 0.5, np.array([10, 10, 10]))
    print(targets)
