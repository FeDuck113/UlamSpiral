import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import kuiper
from sklearn.linear_model import RANSACRegressor

FILE_NAME = 'data/1000000-spiral.csv'
MAX_LINES_TO_FIND = 5
MIN_POINTS_IN_LINE = 5


def run_kuiper_test(pseudos_dataframe):
    print("Тест Кейпера")
    angles = pseudos_dataframe['angle_rad'].values

    def theoretical_expected_angle_density(phi):
        return phi / (2 * np.pi)

    kuiper_test_statistic, randomness_probability = kuiper(angles, theoretical_expected_angle_density)

    print(f"Kuiper_test_statistic: {kuiper_test_statistic:.4f}")
    print(f"Randomness_probability: {randomness_probability}")

    if randomness_probability < 0.05:
        print("Итог: randomness_probability < 0.05. Псевдопростые числа распределены не случайно!\n")
    else:
        print("Итог: randomness_probability >= 0.05. Псевдопростые числа распределены случайно!\n")


def run_ransac_analysis(pseudos_dataframe):
    print("Алгоритм RANSAC")
    remaining_points = pseudos_dataframe.copy()
    found_lines = []

    plt.figure(figsize=(10, 10))
    plt.scatter(pseudos_dataframe['x_coord'], pseudos_dataframe['y_coord'], color='lightgrey', s=10,
                label='Остальные псевдопростые')

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i in range(MAX_LINES_TO_FIND):
        if len(remaining_points) < MIN_POINTS_IN_LINE:
            break

        x_coordinates_input = remaining_points[['x_coord']].values
        y_coordinates_target = remaining_points['y_coord'].values

        ransac = RANSACRegressor(residual_threshold=0.5, random_state=42)
        ransac.fit(x_coordinates_input, y_coordinates_target)

        inlier_mask = ransac.inlier_mask_
        line_points = remaining_points[inlier_mask]
        if len(line_points) < MIN_POINTS_IN_LINE:
            break

        line_slope_coefficient = ransac.estimator_.coef_[0]
        line_vertical_offset_intercept = ransac.estimator_.intercept_

        print(f"Диагональ {i + 1}: Найдено {len(line_points)} псевдопростых чисел.")
        print(f"Уравнение прямой: y = {line_slope_coefficient:.2f}*x + {line_vertical_offset_intercept:.2f}")

        color = colors[i % len(colors)]
        plt.scatter(line_points['x_coord'], line_points['y_coord'], color=color, s=20, label=f'Линия {i + 1}')

        line_x = np.array([x_coordinates_input[inlier_mask].min(), x_coordinates_input[inlier_mask].max()])
        line_y = ransac.predict(line_x.reshape(-1, 1))

        plt.plot(line_x, line_y, color=color, linewidth=2, linestyle='--')
        remaining_points = remaining_points[~inlier_mask]
        found_lines.append(line_points)

    print("\nПостроение графика...")
    plt.title("RANSAC: Главные диагонали псевдопростых чисел")
    plt.xlabel("x координата")
    plt.ylabel("y координата")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    dataframe = pd.read_csv(FILE_NAME)
    pseudos = dataframe[dataframe['is_pseudo'] == 1].copy()
    print(f"Найдено псевдопростых чисел: {len(pseudos)} из {len(dataframe)}\n")
    run_kuiper_test(pseudos)
    run_ransac_analysis(pseudos)
