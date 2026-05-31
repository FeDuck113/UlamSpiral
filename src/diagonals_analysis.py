import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.stats import kuiper
from sklearn.linear_model import RANSACRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'charts'))

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

FILE_NAME = os.path.join(DATA_DIR, '5000000-spiral.csv')
PLOT_1_PATH = os.path.join(CHARTS_DIR, 'plot_1_ransac_lines.png')

MAX_LINES_TO_FIND = 5
MIN_POINTS_IN_LINE = 5

BG_COLOR = 'white'
TEXT_COLOR = 'black'
NOISE_COLOR = '#8c8c8c'
LINE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']


def run_kuiper_test(pseudos_dataframe):
    print("Тест Кейпера")
    angles = pseudos_dataframe['angle_rad'].values

    def theoretical_expected_angle_density(phi):
        return phi / (2 * np.pi)

    kuiper_test_statistic, randomness_probability = kuiper(angles, theoretical_expected_angle_density)
    print(f"Kuiper_test_statistic: {kuiper_test_statistic:.4f}")
    print(f"Randomness_probability (p-value): {randomness_probability}")

    if randomness_probability < 0.05:
        print("Итог: p-value < 0.05. Псевдопростые числа распределены не случайно!\n")
    else:
        print("Итог: p-value >= 0.05. Псевдопростые числа распределены случайно!\n")


def run_ransac_analysis(pseudos_dataframe):
    print("Алгоритм RANSAC (Поиск диагоналей и вертикалей)")
    remaining_points = pseudos_dataframe.copy()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.scatter(pseudos_dataframe['x_coord'], pseudos_dataframe['y_coord'], color=NOISE_COLOR, s=15,
               label='Остальные псевдопростые', zorder=1)

    for i in range(MAX_LINES_TO_FIND):
        if len(remaining_points) < MIN_POINTS_IN_LINE:
            break

        X_coords = remaining_points[['x_coord']].values
        y_coords = remaining_points['y_coord'].values

        ransac_y = RANSACRegressor(residual_threshold=0.5, max_trials=50000, random_state=42)
        ransac_y.fit(X_coords, y_coords)
        mask_y = ransac_y.inlier_mask_
        inliers_count_y = mask_y.sum()

        ransac_x = RANSACRegressor(residual_threshold=0.5, max_trials=50000, random_state=42)
        ransac_x.fit(remaining_points[['y_coord']].values, remaining_points['x_coord'].values)
        mask_x = ransac_x.inlier_mask_
        inliers_count_x = mask_x.sum()

        if inliers_count_y >= inliers_count_x and inliers_count_y >= MIN_POINTS_IN_LINE:
            mask = mask_y
            m = ransac_y.estimator_.coef_[0]
            c = ransac_y.estimator_.intercept_
            sign = "+" if c >= 0 else "-"
            equation_str = f"y = {m:.2f}x {sign} {abs(c):.2f}"
            line_x = np.array([X_coords[mask].min(), X_coords[mask].max()])
            line_y = ransac_y.predict(line_x.reshape(-1, 1))

        elif inliers_count_x > inliers_count_y and inliers_count_x >= MIN_POINTS_IN_LINE:
            mask = mask_x
            m = ransac_x.estimator_.coef_[0]
            c = ransac_x.estimator_.intercept_
            sign = "+" if c >= 0 else "-"
            equation_str = f"x = {m:.2f}y {sign} {abs(c):.2f}"
            line_y = np.array([y_coords[mask].min(), y_coords[mask].max()])
            line_x = ransac_x.predict(line_y.reshape(-1, 1))
        else:
            break

        line_pts = remaining_points[mask]
        legend_label = f"{equation_str}  (n={len(line_pts)})"
        print(f"Прямая {i + 1}: Найдено {len(line_pts)} чисел. Уравнение: {equation_str}")
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.scatter(line_pts['x_coord'], line_pts['y_coord'], color=color, s=35, zorder=3)
        ax.plot(line_x, line_y, color=color, linewidth=2.5, linestyle='--', zorder=4, label=legend_label)
        remaining_points = remaining_points[~mask]

    print("\nОтрисовка и сохранение графика...")
    ax.set_title("Диагонали и вертикали RANSAC", color=TEXT_COLOR, fontsize=16, pad=15)
    ax.set_xlabel("X координата", color=TEXT_COLOR)
    ax.set_ylabel("Y координата", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color='#e0e0e0', linestyle='--', zorder=0)

    if len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(facecolor='white', edgecolor='#cccccc', labelcolor=TEXT_COLOR,
                  loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, title="Уравнения прямых:")

    ax.axis('equal')
    plt.tight_layout()
    fig.savefig(PLOT_1_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"График успешно сохранен: '{PLOT_1_PATH}'")


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"Ошибка: Файл {FILE_NAME} не найден.")
    else:
        dataframe = pd.read_csv(FILE_NAME)
        pseudos = dataframe[dataframe['is_pseudo'] == 1].copy()
        print(f"Найдено псевдопростых чисел: {len(pseudos)} из {len(dataframe)}\n")
        run_kuiper_test(pseudos)
        run_ransac_analysis(pseudos)
