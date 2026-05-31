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
FILE_NAME = os.path.join(DATA_DIR, '1000000-spiral.csv')
PLOT_1_PATH = os.path.join(DATA_DIR, 'plot_1_ransac_lines.png')


MAX_LINES_TO_FIND = 5
MIN_POINTS_IN_LINE = 5


BG_COLOR = 'white'
TEXT_COLOR = 'black'
NOISE_COLOR = '#d3d3d3'
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
    print("Алгоритм RANSAC")
    remaining_points = pseudos_dataframe.copy()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.scatter(pseudos_dataframe['x_coord'], pseudos_dataframe['y_coord'], color=NOISE_COLOR, s=15,
               label='Остальные псевдопростые')

    for i in range(MAX_LINES_TO_FIND):
        if len(remaining_points) < MIN_POINTS_IN_LINE:
            break

        X = remaining_points[['x_coord']].values
        y = remaining_points['y_coord'].values
        ransac = RANSACRegressor(residual_threshold=0.5, random_state=42)
        ransac.fit(X, y)
        mask = ransac.inlier_mask_
        line_pts = remaining_points[mask]

        if len(line_pts) < MIN_POINTS_IN_LINE:
            break

        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        sign = "+" if c >= 0 else "-"
        equation_str = f"y = {m:.2f}x {sign} {abs(c):.2f}"
        legend_label = f"{equation_str}  (n={len(line_pts)})"
        print(f"Диагональ {i + 1}: Найдено {len(line_pts)} чисел. Уравнение: {equation_str}")
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.scatter(line_pts['x_coord'], line_pts['y_coord'], color=color, s=30, zorder=3)
        line_x = np.array([X[mask].min(), X[mask].max()])
        line_y = ransac.predict(line_x.reshape(-1, 1))
        ax.plot(line_x, line_y, color=color, linewidth=2.5, linestyle='--', zorder=4, label=legend_label)
        remaining_points = remaining_points[~mask]

    print("\nОтрисовка и сохранение графика...")
    ax.set_title("Диагонали RANSAC", color=TEXT_COLOR, fontsize=16, pad=15)
    ax.set_xlabel("X координата", color=TEXT_COLOR)
    ax.set_ylabel("Y координата", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color='#e0e0e0', linestyle='--')
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
