import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.stats import kuiper
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.utils import resample

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'charts'))

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

FILE_NAME = os.path.join(DATA_DIR, 'billion-only-primes&pseudo.zip')
PLOT_NOISE_PATH = os.path.join(CHARTS_DIR, 'ransac_lines_with_noise.png')
PLOT_CLEAN_PATH = os.path.join(CHARTS_DIR, 'ransac_lines_clean.png')

MAX_LINES_TO_FIND = 5
MIN_POINTS_IN_LINE = 15
RESIDUAL_THRESHOLD = 0.8

BG_COLOR = 'white'
TEXT_COLOR = 'black'
NOISE_COLOR = '#8c8c8c'
LINE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']


def run_kuiper_test(df_pseudo):
    print("Тест Кейпера")
    angles = df_pseudo['angle_rad'].values

    def uniform_cdf(phi):
        return phi / (2 * np.pi)

    kuiper_stat, p_value = kuiper(angles, uniform_cdf)
    print(f"kuiper_statistic: {kuiper_stat:.4f}")
    print(f"p_value: {p_value:.6g}")

    if p_value < 0.05:
        print("Распределение не случайно\n")
    else:
        print("Распределение случайно\n")


def run_ransac_analysis(df_pseudo):
    print("Анализ RANSAC")
    remaining = df_pseudo.copy()
    max_inliers = 0
    lines_data = []

    for i in range(MAX_LINES_TO_FIND):
        if len(remaining) < MIN_POINTS_IN_LINE:
            break

        X = remaining[['x_coord']].values
        y = remaining['y_coord'].values
        ransac_y = RANSACRegressor(residual_threshold=RESIDUAL_THRESHOLD, max_trials=50000, random_state=42)
        ransac_y.fit(X, y)
        mask_y = ransac_y.inlier_mask_
        count_y = mask_y.sum()
        ransac_x = RANSACRegressor(residual_threshold=RESIDUAL_THRESHOLD, max_trials=50000, random_state=42)
        ransac_x.fit(remaining[['y_coord']].values, remaining['x_coord'].values)
        mask_x = ransac_x.inlier_mask_
        count_x = mask_x.sum()
        is_vertical = count_x > count_y

        if not is_vertical and count_y >= MIN_POINTS_IN_LINE:
            mask = mask_y
            m = ransac_y.estimator_.coef_[0]
            c = ransac_y.estimator_.intercept_
            sign = "+" if c >= 0 else "-"
            equation = f"y = {m:.2f}x {sign} {abs(c):.2f}"
            line_x = np.array([X[mask].min(), X[mask].max()])
            line_y = ransac_y.predict(line_x.reshape(-1, 1))

        elif is_vertical and count_x >= MIN_POINTS_IN_LINE:
            mask = mask_x
            m = ransac_x.estimator_.coef_[0]
            c = ransac_x.estimator_.intercept_
            sign = "+" if c >= 0 else "-"
            equation = f"x = {m:.2f}y {sign} {abs(c):.2f}"
            line_y = np.array([y[mask].min(), y[mask].max()])
            line_x = ransac_x.predict(line_y.reshape(-1, 1))
        else:
            break

        inlier_points = remaining[mask]
        inliers_count = len(inlier_points)

        if inliers_count > max_inliers:
            max_inliers = inliers_count

        print(f"Прямая {i + 1}: Найдено точек: {inliers_count}. Уравнение: {equation}")
        slopes = []

        for _ in range(50):
            sample = resample(inlier_points, replace=True)

            if len(sample) < 2:
                continue

            lr = LinearRegression()

            if is_vertical:
                lr.fit(sample[['y_coord']].values, sample['x_coord'].values)
            else:
                lr.fit(sample[['x_coord']].values, sample['y_coord'].values)

            slopes.append(lr.coef_[0])

        ci_lower, ci_upper = np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)
        print(f"Диапазон наклона: [{ci_lower:.4f}, {ci_upper:.4f}]")
        color = LINE_COLORS[i % len(LINE_COLORS)]
        lines_data.append({
            'pts_x': inlier_points['x_coord'],
            'pts_y': inlier_points['y_coord'],
            'line_x': line_x,
            'line_y': line_y,
            'color': color,
            'label': equation
        })
        remaining = remaining[~mask]

    print("Сохранение графиков...")

    def setup_and_save_plot(show_noise, path, title):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor(BG_COLOR)
        fig.patch.set_facecolor(BG_COLOR)

        if show_noise:
            ax.scatter(df_pseudo['x_coord'], df_pseudo['y_coord'], color=NOISE_COLOR, s=15, label='Псевдопростые числа',
                       zorder=1)

        for line in lines_data:
            ax.scatter(line['pts_x'], line['pts_y'], color=line['color'], s=30, zorder=3)
            ax.plot(line['line_x'], line['line_y'], color=line['color'], linewidth=2.5, linestyle='--', zorder=4,
                    label=line['label'])

        ax.set_title(title, color=TEXT_COLOR, fontsize=16, pad=15)
        ax.set_xlabel("X координата", color=TEXT_COLOR)
        ax.set_ylabel("Y координата", color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, color='#e0e0e0', linestyle='--', zorder=0)

        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(facecolor='white', edgecolor='#cccccc', labelcolor=TEXT_COLOR, loc='center left',
                      bbox_to_anchor=(1.02, 0.5), fontsize=11)

        ax.axis('equal')
        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    setup_and_save_plot(show_noise=True, path=PLOT_NOISE_PATH, title="Диагонали RANSAC")
    setup_and_save_plot(show_noise=False, path=PLOT_CLEAN_PATH, title="Диагонали RANSAC")
    print("Графики сохранены\n")

    return max_inliers


def run_monte_carlo_test(df_pseudo, observed_inliers, num_simulations=50):
    print("Тест Монте-Карло")
    num_points = len(df_pseudo)
    min_x, max_x = df_pseudo['x_coord'].min(), df_pseudo['x_coord'].max()
    min_y, max_y = df_pseudo['y_coord'].min(), df_pseudo['y_coord'].max()
    sim_max_inliers = []

    for _ in range(num_simulations):
        sim_x = np.random.uniform(min_x, max_x, num_points)
        sim_y = np.random.uniform(min_y, max_y, num_points)
        ransac_y = RANSACRegressor(residual_threshold=RESIDUAL_THRESHOLD, max_trials=1000)
        ransac_y.fit(sim_x.reshape(-1, 1), sim_y)
        in_y = ransac_y.inlier_mask_.sum()
        ransac_x = RANSACRegressor(residual_threshold=RESIDUAL_THRESHOLD, max_trials=1000)
        ransac_x.fit(sim_y.reshape(-1, 1), sim_x)
        in_x = ransac_x.inlier_mask_.sum()
        sim_max_inliers.append(max(in_y, in_x))

    p_value = (np.count_nonzero(np.array(sim_max_inliers) >= observed_inliers) + 1) / (num_simulations + 1)
    print(f"p_value: {p_value:.4f}")

    if p_value < 0.05:
        print("Линии статистически значимы\n")
    else:
        print("Линии статистически не значимы\n")


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"Ошибка: Файл {FILE_NAME} не найден.")
    else:
        dataframe = pd.read_csv(FILE_NAME)
        df_pseudo = dataframe[dataframe['is_pseudo'] == 1].copy()
        print(f"Псевдопростые числа: {len(df_pseudo)}\n")
        run_kuiper_test(df_pseudo)
        max_inliers = run_ransac_analysis(df_pseudo)

        if max_inliers > 0:
            run_monte_carlo_test(df_pseudo, max_inliers, num_simulations=50)
