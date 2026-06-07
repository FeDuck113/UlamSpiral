import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.spatial.distance import pdist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'charts'))

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

FILE_NAME = os.path.join(DATA_DIR, 'billion-only-primes&pseudo.zip')
PLOT_DENSITY_PATH = os.path.join(CHARTS_DIR, 'hexbin_density.png')

BG_COLOR = 'white'
TEXT_COLOR = 'black'
CMAP_THEME = 'plasma'


def run_spatial_randomness_test(df_pseudo):
    print("Тест индекса дисперсии")
    hist, _, _ = np.histogram2d(df_pseudo['x_coord'], df_pseudo['y_coord'], bins=50)
    counts = hist.flatten()
    mean = np.mean(counts)
    variance = np.var(counts)
    vmr = variance / (mean + 1e-9)
    n = len(counts)
    chi2_stat = (n - 1) * vmr
    p_value = 1 - chi2.cdf(chi2_stat, n - 1)
    print(f"vmr: {vmr:.4f}")
    print(f"p_value: {p_value:.6g}")

    if p_value < 0.05 and vmr > 1.5:
        print("Распределение не случайно\n")
    else:
        print("Распределение случайно\n")


def run_maup_test(df_pseudo):
    print("Тест MAUP")
    gridsizes = [40, 70, 100]
    hotspots = []
    fig, ax = plt.subplots()

    for g in gridsizes:
        hb = ax.hexbin(df_pseudo['x_coord'], df_pseudo['y_coord'], gridsize=g)
        counts = hb.get_array()
        offsets = hb.get_offsets()

        max_idx = np.argmax(counts)
        hotspot_coords = offsets[max_idx]
        hotspots.append(hotspot_coords)

    plt.close(fig)
    max_drift = np.max(pdist(hotspots))

    for g, coords in zip(gridsizes, hotspots):
        print(f"Сетка {g}x{g}: X={coords[0]:.0f}, Y={coords[1]:.0f}")

    print(f"Максимальное смещение: {max_drift:.1f}")

    if max_drift < 1500:
        print("Устойчивость высокая\n")
    else:
        print("Устойчивость низкая\n")


def generate_density_plot(df_pseudo):
    print("Построение карты плотности...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    hb = ax.hexbin(df_pseudo['x_coord'], df_pseudo['y_coord'],
                   gridsize=70, cmap=CMAP_THEME, mincnt=1, bins='log', edgecolors='none')
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Логарифм концентрации', color=TEXT_COLOR, rotation=270, labelpad=20)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cb.ax, 'yticklabels'), color=TEXT_COLOR)

    ax.set_title("Плотность распределения псевдопростых чисел", color=TEXT_COLOR, fontsize=16, pad=15)
    ax.set_xlabel("X координата")
    ax.set_ylabel("Y координата")
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color='#e0e0e0', linestyle='--', alpha=0.6)
    ax.axis('equal')

    plt.tight_layout()
    fig.savefig(PLOT_DENSITY_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Графики сохранены\n")


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"Ошибка: Файл {FILE_NAME} не найден.")
    else:
        dataframe = pd.read_csv(FILE_NAME)
        df_pseudo = dataframe[dataframe['is_pseudo'] == 1].copy()
        print(f"Псевдопростые числа: {len(df_pseudo)}\n")
        run_spatial_randomness_test(df_pseudo)
        run_maup_test(df_pseudo)
        generate_density_plot(df_pseudo)
