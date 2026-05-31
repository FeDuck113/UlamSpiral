import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FILE_NAME = os.path.join(DATA_DIR, '1000000-spiral.csv')
PLOT_2_PATH = os.path.join(DATA_DIR, 'plot_2_hexbin_density.png')


BG_COLOR = 'white'
TEXT_COLOR = 'black'
CMAP_THEME = 'YlOrRd'


def generate_density_plot(pseudos_dataframe):
    print("Анализ геометрической концентрации")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    hb = ax.hexbin(pseudos_dataframe['x_coord'], pseudos_dataframe['y_coord'],
                   gridsize=40, cmap=CMAP_THEME, mincnt=1, bins='log', edgecolors='none')
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
    fig.savefig(PLOT_2_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"График успешно сохранен: '{PLOT_2_PATH}'!")


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"Ошибка: Файл {FILE_NAME} не найден.")
    else:
        dataframe = pd.read_csv(FILE_NAME)
        pseudos = dataframe[dataframe['is_pseudo'] == 1].copy()
        print(f"Запуск построения карты плотности для {len(pseudos)} чисел...\n")
        generate_density_plot(pseudos)
