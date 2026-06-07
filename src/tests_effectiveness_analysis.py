import pandas as pd
import numpy as np
import os
import itertools
import matplotlib
import ast

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'charts'))

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

FILE_NAME = os.path.join(DATA_DIR, 'billion-only-primes&pseudo.zip')
RANKING_PLOT_PATH = os.path.join(CHARTS_DIR, 'single_tests_ranking.png')
COMBOS_PLOT_PATH = os.path.join(CHARTS_DIR, 'best_combinations.png')
COST_PLOT_PATH = os.path.join(CHARTS_DIR, 'tests_cost.png')

BG_COLOR = 'white'
TEXT_COLOR = 'black'
STANDARD_TEST_COST = 10.0

TEST_COSTS = {
    "MILLER_RABIN": 1.2, "EULER_JACOBI": 1.5, "ABSOLUTE_EULER": 1.5,
    "EULER": 1.5, "BRUCKMAN_LUCAS": 3.0, "ODD_FIBONACCI": 3.0,
    "UNRESTRICTED_PERRIN": 3.0, "RESTRICTED_PERRIN": 3.0, "NSW": 3.0,
    "FROBENIUS": 3.0, "CATALAN": 3.0, "FERMAT": 1.0, "CARMICHAEL": 1.0
}


def get_cost(test_name):
    for key in TEST_COSTS.keys():
        if key in test_name:
            return TEST_COSTS[key]
    return 2.0


def run_efficiency_analysis(df_pseudo):
    print("Анализ эффективности комбинаций тестов\n")
    errors_map = {}

    for _, row in df_pseudo.iterrows():
        num = row['num']
        types_raw = str(row['types_str'])

        if types_raw == 'nan' or types_raw == '':
            continue

        types = types_raw.split('|')

        for t in types:
            if not t:
                continue
            if t not in errors_map:
                errors_map[t] = set()
            errors_map[t].add(num)

    test_names = list(errors_map.keys())
    best_combos = {}
    print("Поиск оптимальных комбинаций...")

    for k in range(1, 6):
        best_combo = None
        min_errors = float('inf')
        min_cost = float('inf')

        for ensemble in itertools.combinations(test_names, k):
            total_cost = sum(get_cost(t) for t in ensemble)
            common_errors = errors_map[ensemble[0]].copy()

            for t in ensemble[1:]:
                common_errors.intersection_update(errors_map[t])

            error_count = len(common_errors)

            if error_count < min_errors or (error_count == min_errors and total_cost < min_cost):
                min_errors = error_count
                min_cost = total_cost
                best_combo = ensemble

        best_combos[k] = {
            'combo': best_combo,
            'errors': min_errors,
            'cost': min_cost
        }

    print("\nОптимальные комбинации:")

    for k in range(1, 6):
        data = best_combos[k]
        combo_str = " + ".join(data['combo'])
        print(f"{k} тест(ов): {combo_str}")
        print(f"Ошибок: {data['errors']}, Стоимость: {data['cost']:.1f}\n")

    print("Построение графика одиночных тестов...")
    stats = [{'Test': t, 'Errors': len(e), 'Cost': get_cost(t)} for t, e in errors_map.items()]
    df_stats = pd.DataFrame(stats)
    df_stats['Score'] = (df_stats['Errors'] + 1) * df_stats['Cost']
    df_stats = df_stats.sort_values(by=['Errors', 'Cost'], ascending=[False, False])
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    fig1.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(BG_COLOR)
    bars = ax1.barh(df_stats['Test'], df_stats['Errors'], color='#4285F4', edgecolor='black', alpha=0.8)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + df_stats['Errors'].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"Ошибок: {int(width)}",
                 va='center', fontsize=10, color=TEXT_COLOR)

    ax1.set_title("Эффективность одиночных тестов", fontsize=16, pad=15)
    ax1.set_xlabel("Количество ошибок", fontsize=12)
    ax1.set_xlim(0, df_stats['Errors'].max() * 1.15)
    ax1.grid(True, axis='x', color='#e0e0e0', linestyle='--')
    plt.tight_layout()
    fig1.savefig(RANKING_PLOT_PATH, dpi=300)
    plt.close(fig1)

    print("Построение графика оптимальных комбинаций...")
    x_labels = [f"{k}" for k in range(1, 6)] + ["Стандарт"]
    costs = [best_combos[k]['cost'] for k in range(1, 6)] + [STANDARD_TEST_COST]
    errors = [best_combos[k]['errors'] for k in range(1, 6)] + [0]
    combos = ["\n+\n".join(best_combos[k]['combo']) for k in range(1, 6)] + ["Стандартный\nалгоритм"]
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    fig2.patch.set_facecolor(BG_COLOR)
    ax2.set_facecolor(BG_COLOR)
    bar_colors = ['#34A853' if e == 0 else '#EA4335' for e in errors[:-1]] + ['#4285F4']
    bars2 = ax2.bar(x_labels, costs, color=bar_colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Оптимальные комбинации тестов", fontsize=16, pad=20)
    ax2.set_xlabel("Количество тестов", fontsize=12)
    ax2.set_ylabel("Вычислительная стоимость", fontsize=12)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        err_text = f"Ошибок: {errors[i]}"

        ax2.text(bar.get_x() + bar.get_width() / 2, height / 2,
                 combos[i], ha='center', va='center', fontsize=9, color='white',
                 fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

        text_color = '#34A853' if errors[i] == 0 and i != 5 else '#EA4335'

        if i == 5:
            text_color = '#4285F4'

        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.15,
                 err_text, ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=text_color)

    ax2.set_ylim(0, max(costs) + 2.0)
    ax2.grid(True, axis='y', color='#e0e0e0', linestyle='--')
    plt.tight_layout()
    fig2.savefig(COMBOS_PLOT_PATH, dpi=300)
    plt.close(fig2)

    print("Построение графика стоимости тестов...")
    df_costs = df_stats[['Test', 'Cost']].copy()
    df_costs.loc[len(df_costs)] = ['STANDARD_ALGORITHM', STANDARD_TEST_COST]
    df_costs = df_costs.sort_values(by=['Cost', 'Test'], ascending=[True, False])
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    fig3.patch.set_facecolor(BG_COLOR)
    ax3.set_facecolor(BG_COLOR)

    def get_color_for_cost(c):
        if c <= 1.2: return '#34A853'
        if c <= 1.5: return '#FBBC04'
        if c < 10.0: return '#EA4335'
        return '#4285F4'

    bar_colors_cost = [get_color_for_cost(c) for c in df_costs['Cost']]
    bars3 = ax3.barh(df_costs['Test'], df_costs['Cost'], color=bar_colors_cost, edgecolor='black', alpha=0.85)

    for bar in bars3:
        width = bar.get_width()
        ax3.text(width + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{width:.1f}",
                 va='center', fontsize=11, color=TEXT_COLOR)

    ax3.set_title("Вычислительная стоимость тестов", fontsize=16, pad=15)
    ax3.set_xlabel("Стоимость", fontsize=12)
    ax3.set_xlim(0, STANDARD_TEST_COST + 1.5)
    ax3.grid(True, axis='x', color='#e0e0e0', linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#34A853', edgecolor='black', label='Низкая (≤1.2)'),
        Patch(facecolor='#FBBC04', edgecolor='black', label='Средняя (1.5)'),
        Patch(facecolor='#EA4335', edgecolor='black', label='Высокая (≥3.0)'),
        Patch(facecolor='#4285F4', edgecolor='black', label='Стандарт (10.0)')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', facecolor='white', edgecolor='#cccccc', fontsize=11)
    plt.tight_layout()
    fig3.savefig(COST_PLOT_PATH, dpi=300)
    plt.close(fig3)
    print("Графики сохранены\n")


def clean_and_parse_types(val):
    if pd.isna(val) or not isinstance(val, str):
        return ""
    val = val.strip()
    if val == "[]" or not val:
        return ""
    if val.startswith('[') and val.endswith(']'):
        val = val[1:-1]
    parts = [p.strip().strip("'\"") for p in val.split(',')]
    return '|'.join(p for p in parts if p)


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        print(f"Ошибка: Файл {FILE_NAME} не найден.")
    else:
        print(f"Чтение файла {FILE_NAME}...")
        chunk_list = []

        for chunk in pd.read_csv(FILE_NAME, chunksize=1000000):
            filtered = chunk[chunk['is_pseudo'] == 1]
            chunk_list.append(filtered)

        df_pseudo = pd.concat(chunk_list, ignore_index=True)
        print(f"Псевдопростые числа: {len(df_pseudo)}\n")
        print("Преобразование форматов колонок...")

        if 'types_str' not in df_pseudo.columns and 'pseudo_types' in df_pseudo.columns:
            df_pseudo['types_str'] = df_pseudo['pseudo_types'].apply(clean_and_parse_types)
        elif 'pseudoprime_types' in df_pseudo.columns:
            df_pseudo['types_str'] = df_pseudo['pseudoprime_types'].apply(clean_and_parse_types)
        elif 'types_str' in df_pseudo.columns:
            df_pseudo['types_str'] = df_pseudo['types_str'].apply(clean_and_parse_types)

        run_efficiency_analysis(df_pseudo)
