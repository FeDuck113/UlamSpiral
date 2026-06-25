import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import chi2_contingency

# Настраиваем zip-архив для миллиарда
data_dir = 'data'
zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

zip_path = os.path.join(data_dir, zip_files[0])

# Меняем радиусы
MAX_RADIUS = 60.0
ring_radii = np.linspace(0, MAX_RADIUS, 6)[1:]

# Для чанков (миллиард будем считать и тут временно хранить значения)
ring_stats = {i: {'total': 0, 'pseudo': 0} for i in range(5)}

# Списки для графиков
bg_x, bg_y = [], []
pseudo_x, pseudo_y = [], []


# Делим на чанки данные
print(f"\nАнализ по радиусу {MAX_RADIUS}:")

chunk_size = 10_000_000
chunk_iterator = pd.read_csv(zip_path, chunksize=chunk_size, compression='zip')

total_rows_processed = 0

for idx, chunk in enumerate(chunk_iterator):
    total_rows_processed += len(chunk)
    
    chunk.rename(columns={'x_coord': 'x', 'y_coord': 'y'}, inplace=True)
    
    # Фильтруем только то, что попало в наш новый большой круг
    chunk_slice = chunk[chunk['radius'] <= MAX_RADIUS]
    
    if chunk_slice.empty:
        print(f"Прогресс: обработано строк {total_rows_processed:,} (вне радиуса, не учитываем)")
        continue

    #Тут отбираем точки - псевдопростые забираем все
    pseudos = chunk_slice[chunk_slice['is_pseudo'] == 1]
    pseudo_x.extend(pseudos['x'].tolist())
    pseudo_y.extend(pseudos['y'].tolist())
    
    # Фон прореживаем: берем только 0.2% точек (каждую 500-ю)
    bg = chunk_slice[chunk_slice['is_pseudo'] == 0]
    if not bg.empty:
        bg_sampled = bg.sample(frac=0.002, random_state=42)
        bg_x.extend(bg_sampled['x'].tolist())
        bg_y.extend(bg_sampled['y'].tolist())

    # Собсвтенно, статистика
    prev_r = 0
    for ring_idx, r in enumerate(ring_radii):
        ring_data = chunk_slice[(chunk_slice['radius'] > prev_r) & (chunk_slice['radius'] <= r)]
        
        ring_stats[ring_idx]['total'] += len(ring_data)
        ring_stats[ring_idx]['pseudo'] += len(ring_data[ring_data['is_pseudo'] == 1])
        
        prev_r = r
        
    print(f"Прогресс: обработано строк {total_rows_processed:,} | Найдено аномалий: {len(pseudo_x)}")


# Подготовка плотностей для графика и легенды
densities = []
prev_r = 0
for ring_idx, r in enumerate(ring_radii):
    total = ring_stats[ring_idx]['total']
    pseudo = ring_stats[ring_idx]['pseudo']
    pct = (pseudo / total * 100) if total > 0 else 0
    densities.append(pct)


# Строим график
plt.figure(figsize=(13, 12), facecolor='white')
ax = plt.subplot(111)

cmap = cm.YlOrRd

min_d = min(densities) if min(densities) > 0 else 1e-6
max_d = max(densities) if max(densities) > min_d else min_d + 1e-5
norm = colors.Normalize(vmin=min_d, vmax=max_d)

# Сортируем кольца по убыванию радиуса, чтобы круги правильно накладывались
for i in range(len(ring_radii) - 1, -1, -1):
    r = ring_radii[i]
    ring_color = cmap(norm(densities[i]))
    
    circle_bg = plt.Circle((0, 0), r, color=ring_color, alpha=0.6, zorder=1)
    ax.add_patch(circle_bg)


theta = np.linspace(0, 2*np.pi, 200)
prev_r = 0
for i, r in enumerate(ring_radii):
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='#334155', linestyle='--', linewidth=1.0, zorder=2)
    
    ax.text(0, r - (MAX_RADIUS/15), f"Кольцо {i+1}", color='#0f172a', fontsize=10, fontweight='bold',
            ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#334155', linewidth=1.0, alpha=1.0))
    
    ax.plot([], [], color='none', label=f"Плотность Кольца {i+1}: {densities[i]:.6f}%")

# Рисуем прореженный фон
ax.scatter(bg_x, bg_y, color='#64748b', s=1, alpha=0.15, zorder=3)

# Рисуем все найденные псевдопростые точки
ax.scatter(pseudo_x, pseudo_y, color='#0284c7', s=12, marker='o', 
           edgecolor='white', linewidth=0.3, alpha=0.4, zorder=4, label='Псевдопростые')

ax.set_aspect('equal')
ax.axis('off')


if total_rows_processed >= 1_000_000_000:
    title_text = f'Радиальное распределение аномалий на больших масштабах\nРадиус обзора: {MAX_RADIUS}'
else:
    title_text = f'Радиальное распределение аномалий\nРадиус обзора: {MAX_RADIUS}'

plt.title(title_text, fontsize=12, fontweight='bold', pad=20)

# Добавляем боковую шкалу градиента (Colorbar) справа
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Концентрация псевдопростых чисел (плотность в %)', fontsize=10, labelpad=10)

# Сначала создаем легенду без zorder
leg = plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0')
# А теперь принудительно закидываем её на самый верхний слой
leg.set_zorder(7)
plt.tight_layout()

graph_name = f'billion_pseudoprimes_radial_R{int(MAX_RADIUS)}.png'
plt.savefig(graph_name, dpi=300)
print(f"График успешно сохранен как '{graph_name}'")

# Выводим статистику

print("\n" + "="*65)
print("ИТОГОВАЯ СТАТИСТИКА ПО МАСШТАБИРОВАННЫМ КОЛЬЦАМ")
print("="*65)

prev_r = 0
for ring_idx, r in enumerate(ring_radii):
    total = ring_stats[ring_idx]['total']
    pseudo = ring_stats[ring_idx]['pseudo']
    pct = densities[ring_idx]
    
    print(f"КОЛЬЦО {ring_idx+1} (Радиус: {prev_r:.1f} -> {r:.1f})")
    print(f"  Всего чисел в кольце: {total:,} шт.")
    print(f"  └── ПСЕВДОПРОСТЫЕ:       {pseudo:,} шт.")
    print(f"      Плотность аномалий: {pct:.6f}%")
    print("-" * 65)
    
    prev_r = r

# Хи-квадрат
print("\n" + "="*65)
print("МАТЕМАТИЧЕСКИЙ АНАЛИЗ ОДНОРОДНОСТИ РАСПРЕДЕЛЕНИЯ")
print("="*65)

pseudos_array = np.array([ring_stats[i]['pseudo'] for i in range(5)])
totals_array = np.array([ring_stats[i]['total'] for i in range(5)])
regulars_array = totals_array - pseudos_array

contingency_table = np.array([pseudos_array, regulars_array])

if np.sum(totals_array) > 0:
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Критерий однородности Хи-квадрат Пирсона:")
    print(f"  - Статистика критерия: {chi2_stat:.4f}")
    print(f"  - Число степеней свободы (dof): {dof}")
    print(f"  - Значение p-value: {p_value:.16e}")
    
    print("\nИнтерпретируем результаты:")
    if p_value < 0.05:
        print("  [ОТВЕРГАЕТСЯ H0] Изменения плотности аномалий СТАТИСТИЧЕСКИ ЗНАЧИМЫ.")
        print(f"  Вероятность случайной ошибки (p-value) ничтожно мала ({p_value:.4e}).")
        print("  Падение концентрации от центра к периферии не является случайным шумом,")
        print("  а отражает фундаментальную математическую закономерность затухания.")
    else:
        print("  [ПОДТВЕРЖДАЕТСЯ H0] Различия в плотности колец не выходят за рамки случайности.")
        print("  Распределение аномалий по кольцам можно считать относительно однородным.")
else:
    print("Ошибка: Нет данных для проведения статистического анализа.")
print("="*65)