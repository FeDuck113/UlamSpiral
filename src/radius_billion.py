import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# 1. Ищем zip-архив нашего миллиарда и настраиваем масштаб

data_dir = 'data'
zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

zip_path = os.path.join(data_dir, zip_files[0])
print(f"Работаем с архивом: {zip_path}")

# Разные радиусы берем
MAX_RADIUS = 10000.0
ring_radii = np.linspace(0, MAX_RADIUS, 6)[1:]

# Структуры для сбора статистики
ring_stats = {i: {'total': 0, 'pseudo': 0} for i in range(5)}

# Списки для сбора координат под график
bg_x, bg_y = [], []      # Для разреженного фона
pseudo_x, pseudo_y = [], [] # Для всех псевдопростых


# 2. Обработка потока: делим на чанки (группки) числа и постепенно добавляем на график

print(f"\nАнализ по радиусу {MAX_RADIUS}:")

chunk_size = 10_000_000
chunk_iterator = pd.read_csv(zip_path, chunksize=chunk_size, compression='zip')

total_rows_processed = 0

for idx, chunk in enumerate(chunk_iterator):
    total_rows_processed += len(chunk)
    
    # Исправляем колонки
    chunk.rename(columns={'x_coord': 'x', 'y_coord': 'y'}, inplace=True)
    
    # Фильтруем только то, что попало в наш новый большой круг
    chunk_slice = chunk[chunk['radius'] <= MAX_RADIUS]
    
    if chunk_slice.empty:
        print(f"Прогресс: обработано строк {total_rows_processed:,} (вне радиуса)")
        continue

    # Отбираем точки для графика:
    # Псевдопростые забираем ВСЕ
    pseudos = chunk_slice[chunk_slice['is_pseudo'] == 1]
    pseudo_x.extend(pseudos['x'].tolist())
    pseudo_y.extend(pseudos['y'].tolist())
    
    # Фон прореживаем: берем только 0.2% точек (каждую 500-ю)
    bg = chunk_slice[chunk_slice['is_pseudo'] == 0]
    if not bg.empty:
        bg_sampled = bg.sample(frac=0.002, random_state=42)
        bg_x.extend(bg_sampled['x'].tolist())
        bg_y.extend(bg_sampled['y'].tolist())

    # Считаем статистику по кольцам
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

# Настройка градиента цветов для колец (бубликов)
# Чем выше плотность, тем ярче/насыщеннее цвет кольца (от бледно-голубого к глубокому синему)
cmap = cm.Blues
norm = colors.Normalize(vmin=min(densities) if min(densities) > 0 else 0.0001, vmax=max(densities))

# 3. Строим график

print("\nРисуем масштабный график (это может занять около минуты)...")
plt.figure(figsize=(13, 12), facecolor='white')
ax = plt.subplot(111)

# Делаем цвета жёстче: берём контрастную палитру YlOrRd
cmap = cm.YlOrRd

# Задаём жёсткие границы строго от минимума до максимума текущих плотностей
min_d = min(densities) if min(densities) > 0 else 1e-6
max_d = max(densities) if max(densities) > min_d else min_d + 1e-5
norm = colors.Normalize(vmin=min_d, vmax=max_d)

# Сортируем кольца по убыванию радиуса, чтобы круги правильно накладывались
for i in range(len(ring_radii) - 1, -1, -1):
    r = ring_radii[i]
    ring_color = cmap(norm(densities[i]))
    
    # Делаем цвет жёстче и сочнее: alpha=0.6
    circle_bg = plt.Circle((0, 0), r, color=ring_color, alpha=0.6, zorder=1)
    ax.add_patch(circle_bg)

# Отрисовка элементов поверх цветных бубликов
theta = np.linspace(0, 2*np.pi, 200)
prev_r = 0
for i, r in enumerate(ring_radii):
    # Пунктирные границы колец (тёмные для видимости на жёлтом/красном фоне)
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='#334155', linestyle='--', linewidth=1.0, zorder=2)
    
    # Текстовые метки "Кольцо X" — ТЕПЕРЬ СТОПРОЦЕНТНО ВИДИМЫЕ
    # alpha=1.0 делает плашку полностью непрозрачной белой, перекрывая синие точки под ней
    ax.text(0, r - (MAX_RADIUS/15), f"Кольцо {i+1}", color='#0f172a', fontsize=10, fontweight='bold',
            ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#334155', linewidth=1.0, alpha=1.0))
    
    # Фиктивная линия для вывода плотности в легенду
    ax.plot([], [], color='none', label=f"Плотность Кольца {i+1}: {densities[i]:.6f}%")

# Рисуем прореженный фон из обычных чисел (zorder=3)
ax.scatter(bg_x, bg_y, color='#64748b', s=1, alpha=0.15, zorder=3)

# Рисуем ВСЕ найденные псевдопростые точки (zorder=4)
# Добавили alpha=0.4, чтобы они просвечивали и не забивали собой красный цвет центра
ax.scatter(pseudo_x, pseudo_y, color='#0284c7', s=12, marker='o', 
           edgecolor='white', linewidth=0.3, alpha=0.4, zorder=4, label='Псевдопростые')

ax.set_aspect('equal')
ax.axis('off')

# Управляем заголовком
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
# 4. Выводим статистику

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