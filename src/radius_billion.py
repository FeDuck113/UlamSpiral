import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Ищем zip-архив нашего миллиарда и настраиваем масштаб

data_dir = 'data'
zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

zip_path = os.path.join(data_dir, zip_files[0])
print(f"📦 Работаем с архивом: {zip_path}")

# Разные радиусы берем
MAX_RADIUS = 15000.0
ring_radii = np.linspace(0, MAX_RADIUS, 6)[1:]

# Структуры для сбора статистики
ring_stats = {i: {'total': 0, 'pseudo': 0} for i in range(5)}

# Списки для сбора координат под график
bg_x, bg_y = [], []      # Для разреженного фона
pseudo_x, pseudo_y = [], [] # Для всех псевдопростых


# 2. Обработка потока: делим на чанки (группки) числа и постепенно добавляем на график

print(f"\n🚀 Анализ по радиусу {MAX_RADIUS}:")

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
        print(f"📈 Прогресс: обработано строк {total_rows_processed:,} (вне радиуса)")
        continue

    # Отбираем точки для графика:
    # Псевдопростые забираем ВСЕ
    pseudos = chunk_slice[chunk_slice['is_pseudo'] == 1]
    pseudo_x.extend(pseudos['x'].tolist())
    pseudo_y.extend(pseudos['y'].tolist())
    
    # Фон прореживаем: берем только 0.2% точек (каждую 500-ю), чтобы не взорвать Matplotlib
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
        
    print(f"📈 Прогресс: обработано строк {total_rows_processed:,} | Найдено аномалий: {len(pseudo_x)}")




# 3. Строим график

print("\n🎨 Рисуем масштабный график (это может занять около минуты)...")
plt.figure(figsize=(12, 12), facecolor='white')
ax = plt.subplot(111)

# Рисуем прореженный фон
ax.scatter(bg_x, bg_y, color='#e2e8f0', s=1, alpha=0.5, label='Фон (разреженный 0.2%)')

# Рисуем ВСЕ найденные на этом масштабе псевдопростые
ax.scatter(pseudo_x, pseudo_y, color='#1e40af', s=15, marker='o', 
           edgecolor='white', linewidth=0.3, zorder=5, label=f'Псевдопростые ({len(pseudo_x)} шт.)')

# Чертим границы колец
theta = np.linspace(0, 2*np.pi, 200)
for i, r in enumerate(ring_radii):
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='#94a3b8', linestyle='--', linewidth=0.8)
    ax.text(0, r - (MAX_RADIUS/15), f"Кольцо {i+1}", color='#64748b', fontsize=9, 
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#e2e8f0', alpha=0.8))

ax.set_aspect('equal')
ax.axis('off')
plt.title(f'Радиальное распределение аномалий на больших масштабах\nРадиус обзора: {MAX_RADIUS} | Всего строк в обработке: {total_rows_processed:,}', 
          fontsize=12, fontweight='bold', pad=20)
plt.legend(loc='upper right')
plt.tight_layout()

graph_name = f'billion_pseudoprimes_radial_R{int(MAX_RADIUS)}.png'
plt.savefig(graph_name, dpi=300)
print(f"✨ График успешно сохранен как '{graph_name}'")

# 4. Выводим статистику
print("\n" + "="*65)
print("🏆 ИТОГОВАЯ СТАТИСТИКА ПО МАСШТАБИРОВАННЫМ КОЛЬЦАМ")
print("="*65)

prev_r = 0
for ring_idx, r in enumerate(ring_radii):
    total = ring_stats[ring_idx]['total']
    pseudo = ring_stats[ring_idx]['pseudo']
    pct = (pseudo / total * 100) if total > 0 else 0
    
    print(f"КОЛЬЦО {ring_idx+1} (Радиус: {prev_r:.1f} -> {r:.1f})")
    print(f"  Всего чисел в кольце: {total:,} шт.")
    print(f"  └──🔹 ПСЕВДОПРОСТЫЕ:       {pseudo:,} шт.")
    print(f"      👉 Плотность аномалий: {pct:.6f}%")
    print("-" * 65)
    
    prev_r = r