# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# НАСТРОЙКИ ФИЛЬТРАЦИИ И ФАЙЛА
# ==========================================
FILE_NAME = 'data/1000000-spiral.csv' 

# Задаем максимальный радиус. Все точки дальше этого значения станут невидимыми (отсекутся).
# Для файла на 10к чисел радиус — около 60.
# Для файла на миллион берем 500. Для 100к возьмем 180.
MAX_RADIUS = 500.0  

print(f"Читаю датасет: {FILE_NAME}")
df = pd.read_csv(FILE_NAME)

# Фильтрация: оставляем ТОЛЬКО те числа, которые лежат внутри заданного радиуса (никаких углов)
df_slice = df[df['radius'] <= MAX_RADIUS].copy()

# ==========================================
# Разбиваем точки на категории
# ==========================================
df_comp = df_slice[(df_slice['label'] == 'comp') & (df_slice['is_pseudo'] == 0)]
df_prime = df_slice[df_slice['label'] == 'prime']
df_pseudo = df_slice[df_slice['is_pseudo'] == 1]

# Группируем псевдопростые
df_f3 = df_pseudo[df_pseudo['pseudo_types'].str.contains('FERMAT_3', na=False)]
df_f5 = df_pseudo[df_pseudo['pseudo_types'].str.contains('FERMAT_5', na=False)]
df_nsw = df_pseudo[df_pseudo['pseudo_types'].str.contains('NSW', na=False)]
df_other_pseudo = df_pseudo[~df_pseudo['pseudo_types'].str.contains('FERMAT_3|FERMAT_5|NSW', na=False)]

# ==========================================
# ПОСТРОЕНИЕ ГРАФИКА
# ==========================================
plt.figure(figsize=(12, 12), facecolor='white')
ax = plt.subplot(111)

# Фон и простые числа — обычные аккуратные точки
ax.scatter(df_comp['x_coord'], df_comp['y_coord'], 
           color='#e2e8f0', s=2, marker='.', alpha=0.6, zorder=1, label='Составные числа (Фон)')

ax.scatter(df_prime['x_coord'], df_prime['y_coord'], 
           color='#1f77b4', s=8, marker='.', alpha=0.8, zorder=2, label='Простые числа')

# Псевдопростые — точки покрупнее, чтобы выделялись
ax.scatter(df_f3['x_coord'], df_f3['y_coord'], 
           color='#2ca02c', s=35, marker='o', edgecolor='black', linewidth=0.5, zorder=10, label='Псевдопростые Ферма (осн. 3)')

ax.scatter(df_f5['x_coord'], df_f5['y_coord'], 
           color='#ff7f0e', s=35, marker='o', edgecolor='black', linewidth=0.5, zorder=10, label='Псевдопростые Ферма (осн. 5)')

ax.scatter(df_nsw['x_coord'], df_nsw['y_coord'], 
           color='#d62728', s=40, marker='o', edgecolor='black', linewidth=0.5, zorder=10, label='Псевдопростые NSW')

if not df_other_pseudo.empty:
    ax.scatter(df_other_pseudo['x_coord'], df_other_pseudo['y_coord'], 
               color='#9467bd', s=35, marker='o', edgecolor='black', linewidth=0.5, zorder=10, label='Другие псевдопростые')

# ==========================================
# РАЗМЕТКА ГРАНИЦ И ВЫВОД ЧИСЛЕННЫХ РАДИУСОВ
# ==========================================
num_rings = 5
# Считаем шаги радиусов строго до нашего MAX_RADIUS
ring_radii = np.linspace(MAX_RADIUS / num_rings, MAX_RADIUS, num_rings)

t = np.linspace(0, 2 * np.pi, 400)
prev_r = 0

for i, r in enumerate(ring_radii):
    # Рисуем ЧЁТКУЮ темно-серую пунктирную окружность-границу
    ax.plot(r * np.cos(t), r * np.sin(t), color='#334155', linestyle='--', linewidth=1.8, alpha=0.8, zorder=15)
    
    # Считаем место для плашки (посередине текущего кольца)
    label_r = prev_r + ((r - prev_r) / 2)
    
    # Координаты для текста по диагонали (под углом 45 градусов)
    text_x = label_r * np.cos(np.pi / 4)
    text_y = label_r * np.sin(np.pi / 4)
    
    # Текст теперь содержит и номер кольца, и диапазон его радиусов из CSV!
    ring_text = f"Кольцо {i+1}\nR: {prev_r:.1f} – {r:.1f}"
    
    ax.text(text_x, text_y, ring_text, 
            color='#0f172a', fontsize=9, fontweight='bold', ha='center', va='center', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#94a3b8', alpha=0.95))
    prev_r = r

# ==========================================
# ОФОРМЛЕНИЕ И КРОП ПО КРУГУ
# ==========================================
# Обрезаем оси холста ровно по размеру максимального радиуса (+ небольшой запас)
ax.set_xlim(-MAX_RADIUS - 2, MAX_RADIUS + 2)
ax.set_ylim(-MAX_RADIUS - 2, MAX_RADIUS + 2)

ax.set_aspect('equal')
ax.axis('off')

plt.title(f'Радиальный анализ спирали Улама (Обрезка по радиусу R ≤ {MAX_RADIUS})\nРаспределение чисел по диапазонам расстояний', 
          fontsize=13, fontweight='bold', pad=30)

plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, 
           fontsize=10, title="Классификация", title_fontsize=11)

plt.tight_layout()

# Сохраняем
output_name = f'clipped_circular_ulam_{int(MAX_RADIUS)}.png'
plt.savefig(output_name, dpi=300)
plt.close()
# ==========================================
# ВЫВОД СТАТИСТИКИ (ПРОЦЕНТНАЯ ПЛОТНОСТЬ) В КОНСОЛЬ
# ==========================================
print("\n" + "="*65)
print("АНАЛИЗ ОТНОСИТЕЛЬНОЙ ПЛОТНОСТИ ПО КОЛЬЦАМ (%)")
print("="*65)

prev_r = 0
for i, r in enumerate(ring_radii):
    # Фильтруем строки датасета, которые попали строго в текущее кольцо
    ring_data = df_slice[(df_slice['radius'] > prev_r) & (df_slice['radius'] <= r)]
    total_in_ring = len(ring_data)
    
    if total_in_ring == 0:
        print(f"КОЛЬЦО {i+1} (Радиус: {prev_r:.1f} -> {r:.1f}) — Данные отсутствуют\n" + "-"*65)
        prev_r = r
        continue
        
    # Считаем абсолютное количество для контроля
    comps = len(ring_data[(ring_data['label'] == 'comp') & (ring_data['is_pseudo'] == 0)])
    primes = len(ring_data[ring_data['label'] == 'prime'])
    
    f3 = len(ring_data[ring_data['pseudo_types'].str.contains('FERMAT_3', na=False)])
    f5 = len(ring_data[ring_data['pseudo_types'].str.contains('FERMAT_5', na=False)])
    nsw = len(ring_data[ring_data['pseudo_types'].str.contains('NSW', na=False)])
    
    # Считаем 'others' через логическое отрицание, чтобы избежать багов с наложением f3/f5
    others = len(ring_data[(ring_data['is_pseudo'] == 1) & 
                          ~ring_data['pseudo_types'].str.contains('FERMAT_3|FERMAT_5|NSW', na=False)])
    
    # Переводим в относительную плотность (%) внутри ТЕКУЩЕГО кольца
    comps_pct = (comps / total_in_ring) * 100
    primes_pct = (primes / total_in_ring) * 100
    f3_pct = (f3 / total_in_ring) * 100
    f5_pct = (f5 / total_in_ring) * 100
    nsw_pct = (nsw / total_in_ring) * 100
    others_pct = (others / total_in_ring) * 100
    
    print(f"КОЛЬЦО {i+1} (Радиус: {prev_r:.1f} -> {r:.1f})")
    print(f"  Всего точек в кольце: {total_in_ring}")
    print(f"  📊 Доля составных чисел (фон): {comps_pct:.2f}%  ({comps} шт.)")
    print(f"  🔹 Доля простых чисел:         {primes_pct:.2f}%  ({primes} шт.)")
    print(f"  🔸 Псевдопростые Ферма (осн.3): {f3_pct:.4f}%  ({f3} шт.)")
    print(f"  🔸 Псевдопростые Ферма (осн.5): {f5_pct:.4f}%  ({f5} шт.)")
    print(f"  🔺 Псевдопростые NSW:           {nsw_pct:.4f}%  ({nsw} шт.)")
    if others > 0:
        print(f"  🤷 Другие псевдопростые:        {others_pct:.4f}%  ({others} шт.)")
    print("-" * 65)
    
    prev_r = r


print(f"Готово! График Улама сохранен как: {output_name}")