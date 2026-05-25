# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def is_prime(n):
    n = int(n)
    if n < 2: return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

def is_fermat_pseudoprime(n, base=2):
    n = int(n)
    if n < 2 or is_prime(n): return False
    return pow(base, n - 1, n) == 1

N_max = 15000
numbers = np.arange(1, N_max + 1)

primes = [n for n in numbers if is_prime(n)]
pseudoprimes = [n for n in numbers if is_fermat_pseudoprime(n, 2)]
composites = [n for n in numbers if not is_prime(n) and not is_fermat_pseudoprime(n, 2)]

golden_angle = np.pi * (3 - np.sqrt(5))

def get_coords(arr):
    arr = np.array(arr)
    r = np.sqrt(arr)
    theta = arr * golden_angle
    return r * np.cos(theta), r * np.sin(theta)

x_p, y_p = get_coords(primes)
x_pp, y_pp = get_coords(pseudoprimes)
x_c, y_c = get_coords(composites)

plt.figure(figsize=(10, 10))
ax = plt.subplot(111)

ax.scatter(x_c, y_c, color='#f1f5f9', s=3, alpha=0.6, label='Составные числа (Фон)')
ax.scatter(x_p, y_p, color='#1f77b4', s=10, alpha=0.8, label='Простые числа')
ax.scatter(x_pp, y_pp, color='#d62728', s=120, marker='*', edgecolor='black', zorder=10, label='Псевдопростые Ферма (b=2)')

max_r = np.sqrt(N_max)
bin_edges = np.linspace(15, max_r, 6)
theta_circle = np.linspace(0, 2*np.pi, 200)

for i in range(len(bin_edges) - 1):
    r_start, r_end = bin_edges[i], bin_edges[i+1]
    ax.plot(r_start * np.cos(theta_circle), r_start * np.sin(theta_circle), color='#94a3b8', linestyle='--', linewidth=1.0)
    ax.text(0, (r_start + r_end)/2, f"Кольцо {i+1}", color='#1e293b', fontsize=10, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

ax.set_aspect('equal')
ax.axis('off')
plt.title('Разметка областей (колец) на спирали\nРаспределение простых и псевдопростых чисел', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('spiral_regions_plot.png', dpi=300)
print("Новый график и скрипт успешно созданы!")