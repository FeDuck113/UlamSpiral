import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== ЗАГРУЗКА ==========
df_small = pd.read_csv("10000-spiral.csv")
df_large = pd.read_csv("1000000-spiral.csv")

# ========== ФИЛЬТРЫ ==========
# Малый датасет
primes_s = df_small[df_small["label"] == "prime"]
pseudo_s = df_small[df_small["label"] == "pseudoprime"]

# Большой датасет
primes_l = df_large[df_large["label"] == "prime"]
pseudo_l = df_large[df_large["label"] == "pseudoprime"]

# ========== СРАВНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# --- Верхний ряд: малый датасет (10k) ---
axes[0, 0].scatter(primes_s["x_coord"], primes_s["y_coord"], c="black", s=3)
axes[0, 0].set_title(f"10k: Простые числа (n={len(primes_s)})")
axes[0, 0].set_aspect("equal")

axes[0, 1].scatter(primes_s["x_coord"], primes_s["y_coord"], c="lightgray", s=1, alpha=0.5)
axes[0, 1].scatter(pseudo_s["x_coord"], pseudo_s["y_coord"], c="red", s=15, edgecolors="black", linewidth=0.5)
axes[0, 1].set_title(f"10k: Псевдопростые (n={len(pseudo_s)})")
axes[0, 1].set_aspect("equal")

# --- Нижний ряд: большой датасет (1M) ---
axes[1, 0].scatter(primes_l["x_coord"], primes_l["y_coord"], c="black", s=0.5)
axes[1, 0].set_title(f"1M: Простые числа (n={len(primes_l)})")
axes[1, 0].set_aspect("equal")

axes[1, 1].scatter(primes_l["x_coord"], primes_l["y_coord"], c="lightgray", s=0.1, alpha=0.3)
axes[1, 1].scatter(pseudo_l["x_coord"], pseudo_l["y_coord"], c="red", s=2)
axes[1, 1].set_title(f"1M: Псевдопростые (n={len(pseudo_l)})")
axes[1, 1].set_aspect("equal")

plt.tight_layout()
plt.savefig("spiral_comparison.png", dpi=150)
plt.show()

# ========== БЫСТРАЯ СТАТИСТИКА ==========
print("=" * 50)
print("СТАТИСТИКА ПО ДАТАСЕТАМ")
print("=" * 50)
print(f"10k:   простых = {len(primes_s)}, псевдопростых = {len(pseudo_s)}")
print(f"1M:    простых = {len(primes_l)}, псевдопростых = {len(pseudo_l)}")
print("=" * 50)
print("Типы псевдопростых (1M):")
for t in pseudo_l["pseudoprime_types"].value_counts().head(10).index:
    print(f"  {t}: {pseudo_l['pseudoprime_types'].value_counts()[t]}")