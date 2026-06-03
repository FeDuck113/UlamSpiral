import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def run_sector_significance_tests(
    df: pd.DataFrame,
    sector_stats: pd.DataFrame,
    num_sectors: int,
    num_simulations: int = 5000,
    random_seed: int = 42,
):
    contingency_table = np.vstack(
        [
            sector_stats["pseudo_count"].to_numpy(dtype=int),
            sector_stats["prime_count"].to_numpy(dtype=int),
        ]
    )

    chi2_statistic, chi2_p_value, _, expected = chi2_contingency(contingency_table)

    print("\n\n")
    print("Chi square: Доля псевдопростых относительно простых во всех секторах одинакова")
    print(f"p_value = {chi2_p_value:.6g}")
    print(f"min_expected_count = {expected.min():.3f}")
    if chi2_p_value < 0.05:
        print("Итог: p_value < 0.05, различия между секторами стат значимы\n")
    else:
        print("Итог: p_value >= 0.05, стат значимых различий между секторами не найдено\n")

    relevant_df = df[df["label"].isin(["prime", "pseudo"])].copy()
    sector_indices = relevant_df["sector"].astype(int).to_numpy()
    pseudo_flags = (relevant_df["label"] == "pseudo").to_numpy(dtype=int)
    observed_densities = sector_stats["pseudo_density"].to_numpy(dtype=float)
    observed_best_density = observed_densities.max()
    observed_best_sector = int(sector_stats.loc[sector_stats["pseudo_density"].idxmax(), "sector"])

    rng = np.random.default_rng(random_seed)
    simulated_chi2_statistics = np.empty(num_simulations, dtype=float)
    simulated_max_densities = np.empty(num_simulations, dtype=float)

    for simulation_idx in range(num_simulations):
        shuffled_flags = rng.permutation(pseudo_flags)
        simulated_pseudo_counts = np.bincount(
            sector_indices,
            weights=shuffled_flags,
            minlength=num_sectors,
        ).astype(int)
        simulated_prime_counts = np.bincount(
            sector_indices,
            weights=1 - shuffled_flags,
            minlength=num_sectors,
        ).astype(int)

        simulated_table = np.vstack([simulated_pseudo_counts, simulated_prime_counts])
        simulated_chi2_statistics[simulation_idx] = chi2_contingency(simulated_table)[0]
        simulated_densities = simulated_pseudo_counts / np.maximum(simulated_prime_counts, 1)
        simulated_max_densities[simulation_idx] = simulated_densities.max()

    monte_carlo_chi2_p_value = (
        np.count_nonzero(simulated_chi2_statistics >= chi2_statistic) + 1
    ) / (num_simulations + 1)

    print(f"monte_carlo_p_value = {monte_carlo_chi2_p_value:.6g}")
    if monte_carlo_chi2_p_value < 0.05:
        print("Итог: Monte Carlo подтверждает значимые различия между секторами")
    else:
        print("Итог: Monte Carlo не подтверждает значимые различия между секторами")

    best_sector_p_value = (
        np.count_nonzero(simulated_max_densities >= observed_best_density) + 1
    ) / (num_simulations + 1)

    print("\nMonte Carlo для лучшего сектора: необычна ли максимальная плотность после выбора лучшего сектора по данным?")
    print(f"best_sector = {observed_best_sector}")
    print(f"observed_best_density = {observed_best_density:.6f}")
    print(f"best_sector_monte_carlo_p_value = {best_sector_p_value:.6g}")
    if best_sector_p_value < 0.05:
        print("Итог: лучший сектор выделяется сильнее, чем ожидалось бы случайно")
    else:
        print("Итог: выделение лучшего сектора можно объяснить случайностью")

    return {
        "chi2_statistic": chi2_statistic,
        "chi2_p_value": chi2_p_value,
        "chi2_monte_carlo_p_value": monte_carlo_chi2_p_value,
        "min_expected_count": float(expected.min()),
        "best_sector_density": float(observed_best_density),
        "best_sector_p_value": best_sector_p_value,
        "num_simulations": num_simulations,
    }


def analyze_and_plot_spiral(
    df_or_path: str,
    num_sectors: int = 10,
    save_path: str = "charts/spiral_sectors_plot.png",
):
    df = (
        pd.read_csv(df_or_path)
    )

    df['angle_deg'] = np.degrees(df['angle_rad'])

    # Sectoring
    sector_edges = np.linspace(0, 360, num_sectors + 1)
    df["sector"] = pd.cut(
        df["angle_deg"],
        bins=sector_edges,
        labels=range(num_sectors),
        include_lowest=True,
    )
    sector_stats = (
        df.groupby("sector", observed=False)
        .agg(
            total_count=("num", "count"),
            prime_count=("label", lambda x: (x == "prime").sum()),
            pseudo_count=("label", lambda x: (x == "pseudo").sum()),
        )
        .reset_index()
    )

    # calc density
    sector_stats["pseudo_density"] = sector_stats["pseudo_count"] / (
        sector_stats["prime_count"] + 1e-9
    )

    print("\n=== СТАТИСТИКА ПО СЕКТОРAM ===")
    print(sector_stats.to_string(index=False))

    # Get sector with max density
    best_sector_idx = sector_stats["pseudo_density"].idxmax()
    best_sector = int(sector_stats.loc[best_sector_idx, "sector"])
    best_sector_start = sector_edges[best_sector]
    best_sector_end = sector_edges[best_sector + 1]
    print(
        f"\nСектор с максимальной плотностью псевдопростых: "
        f"№{best_sector} ({best_sector_start:.1f}°-{best_sector_end:.1f}°)"
    )

    run_sector_significance_tests(df, sector_stats, num_sectors)

    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")

    # Sectors
    theta_edges = np.linspace(0, 2 * np.pi, num_sectors + 1)
    sector_width = 2 * np.pi / num_sectors
    max_radius = df["radius"].max() if df["radius"].max() > 0 else 1.0

    densities = sector_stats["pseudo_density"].values
    v_min, v_max = (
        (densities.min(), densities.max())
        if densities.max() > 0
        else (0.0, 1.0)
    )
    norm = plt.Normalize(v_min, v_max)
    cmap = sns.color_palette("Blues", as_cmap=True)

    for i in range(num_sectors):
        sector_color = cmap(norm(densities[i]))
        ax.bar(
            x=theta_edges[i] + sector_width / 2,  # Центр луча
            height=max_radius,
            width=sector_width,
            bottom=0.0,
            color=sector_color,
            alpha=0.6,
            edgecolor="lightgrey",
            linewidth=0.5,
            zorder=1,
        )

    # Draw points
    pseudo_df = df[df["label"] == "pseudo"].copy()

    is_in_best_sector = pseudo_df["sector"] == best_sector
    best_pseudo = pseudo_df[is_in_best_sector]
    other_pseudo = pseudo_df[~is_in_best_sector]

    
    ax.scatter(
        best_pseudo["angle_rad"],
        best_pseudo["radius"],
        color="red",
        s=55,
        edgecolors="black",
        linewidths=0.8,
        label=(
            "Псевдопростые (Сектор с наибольшой плотностью "
            f"{best_sector_start:.1f}°-{best_sector_end:.1f}°)"
        ),
        zorder=4,
    )
    ax.scatter(
        other_pseudo["angle_rad"],
        other_pseudo["radius"],
        color="hotpink",
        s=25,
        alpha=0.7,
        label="Псевдопростые (Другие сектора)",
        zorder=3,
    )


    # Polar grid
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=2)

    # Colorbar
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable, 
        ax=ax, 
        orientation="horizontal", 
        pad=0.08,
        shrink=0.6
    )
    colorbar.set_label("Плотность распределения псевдопростых чисел относительно простых", fontsize=11)

    # Title
    plt.title(
        f"Анализ спирали по секторам угла\n(Кол-во секторов: {num_sectors})",
        fontsize=14,
        fontweight="bold",
        pad=45,
    )

    # Legend
    plt.legend(
        loc="upper left", 
        bbox_to_anchor=(1.0, 1.0),
        fontsize=10,
        frameon=True,
        facecolor="white", 
        edgecolor="none"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot_spiral("data/1000000-spiral.csv", num_sectors=10)
    # analyze_and_plot_spiral("data/billion-only-primes&pseudo.csv", num_sectors=10)