import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_and_plot_spiral(
    df_or_path: str,
    num_sectors: int = 10,
    save_path: str = "charts/spiral_sectors_plot.png",
):
    df = (
        pd.read_csv(df_or_path)
    )

    # Convert radians (0 -> 2*pi) to degrees (0 -> 360)
    df["angle_deg"] = np.degrees(df["angle"]) % 360

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
            pseudo_count=("label", lambda x: (x == "pseudoprime").sum()),
        )
        .reset_index()
    )

    # calc density
    sector_stats["pseudo_density"] = sector_stats["pseudo_count"] / (
        sector_stats["total_count"] + 1e-9
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
    pseudo_df = df[df["label"] == "pseudoprime"].copy()

    is_in_best_sector = pseudo_df["sector"] == best_sector
    best_pseudo = pseudo_df[is_in_best_sector]
    other_pseudo = pseudo_df[~is_in_best_sector]

    if not other_pseudo.empty:
        ax.scatter(
            other_pseudo["angle"],
            other_pseudo["radius"],
            color="hotpink",
            s=25,
            alpha=0.7,
            label="Псевдопростые (Другие сектора)",
            zorder=3,
        )

    if not best_pseudo.empty:
        ax.scatter(
            best_pseudo["angle"],
            best_pseudo["radius"],
            color="red",
            s=55,
            edgecolors="black",
            linewidths=0.8,
            label=(
                "Псевдопростые (Лучший сектор "
                f"{best_sector_start:.1f}°-{best_sector_end:.1f}°)"
            ),
            zorder=4,
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