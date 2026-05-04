import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "old/race_results_cleaned.csv"
OUT_DIR = "docs/images"

df = pd.read_csv(CSV_PATH)

sns.set_theme(style="whitegrid")

# --- Plot 1: Overview (3-panel) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax1, ax2, ax3 = axes

sns.histplot(data=df, x="age", bins=25, kde=True, color="skyblue", ax=ax1)
ax1.set_title("Distribution of Runner Ages", fontsize=14)
ax1.set_xlabel("Age")
ax1.set_ylabel("Count")

sns.histplot(data=df, x="zeit_min", bins=30, kde=True, color="lightgreen", ax=ax2)
ax2.set_title("Distribution of Finishing Times (Minutes)", fontsize=14)
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Count")

top_vereine = df["verein"].value_counts().nlargest(15)
sns.barplot(x=top_vereine.index, y=top_vereine.values, ax=ax3)
ax3.bar_label(ax3.containers[0], fontsize=8)
ax3.set_title("Top 15 Clubs by Number of Finishers")
ax3.set_xticks(range(len(top_vereine)))
ax3.set_xticklabels(top_vereine.index, rotation=60, ha="right", fontsize=7)
ax3.set_ylabel("")
ax3.set_xlabel("")

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_overview.png", bbox_inches="tight", dpi=180)
plt.close()
print("Saved plot_overview.png")

# --- Plot 2: Age vs. Pace ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=df, x="age", y="pace",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red", "linewidth": 2},
    ax=ax,
)
ax.set_title("Does Age Slow You Down? (Age vs. Pace)", fontsize=14)
ax.set_xlabel("Age")
ax.set_ylabel("Pace (min/km)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_age_vs_pace.png", bbox_inches="tight", dpi=180)
plt.close()
print("Saved plot_age_vs_pace.png")

# --- Plot 3: Time by Age Group & Gender ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=df, x="age_group", y="zeit_min", hue="gender",
    palette={"M": "#add8e6", "W": "#ffb6c1"}, ax=ax,
)
ax.set_title("Finishing Time by Age Group & Gender")
ax.set_xlabel("Age Group")
ax.set_ylabel("Time (min)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_time_by_agegroup.png", bbox_inches="tight", dpi=180)
plt.close()
print("Saved plot_time_by_agegroup.png")

# --- Plot 4: Fastest Clubs ---
fig, ax = plt.subplots(figsize=(10, 6))
club_stats = (
    df.groupby("verein")
    .agg(runner_count=("stn", "count"), avg_time=("zeit_min", "mean"))
    .reset_index()
)
valid_clubs = club_stats[club_stats["runner_count"] >= 5]
fastest_clubs = valid_clubs.nsmallest(15, "avg_time")
sns.barplot(data=fastest_clubs, y="avg_time", x="verein", ax=ax)
ax.set_title("Fastest Clubs (Min. 5 Runners)")
ax.set_ylabel("Average Time (min)")
ax.set_xlabel("")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_fastest_clubs.png", bbox_inches="tight", dpi=180)
plt.close()
print("Saved plot_fastest_clubs.png")

# --- Plot 5: ECDF by Gender ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.ecdfplot(
    data=df, x="zeit_min", hue="gender",
    palette={"M": "#1f77b4", "W": "#e377c2"},
    linewidth=2, ax=ax,
)
ax.set_title("Cumulative Finish Distribution (Percentiles)", fontsize=14)
ax.set_xlabel("Finishing Time (min)")
ax.set_ylabel("Proportion of Runners Finished")
ax.axvline(60, color="gray", linestyle="--", alpha=0.7)
ax.text(61, 0.5, "1-Hour Mark", color="gray", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_cumulative.png", bbox_inches="tight", dpi=180)
plt.close()
print("Saved plot_cumulative.png")

print("\nAll plots saved to", OUT_DIR)
