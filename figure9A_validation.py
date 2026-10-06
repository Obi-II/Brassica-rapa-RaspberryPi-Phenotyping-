
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
from adjustText import adjust_text

# ======================================================
# 1. FILE LOCATIONS
# ======================================================

BASE_FOLDER = Path(
    r"C:\Users\emman\Desktop\Plant_Growth_Analysis"
)

CSV_FILE = (
    BASE_FOLDER
    / "Final_Finetuned_Phenotyping_Outputs"
    / "final_finetuned_master_dataset.csv"
)

OUTPUT_FOLDER = (
    BASE_FOLDER / "AoB_PLANTS_Validation_Figures"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================
# 2. FIGURE SETTINGS
# ======================================================

SAVE_DPI = 600
DISPLAY_DPI = 100

WIDTH_CM = 17
HEIGHT_CM = 10

FIG_SIZE = (
    WIDTH_CM / 2.54,
    HEIGHT_CM / 2.54
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": DISPLAY_DPI,
    "savefig.dpi": SAVE_DPI,
})

# ======================================================
# 3. LOAD AND VALIDATE DATA
# ======================================================

df = pd.read_csv(CSV_FILE)

required = [
    "Plant_ID",
    "Raw_Final_Leaf_Area_cm2",
    "Fiji_Final_Leaf_Area_cm2",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df[required].copy()

for col in required[1:]:
    df[col] = pd.to_numeric(
        df[col],
        errors="raise"
    )

if (
    len(df) != 14
    or df[required].isna().any().any()
    or df["Plant_ID"].duplicated().any()
):
    raise ValueError(
        "Expected 14 complete, unique plant records."
    )

x = df[
    "Fiji_Final_Leaf_Area_cm2"
].to_numpy(dtype=float)

y = df[
    "Raw_Final_Leaf_Area_cm2"
].to_numpy(dtype=float)

if not (
    np.isfinite(x).all()
    and np.isfinite(y).all()
):
    raise ValueError(
        "Non-finite measurements detected."
    )

# ======================================================
# 4. STATISTICAL ANALYSIS
# ======================================================

reg = stats.linregress(x, y)

slope = reg.slope
intercept = reg.intercept

pearson_r = reg.rvalue
p_value = reg.pvalue
r_squared = pearson_r ** 2

# Direct measurement errors:
# HSV minus Fiji

difference = y - x

mae = np.mean(
    np.abs(difference)
)

rmse = np.sqrt(
    np.mean(difference ** 2)
)

print("\nHSV VERSUS FIJI AGREEMENT")
print("--------------------------------")
print(f"n = {len(df)}")
print(f"Pearson r = {pearson_r:.6f}")
print(f"p-value = {p_value:.6f}")
print(f"R² = {r_squared:.6f}")
print(f"MAE = {mae:.6f} cm²")
print(f"RMSE = {rmse:.6f} cm²")
print(f"Slope = {slope:.6f}")
print(f"Intercept = {intercept:.6f}")

# ======================================================
# 5. CREATE FIGURE
# ======================================================

fig = plt.figure(
    figsize=FIG_SIZE,
    dpi=DISPLAY_DPI
)

# Fixed axes position.
# The graph occupies most of the figure.

ax = fig.add_axes([
    0.12,
    0.16,
    0.83,
    0.79
])

axis_max = max(
    x.max(),
    y.max()
) * 1.10

line_x = np.linspace(
    0,
    axis_max,
    300
)

line_y = (
    intercept + slope * line_x
)

# ======================================================
# 6. PLOT MEASUREMENTS
# ======================================================

ax.scatter(
    x,
    y,
    s=32,
    color="#2563A6",
    label="Individual plants",
    zorder=3
)

# Regression line

ax.plot(
    line_x,
    line_y,
    color="#C23B3B",
    linewidth=1.6,
    label="Linear regression",
    zorder=2
)

# Perfect agreement line

ax.plot(
    line_x,
    line_x,
    linestyle="--",
    color="gray",
    linewidth=1.1,
    label="1:1 reference",
    zorder=1
)

# ======================================================
# 7. AXIS FORMATTING
# ======================================================

ax.set_xlim(
    0,
    axis_max
)

ax.set_ylim(
    0,
    axis_max
)

ax.set_xlabel(
    "Fiji reference leaf area (cm²)"
)

ax.set_ylabel(
    "Raw HSV-derived leaf area (cm²)"
)

# No title inside the graph.
# The title will be included in the caption.

ax.grid(
    alpha=0.18,
    linewidth=0.5
)

# ======================================================
# 8. PLANT LABELS
# ======================================================

# Specific label positions for closely
# grouped plants.
#
# These offsets affect labels only.
# The experimental measurements remain
# unchanged.

SPECIAL_OFFSETS = {
    "T4_P4": (-10, 12),
    "T1_P3": (10, 17),
    "T2_P3": (13, -15),
}

labels = []

for _, row in df.iterrows():

    plant_id = str(row["Plant_ID"])

    px = row[
        "Fiji_Final_Leaf_Area_cm2"
    ]

    py = row[
        "Raw_Final_Leaf_Area_cm2"
    ]

    dx, dy = SPECIAL_OFFSETS.get(
        plant_id,
        (5, 5)
    )

    # Convert offset in points into an
    # initial label position in data units.
    # This allows adjustText to refine it.

    display_point = ax.transData.transform(
        (px, py)
    )

    offset_pixels = np.array(
        [dx, dy]
    ) * fig.dpi / 72.0

    label_position = (
        ax.transData.inverted().transform(
            display_point + offset_pixels
        )
    )

    label = ax.text(
        label_position[0],
        label_position[1],
        plant_id,
        fontsize=6.8,
        ha="left",
        va="bottom",
        zorder=5
    )

    labels.append(label)

# ======================================================
# 9. AUTOMATIC LABEL ADJUSTMENT
# ======================================================

# Force a draw so Matplotlib calculates
# accurate text dimensions.

fig.canvas.draw()

adjust_text(
    labels,
    x=x,
    y=y,
    ax=ax,
    ensure_inside_axes=True,
    expand=(1.30, 1.50),
    force_text=(0.8, 1.0),
    force_static=(0.6, 0.8),
    force_pull=(0.005, 0.005),
    min_arrow_len=4,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.45
    )
)

# ======================================================
# 10. STATISTICAL INFORMATION BOX
# ======================================================

equation = (
    f"HSV = {intercept:.3f} "
    f"{slope:+.4f} × Fiji"
)

stats_text = (
    f"{equation}\n"
    f"r = {pearson_r:.3f}; "
    f"p = {p_value:.3f}\n"
    f"R² = {r_squared:.3f}; "
    f"n = {len(df)}\n"
    f"MAE = {mae:.2f} cm²\n"
    f"RMSE = {rmse:.2f} cm²"
)

ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.2,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="lightgray",
        alpha=1.0
    ),
    zorder=10
)

# ======================================================
# 11. LEGEND
# ======================================================

ax.legend(
    loc="upper right",
    fontsize=7,
    frameon=True,
    facecolor="white",
    edgecolor="lightgray"
)

# ======================================================
# 12. SAVE AT 600 DPI
# ======================================================

output_png = (
    OUTPUT_FOLDER
    / "Figure_9A_HSV_Fiji_Regression.png"
)

output_jpg = (
    OUTPUT_FOLDER
    / "Figure_9A_HSV_Fiji_Regression.jpg"
)

fig.savefig(
    output_png,
    dpi=SAVE_DPI,
    facecolor="white"
)

fig.savefig(
    output_jpg,
    dpi=SAVE_DPI,
    facecolor="white",
    pil_kwargs={"quality": 95}
)

# ======================================================
# 13. VERIFY SAVED IMAGES
# ======================================================

print("\nSAVED FIGURE VERIFICATION")
print("--------------------------------")

for output_file in [
    output_png,
    output_jpg
]:

    with Image.open(output_file) as img:

        print(
            f"\nFile: {output_file.name}"
        )

        print(
            f"Dimensions: {img.size}"
        )

        print(
            f"DPI: {img.info.get('dpi')}"
        )

        print(
            f"Colour mode: {img.mode}"
        )

print("\nOutput folder:")
print(OUTPUT_FOLDER)

# ======================================================
# 14. DISPLAY SMALL PREVIEW
# ======================================================

plt.show()
