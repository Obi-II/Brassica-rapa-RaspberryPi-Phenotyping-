from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

BASE_FOLDER = Path(
    r"C:\Users\emman\Desktop\Plant_Growth_Analysis"
)

# This dataset is required for the Bland–Altman analysis
MASTER_CSV = (
    BASE_FOLDER
    / "Final_Finetuned_Phenotyping_Outputs"
    / "final_finetuned_master_dataset.csv"
)

OUTPUT_FOLDER = (
    BASE_FOLDER
    / "AoB_PLANTS_Validation_Figures"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. AoB PLANTS FIGURE SETTINGS
# ============================================================

DPI = 600

WIDTH_CM = 17
HEIGHT_CM = 10

WIDTH_IN = WIDTH_CM / 2.54
HEIGHT_IN = HEIGHT_CM / 2.54


# ============================================================
# 3. PLANT ORDER
# ============================================================

PLANT_ORDER = [
    "T1_P1", "T1_P2", "T1_P3", "T1_P4",
    "T2_P1", "T2_P3", "T2_P4",
    "T3_P1", "T3_P2", "T3_P4",
    "T4_P1", "T4_P2", "T4_P3", "T4_P4",
]


# ============================================================
# 4. DESTRUCTIVE VALIDATION DATA
#    Fiji leaf area (cm²), measured dry biomass (g)
# ============================================================

VALIDATION_DATA = [
    ("T1_P1", 146.903, 1.085),
    ("T1_P2",  35.763, 0.089),
    ("T1_P3", 147.112, 0.599),
    ("T1_P4", 140.263, 0.768),

    ("T2_P1",  60.042, 0.138),
    ("T2_P3", 154.449, 0.755),
    ("T2_P4", 149.849, 0.916),

    ("T3_P1", 144.211, 0.606),
    ("T3_P2", 175.174, 0.930),
    ("T3_P4", 163.691, 0.928),

    ("T4_P1", 109.144, 0.502),
    ("T4_P2", 158.602, 0.835),
    ("T4_P3", 135.900, 0.820),
    ("T4_P4", 150.568, 1.333),
]


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def r_squared(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_residual = np.sum(
        (observed - predicted) ** 2
    )

    ss_total = np.sum(
        (observed - observed.mean()) ** 2
    )

    return 1.0 - (
        ss_residual / ss_total
    )


def rmse(observed, predicted):
    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return math.sqrt(
        np.mean(
            (observed - predicted) ** 2
        )
    )


def save_aob_figure(fig, filename):
    """
    Save an AoB PLANTS-ready figure:
    JPG, RGB, 600 dpi, 17 × 10 cm.
    """

    output_path = (
        OUTPUT_FOLDER / filename
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        format="jpg",
        facecolor="white",
        edgecolor="white"
    )

    plt.close(fig)

    # Force RGB and store 600 dpi metadata
    image = Image.open(
        output_path
    ).convert("RGB")

    image.save(
        output_path,
        "JPEG",
        quality=100,
        dpi=(DPI, DPI)
    )

    # Verify
    check = Image.open(
        output_path
    )

    width_px, height_px = check.size

    width_cm_actual = (
        width_px / DPI
    ) * 2.54

    height_cm_actual = (
        height_px / DPI
    ) * 2.54

    print("\nCreated:", output_path.name)
    print("Colour mode:", check.mode)
    print(
        "Pixel dimensions:",
        check.size
    )
    print(
        "DPI:",
        check.info.get("dpi")
    )
    print(
        "Physical size:",
        f"{width_cm_actual:.2f} cm × "
        f"{height_cm_actual:.2f} cm"
    )


# ============================================================
# 6. LOAD DATA FOR BLAND–ALTMAN ANALYSIS
# ============================================================

if not MASTER_CSV.exists():
    raise FileNotFoundError(
        f"Could not find:\n{MASTER_CSV}"
    )

master = pd.read_csv(
    MASTER_CSV
)

required_columns = [
    "Plant_ID",
    "Raw_Final_Leaf_Area_cm2",
    "Fiji_Final_Leaf_Area_cm2",
]

missing = [
    column
    for column in required_columns
    if column not in master.columns
]

if missing:
    raise ValueError(
        "Missing columns in master dataset:\n"
        f"{missing}"
    )


# Convert to numeric
master[
    "Raw_Final_Leaf_Area_cm2"
] = pd.to_numeric(
    master[
        "Raw_Final_Leaf_Area_cm2"
    ],
    errors="coerce"
)

master[
    "Fiji_Final_Leaf_Area_cm2"
] = pd.to_numeric(
    master[
        "Fiji_Final_Leaf_Area_cm2"
    ],
    errors="coerce"
)


master = master.dropna(
    subset=[
        "Plant_ID",
        "Raw_Final_Leaf_Area_cm2",
        "Fiji_Final_Leaf_Area_cm2",
    ]
).copy()


master = master[
    master["Plant_ID"].isin(
        PLANT_ORDER
    )
].copy()


master["Plant_ID"] = pd.Categorical(
    master["Plant_ID"],
    categories=PLANT_ORDER,
    ordered=True
)

master = master.sort_values(
    "Plant_ID"
).reset_index(drop=True)


# ============================================================
# 7. BLAND–ALTMAN STATISTICS
# ============================================================

master["HSV_minus_Fiji_cm2"] = (
    master["Raw_Final_Leaf_Area_cm2"]
    -
    master["Fiji_Final_Leaf_Area_cm2"]
)

master["HSV_Fiji_Mean_cm2"] = (
    master["Raw_Final_Leaf_Area_cm2"]
    +
    master["Fiji_Final_Leaf_Area_cm2"]
) / 2.0


differences = master[
    "HSV_minus_Fiji_cm2"
].to_numpy(dtype=float)

ba_means = master[
    "HSV_Fiji_Mean_cm2"
].to_numpy(dtype=float)


mean_bias = np.mean(
    differences
)

difference_sd = np.std(
    differences,
    ddof=1
)

lower_loa = (
    mean_bias
    - 1.96 * difference_sd
)

upper_loa = (
    mean_bias
    + 1.96 * difference_sd
)


# ============================================================
# 8. FIGURE 1: BLAND–ALTMAN
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        WIDTH_IN,
        HEIGHT_IN
    ),
    dpi=DPI
)


ax.scatter(
    ba_means,
    differences,
    s=40
)


ax.axhline(
    mean_bias,
    linewidth=1.5,
    label=(
        f"Bias = "
        f"{mean_bias:.2f} cm²"
    )
)


ax.axhline(
    upper_loa,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"Upper LoA = "
        f"{upper_loa:.2f} cm²"
    )
)


ax.axhline(
    lower_loa,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"Lower LoA = "
        f"{lower_loa:.2f} cm²"
    )
)


for _, row in master.iterrows():

    ax.annotate(
        str(row["Plant_ID"]),
        (
            row[
                "HSV_Fiji_Mean_cm2"
            ],
            row[
                "HSV_minus_Fiji_cm2"
            ],
        ),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=6.5
    )


ax.set_xlabel(
    "Mean of HSV and Fiji leaf area (cm²)",
    fontsize=9
)

ax.set_ylabel(
    "Difference: HSV − Fiji (cm²)",
    fontsize=9
)

ax.set_title(
    "Bland–Altman agreement: raw HSV versus Fiji",
    fontsize=10
)

ax.tick_params(
    axis="both",
    labelsize=8
)

ax.grid(
    alpha=0.25,
    linewidth=0.5
)

ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=7
)


fig.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.13,
    top=0.92
)


save_aob_figure(
    fig,
    "Bland_Altman_HSV_vs_Fiji_AoB_PLANTS.jpg"
)


# ============================================================
# 9. BIOMASS REGRESSION
# ============================================================

validation = pd.DataFrame(
    VALIDATION_DATA,
    columns=[
        "Plant_ID",
        "Fiji_Leaf_Area_cm2",
        "Measured_Final_Dry_Weight_g",
    ]
)


x = validation[
    "Fiji_Leaf_Area_cm2"
].to_numpy(dtype=float)

y = validation[
    "Measured_Final_Dry_Weight_g"
].to_numpy(dtype=float)


slope, intercept = np.polyfit(
    x,
    y,
    1
)


validation[
    "Predicted_Final_Dry_Biomass_g"
] = (
    slope
    * validation[
        "Fiji_Leaf_Area_cm2"
    ]
    + intercept
).clip(lower=0)


observed = validation[
    "Measured_Final_Dry_Weight_g"
].to_numpy(dtype=float)

predicted = validation[
    "Predicted_Final_Dry_Biomass_g"
].to_numpy(dtype=float)


biomass_r2 = r_squared(
    observed,
    predicted
)

biomass_rmse = rmse(
    observed,
    predicted
)


# ============================================================
# 10. FIGURE 2:
#     MEASURED VS PREDICTED DRY BIOMASS
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        WIDTH_IN,
        HEIGHT_IN
    ),
    dpi=DPI
)


ax.scatter(
    observed,
    predicted,
    s=40
)


upper_limit = max(
    observed.max(),
    predicted.max()
)


ax.plot(
    [0, upper_limit],
    [0, upper_limit],
    linestyle="--",
    linewidth=1.2,
    label="1:1 line"
)


for _, row in validation.iterrows():

    ax.annotate(
        row["Plant_ID"],
        (
            row[
                "Measured_Final_Dry_Weight_g"
            ],
            row[
                "Predicted_Final_Dry_Biomass_g"
            ],
        ),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=6.5
    )


ax.set_xlabel(
    "Measured final dry biomass (g)",
    fontsize=9
)

ax.set_ylabel(
    "Predicted final dry biomass (g)",
    fontsize=9
)


ax.set_title(
    "Measured versus predicted final dry biomass\n"
    f"R² = {biomass_r2:.3f}; "
    f"RMSE = {biomass_rmse:.3f} g",
    fontsize=10
)


ax.tick_params(
    axis="both",
    labelsize=8
)


ax.grid(
    alpha=0.25,
    linewidth=0.5
)


ax.legend(
    loc="upper left",
    frameon=False,
    fontsize=7
)


fig.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.13,
    top=0.88
)


save_aob_figure(
    fig,
    "Measured_vs_Predicted_Biomass_AoB_PLANTS.jpg"
)


# ============================================================
# 11. REPORT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FIGURES COMPLETED")
print("=" * 70)

print("\nBLAND–ALTMAN")
print(
    f"Mean bias: {mean_bias:.2f} cm²"
)
print(
    f"Lower LoA: {lower_loa:.2f} cm²"
)
print(
    f"Upper LoA: {upper_loa:.2f} cm²"
)

print("\nBIOMASS")
print(
    f"R²: {biomass_r2:.3f}"
)
print(
    f"RMSE: {biomass_rmse:.3f} g"
)

print(
    f"\nFigures saved in:\n"
    f"{OUTPUT_FOLDER}"
)

print("=" * 70)

input("\nPress ENTER to close.")