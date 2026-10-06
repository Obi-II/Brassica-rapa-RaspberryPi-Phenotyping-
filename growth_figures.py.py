"""Regenerate manuscript Figures 5–8 with a single planting-based time origin.
No data are fabricated for missing days, and the source CSV is unchanged.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

BASE_FOLDER = Path(r'C:\Users\emman\Desktop\Plant_Growth_Analysis')
DAILY_CSV = BASE_FOLDER / 'daily_plant_traits.csv'
OUTPUT_FOLDER = BASE_FOLDER / 'Revised_DAP_Figures'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PLANTING_DATE = pd.Timestamp('2026-06-02')
HARVEST_DATE = pd.Timestamp('2026-06-22')
PIXELS_PER_CM = 38.46
DPI = 600
TRAY_ORDER = ['T1', 'T2', 'T3', 'T4']
PLANT_ORDER = ['T1_P1','T1_P2','T1_P3','T1_P4',
               'T2_P1','T2_P3','T2_P4',
               'T3_P1','T3_P2','T3_P4',
               'T4_P1','T4_P2','T4_P3','T4_P4']
EMPTY_POSITIONS = {'T2_P2', 'T3_P3'}

if not DAILY_CSV.exists():
    raise FileNotFoundError(f'Cannot find {DAILY_CSV}')
daily = pd.read_csv(DAILY_CSV)
required = {'Date', 'Tray', 'Plant_ID', 'Area_Median_pixels', 'Width_Median_pixels'}
if not required.issubset(daily.columns):
    raise ValueError(f'Missing CSV columns: {sorted(required - set(daily.columns))}')
daily['Date'] = pd.to_datetime(daily['Date'], errors='raise').dt.normalize()
for col in ['Area_Median_pixels', 'Width_Median_pixels']:
    daily[col] = pd.to_numeric(daily[col], errors='coerce').fillna(0)
daily = daily[~daily['Plant_ID'].isin(EMPTY_POSITIONS)].copy()
if not daily['Date'].between(PLANTING_DATE, HARVEST_DATE).all():
    raise ValueError('Some daily dates fall outside 2–22 June; inspect the CSV.')
# Derive DAP from actual dates, not the previous 8 June Day-0 index.
daily['DAP'] = (daily['Date'] - PLANTING_DATE).dt.days
daily['Leaf_Area_cm2'] = daily['Area_Median_pixels'] / PIXELS_PER_CM**2
daily['Plant_Width_cm'] = daily['Width_Median_pixels'] / PIXELS_PER_CM

# Report the observed date coverage without adding data to the plots.
observed = set(daily['DAP'].dropna().astype(int))
print('Observed DAP:', sorted(observed))
print('Days with no daily summary:', sorted(set(range(6, 21)) - observed))
print('Plant counts by tray:', daily.groupby('Tray')['Plant_ID'].nunique().to_dict())

def new_axes():
    return plt.subplots(figsize=(17 / 2.54, 10 / 2.54), dpi=100)

def save(fig, filename):
    out = OUTPUT_FOLDER / filename
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.92)
    fig.savefig(out, dpi=DPI, format='jpg', facecolor='white')
    with Image.open(out) as im:
        im.convert('RGB').save(out, 'JPEG', quality=100, dpi=(DPI, DPI))
    print('Saved:', out)
    plt.close(fig)

def common_axis(ax, ylabel, title):
    ax.set_xlabel('Days after planting (DAP)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(5.5, 20.5)
    ax.set_xticks(np.arange(6, 21, 2))
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

# Insert NaNs for unobserved dates in PLOTTING SERIES ONLY, to prevent
# Matplotlib drawing a false uninterrupted line across a missing day.
all_growth_days = pd.Index(range(6, 21), name='DAP')

def individual_plot(value, ylabel, title, filename):
    fig, ax = new_axes()
    for plant in PLANT_ORDER:
        sub = daily[daily['Plant_ID'] == plant].groupby('DAP')[value].median()
        if sub.empty:
            continue
        sub = sub.reindex(all_growth_days)
        ax.plot(sub.index, sub.to_numpy(), marker='o', markersize=2.8,
                linewidth=1.2, label=plant)
    common_axis(ax, ylabel, title)
    ax.legend(title='Plant ID', ncol=2, frameon=False, fontsize=6.5,
              title_fontsize=7, loc='upper left', handlelength=1.5, columnspacing=0.8)
    save(fig, filename)

def tray_plot(value, ylabel, title, filename):
    summary = daily.groupby(['Tray','DAP'])[value].agg(['mean','std'])
    fig, ax = new_axes()
    for tray in TRAY_ORDER:
        if tray not in summary.index.get_level_values('Tray'):
            continue
        sub = summary.loc[tray].reindex(all_growth_days)
        means = sub['mean'].to_numpy(dtype=float)
        sd = sub['std'].fillna(0).to_numpy(dtype=float)
        x = all_growth_days.to_numpy(dtype=float)
        line = ax.plot(x, means, marker='o', markersize=3.5,
                       linewidth=1.5, label=tray)[0]
        ax.fill_between(x, np.maximum(0, means - sd), means + sd,
                        color=line.get_color(), alpha=0.15)
    common_axis(ax, ylabel, title)
    ax.legend(title='Tray', frameon=False, fontsize=7.5,
              title_fontsize=8, loc='upper left')
    save(fig, filename)

individual_plot('Leaf_Area_cm2', 'Leaf area (cm²)',
                'Leaf-area growth curves for individual plants', 'Figure_5_DAP.jpg')
tray_plot('Leaf_Area_cm2', 'Mean leaf area (cm²)',
          'Mean leaf-area growth by tray', 'Figure_6_DAP.jpg')
individual_plot('Plant_Width_cm', 'Plant width (cm)',
                'Plant-width growth curves for individual plants', 'Figure_7_DAP.jpg')
tray_plot('Plant_Width_cm', 'Mean plant width (cm)',
          'Mean plant-width growth by tray', 'Figure_8_DAP.jpg')
print('Finished. CSV source unchanged. Output:', OUTPUT_FOLDER)
