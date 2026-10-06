"""Figure 4: photographic first-emergence event, not emergence percentage."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_FOLDER = Path(r'C:\Users\emman\Desktop\Plant_Growth_Analysis')
OUTPUT_FOLDER = BASE_FOLDER / 'Revised_DAP_Figures'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Planting / monitoring start: 2 June 2026 = 0 DAP.
# First visually observed seedling: 7 June 2026 = 5 DAP.
# Harvest: 22 June 2026 = 20 DAP.
EMERGENCE_DAP = 5
HARVEST_DAP = 20
DPI = 600

fig, ax = plt.subplots(figsize=(17 / 2.54, 10 / 2.54), dpi=100)
# A binary indicator of whether first emergence has occurred by this day.
# This is NOT the percentage of plants emerged.
days = np.arange(0, HARVEST_DAP + 1)
observed_event = (days >= EMERGENCE_DAP).astype(int)
ax.step(days, observed_event, where='post', linewidth=1.8, color='#246c9c')
ax.scatter([EMERGENCE_DAP], [1], s=42, color='#246c9c', zorder=5)
ax.axvline(EMERGENCE_DAP, linestyle='--', linewidth=1.2, color='#a03e34',
           label='First visible emergence: 5 DAP (7 June)')
ax.set(title='First visible seedling emergence', xlabel='Days after planting (DAP)',
       ylabel='First emergence observed')
ax.set_xticks(days)
ax.set_yticks([0, 1], ['Not yet observed', 'Observed'])
ax.set_xlim(0, HARVEST_DAP)
ax.set_ylim(-0.15, 1.2)
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=8)
ax.grid(alpha=0.25)
ax.legend(loc='lower right', fontsize=8, frameon=False)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.92)

out = OUTPUT_FOLDER / 'Figure_4_First_Emergence_DAP.jpg'
fig.savefig(out, dpi=DPI, format='jpg', facecolor='white')
with Image.open(out) as im:
    im.convert('RGB').save(out, 'JPEG', quality=100, dpi=(DPI, DPI))
plt.show()
plt.close(fig)
print('Saved:', out)
