# Brassica rapa Raspberry Pi phenotyping — supplemental repository notes

This supplement accompanies existing acquisition, emergence, growth-figure and README files already uploaded by the author.

## Contents
- `analysis/original/`: original HSV extraction source (preserved for provenance).
- `analysis/quality_control/`: later revised script with quality-control overlays and checks; **not verified to be the script that generated the reported CSV**.
- `analysis/validation/`: Figure 9A, Figure 9B and Figure 10 analysis code. Figure 9B script contains an **older biomass calculation** not used for final Figure 10.
- `data/`: image-level and daily trait series, validation master dataset, and leaf-area statistics. Fine-tuned leaf-area values are **not independent of Fiji reference values** and must not be used to claim independent validation.
- `roi/daily_json/`: unchanged daily ROIs, 8–20 and 22 June 2026 (no 21 June file).
- `roi/previews/`: supplied annotated previews, not original raw image series.
- `quality_control/june22/`: diagnostic audit for one reference photograph, not a whole-experiment reanalysis.

## Experimental timeline
Planting 2 June 2026 = 0 DAP; first visible emergence 7 June = 5 DAP; quantitative measurements start 8 June = 6 DAP; 21 June = 19 DAP has no daily ROI; 22 June = 20 DAP.

## Settings requiring provenance check before a formal release
1. The historical camera capture interval and operating hours must be confirmed from actual capture records; later script changes should not be attributed to the experiment without evidence.
2. Original HSV extraction uses H 35–85, S 60–255, V 40–255, morphology 5×5 and minimum component area 80 px. Reconcile the manuscript's earlier V=60 statement.
3. Calibration 38.46 pixels/cm requires resolution-specific confirmation; the QC script may not reproduce the published values.
4. Historical T3/T4 swap before 17 June 2026 16:00:09 must not be applied twice.
5. ROIs at large plant sizes can intersect neighbouring foliage; QC flags are review prompts, not automatic data exclusions.
6. Software versions reported by the author: Python 3.10.11, OpenCV 4.11.0, Fiji/ImageJ 1.54p. Other dependencies in `requirements.txt` are provisional and should be checked against the environment used.
7. Source photographs for the full experiment are not included here; one diagnostic reference photograph is included. Full extraction from raw images has not been reproduced.

**Status:** Repository preparation, not an assertion of end-to-end validated reproducibility. Preserve published analyses and original files; document changes transparently.
