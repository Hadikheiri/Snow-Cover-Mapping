# Snow Cover Mapping in Lombardy, Italy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)

A modular pipeline to harmonize, compare, and classify multi‐sensor snow‐cover products (MODIS, Copernicus GFSC, Sentinel-2, Sentinel-3) over the Lombardy Alps. This project was developed under the supervision of the National Research Council of Italy (CNR) using data provided by ARPA Lombardia.  

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Data Layout](#data-layout)  
- [Quick Start / Usage](#quick-start--usage)  
  - [Preprocessing & Aggregation](#preprocessing--aggregation)  
  - [Agreement & Bias Analysis](#agreement--bias-analysis)  
  - [Classification & Regression](#classification--regression)  
- [Jupyter Notebooks](#jupyter-notebooks)  
- [Results Gallery](#results-gallery)  
- [Contributing](#contributing)  
- [License & Citation](#license--citation)  
- [Acknowledgements](#acknowledgements)  
- [Contact](#contact)  

---

## Overview

Snow cover governs hydrology, ecology, and climate feedbacks in mountain regions. This repository implements a fully-automated workflow to:

1. **Preprocess** and **harmonize** four snow products—MODIS (500 m), GFSC (60 m), Sentinel-2 (20 m), Sentinel-3 (300 m)—to a unified 60 m grid and ISO-week temporal cadence.  
2. **Compare** them via pixel-wise agreement, consensus maps, and area-bias time series.  
3. **Classify** snow presence using both a Random Forest model and a simplified Let-It-Snow logic.  
4. **Quantify** continuous agreement with MODIS through linear regression (slope & correlation).  

All code is written in Python, leverages open-source geospatial libraries, and is designed for reproducibility and extension.

---

## Features

- **Multi-sensor preprocessing**: reprojection, resampling, quality filtering, weekly stacking  
- **Agreement analysis**: pairwise % agreement, multi-sensor consensus, spatial discrepancy maps  
- **Area-bias evaluation**: time-series of km² differences between products  
- **Snow classification**: Random Forest fusion vs. logic-based voting  
- **Regression module**: correlation & slope against MODIS reference  
- **Notebook demos**: exploratory analysis, focused S2 vs. S3 comparison, end-to-end pipeline  

---

## Repository Structure

```text
Snow-Cover-Mapping/
├── Snow/                            # Core code & notebooks
│   ├── snow_processing.py           # Data‐processing functions
│   ├── main.ipynb                   # End-to-end pipeline walkthrough
│   ├── EDA.ipynb                    # Exploratory Data Analysis & plots
│   └── S2-S3.ipynb                  # Sentinel-2 vs. Sentinel-3 comparison
├── data/                            # (Local only; not tracked)
│   ├── raw/                         # Original sensor TIFFs
│   ├── processed/                   # Intermediate files (weekly, clipped)
│   └── roi/                         # Region‐of‐interest shapefile
├── figures/                         # Generated plots & maps
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT license
└── README.md                        # Project documentation
```


---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hadikheiri/Snow-Cover-Mapping.git
   cd Snow-Cover-Mapping
   ```

2. **Create and activate a Python 3.13+ virtual environment:**
   ```bash
   python3.13 -m venv .env
   # macOS / Linux
   source .env/bin/activate
   # Windows
   .env\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Layout

Place your raw data under `data/raw/` following this structure:

```bash
data/raw/
├── MODIS/
│   └── YYYYMMDD_MODIS.tif
├── GFSC/
│   ├── YYYYMMDD_GF.tif
│   └── YYYYMMDD_QC.tif
├── S2/
│   ├── YYYYMMDD_S2.tif
│   └── YYYYMMDD_datemap.tif
└── S3/
    └── YYYYMMDD_S3.tif
```

After preprocessing, outputs are saved as:

- `data/processed/weekly/<PRODUCT>/<ISO_WEEK>.tif`
- `data/processed/clipped/<PRODUCT>/<ISO_WEEK>.tif`

---

## Quick Start / Usage

Below are Python snippets illustrating core workflows. Adapt them into scripts or Jupyter cells.

### Preprocessing & Aggregation

```python
from Snow.snow_processing import (
    prepare_modis_mask,
    reproject_resample_visualize,
    resample_reproject_gfsc,
    process_s2_weekly,
    reproject_s3_weekly,
    aggregate_weekly,
    aggregate_weekly_gfsc
)

# 1. Clean & binarize a raw MODIS mask
prepare_modis_mask(
    src_path="data/raw/MODIS/20220101_MODIS.tif",
    dst_path="data/processed/modis_clean/2022_W02.tif"
)

# 2. Stack daily files into weekly composites
aggregate_weekly(
    input_folder="data/processed/modis_clean",
    output_folder="data/processed/weekly/MODIS",
    method="max"
)
```

### Agreement & Bias Analysis

```python
from Snow.snow_processing import compute_pairwise_agreement

df = compute_pairwise_agreement(
    aligned_root="data/processed/clipped",
    products=["MODIS", "GFSC", "S2", "S3"],
    common_weeks=[...list of ISO weeks...]
)
```

### Classification & Regression

```python
from Snow.snow_processing import run_regression_batch, logic_based_mask
# See `main.ipynb` for orchestrated Random Forest & logic-based workflows.
```

---

## Jupyter Notebooks

- **`Snow/main.ipynb`** — Full pipeline: preprocess, classify, evaluate, and visualize.
- **`Snow/EDA.ipynb`**  — Exploratory plots: consensus maps, agreement & bias time series.
- **`Snow/S2-S3.ipynb`** — Dedicated Sentinel-2 vs Sentinel-3 comparison.

> Each notebook’s first cell outlines the paths you need to adjust for your local setup.

---

## Results Gallery

**Agreement (%) between each sensor pair over 23 ISO weeks.**



**Difference in snow-covered area (km²) between sensor pairs.**

More maps and plots are available in the `figures/` directory.

---

## Contributing

We welcome improvements and bug fixes:

1. Fork the repo
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to your fork: `git push origin feature/YourFeature`
5. Open a Pull Request

Please follow PEP8 styling and include tests for new functionality.

---

## License & Citation

This project is distributed under the [MIT License](LICENSE).

If you use this work, please cite:

> H. Kheiri Gharajeh & O. Y. Yousif (2025). *Implementation of a Multi-Sensor Algorithm for Time Series Snow Cover Mapping in Lombardy, Italy.* Geoinformatics Project, Politecnico di Milano.

---

## Acknowledgements

- **Advisors:** Prof. Giovanna Venuti & Prof. Daniela Stroppiana
- **Data Provider:** ARPA Lombardia
- **Technical Guidelines:** National Research Council of Italy (CNR)

---

## Contact

For questions or support, please open an issue or contact:

- **Hadi Kheiri Gharajeh** — [hadi.kheiri@mail.polimi.it](mailto\:hadi.kheiri@mail.polimi.it)
- **Ola Elwasila A. Yousif** — [olaelwasila@mail.polimi.it](mailto\:olaelwasila@mail.polimi.it)




