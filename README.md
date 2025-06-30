# Snow Cover Mapping

This project is a collaborative snow mapping initiative developed in alignment with the expectations and requirements of **CNR** and using data provided by **ARPA Lombardia**. The objective is to compare existing snow cover products (MODIS, GFSC, S2, S3), implement custom algorithms (machine learning and logic-based), and perform regression analysis to assess consistency and performance.


---

## Final Pairwise Agreement Plot

Below is the result of the pairwise comparison of snow masks over time, illustrating the pixel-wise agreement percentage between sensor products across ISO weeks:

![image](https://github.com/user-attachments/assets/01454d18-20c2-4bf6-87f6-86bb147f933e)


---

## Repository Structure

```
Snow-Cover-Mapping/
├── Snow/                  # Core processing code and notebooks
│   ├── snow_processing.py # Core processing functions
│   ├── main.ipynb         # Main pipeline demo (MODIS)
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── S2-S3.ipynb        # Sentinel-2 vs Sentinel-3 comparison workflow
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore patterns
├── LICENSE                # Project license (MIT)
└── README.md              # Project overview and instructions
```


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Hadikheiri/Snow-Cover-Mapping.git
cd Snow-Cover-Mapping
```

**Requirements**:
- Python ≥ 3.8
- `numpy`, `pandas`, `rasterio`, `matplotlib`, `seaborn`, `scikit-learn`
- Data folders with aligned snow masks and MODIS references

2. Install dependencies:

```bash
pip install numpy pandas rasterio matplotlib seaborn scikit-learn
```



---

## Usage

Most functionality is exposed via `snow_processing.py`. Import and call as needed:

```python
from snow_processing import (
    prepare_modis_mask,
    reproject_resample_visualize,
    aggregate_weekly,
    resample_reproject_gfsc,
    process_s2_weekly,
    reproject_s3_weekly,
    match_raster_grid,
    clip_weekly_to_roi,
    calculate_agreement,
    compute_weekly_statistics,
    compute_pairwise_agreement,
    compute_multisensor_snow_agreement,
    plot_agreement_series
)

# Example: Clean a raw MODIS mask
data_clean_fp = prepare_modis_mask("raw/20230101.tif", "clean/20230101.tif")
```

### Key Functions

- **extract\_weeks\_from\_filenames**: Extract ISO-week keys from filenames or folder names.
- **check\_all\_products\_side\_by\_side**: Compare weekly coverage across multiple products.
- **prepare\_modis\_mask**: Remap raw MODIS snow mask to binary/no-data values.
- **reproject\_resample\_visualize**: Reproject & resample rasters, with optional preview.
- **aggregate\_weekly / aggregate\_weekly\_gfsc**: Build ISO-week composites for raw or GFSC data.
- **process\_s2\_weekly**: Split bi-weekly Sentinel-2 masks into weekly composites.
- **reproject\_s3\_weekly**: Aggregate and reproject Sentinel-3 daily masks into weekly composites.
- **match\_raster\_grid**: Align one raster to the grid of a reference.
- **visual\_compare\_rasters / visual\_compare\_rasters\_strict**: Side-by-side, overlay, or difference visualization.
- **clip\_weekly\_to\_roi**: Clip weekly TIFFs to a region of interest.
- **calculate\_agreement**: Compute pixel-wise agreement between two masks.
- **compute\_weekly\_statistics**: Summary stats (snow area, coverage) for clipped weekly products.
- **compute\_pairwise\_agreement**: Pairwise agreement metrics and difference maps across products.
- **compute\_multisensor\_snow\_agreement**: Consensus maps across multiple products.
- **plot\_agreement\_series**: Time series of agreement between two products.

---

## Jupyter Notebooks

**main.ipynb - classification and comparison**:  
   Open `main.ipynb` to:
   - Preprocess and align data (Sentinel, GFSC, MODIS)
   - Apply Random Forest classifier on raw band combinations
   - Generate logic-based masks (simplified Let-It-Snow)
   - Evaluate performance (accuracy, precision, recall, F1)
   - Plot confusion matrices and save metrics
- **EDA.ipynb**:
 Exploratory data analysis on processed weekly products: summary statistics, visualization of snow extent, and coverage comparisons.
- **S2-S3.ipynb**:
 Compares Sentinel-2 and Sentinel-3 weekly snow products: coverage table, spatial agreement analysis, and time series plotting.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add awesome feature"`)
4. Push to your fork (`git push origin feature/YourFeature`)
5. Open a Pull Request

---
## 📝 Notes

- Data strictly follows the guidelines from **CNR** and was provided by **ARPA Lombardia**.
- MODIS masks are used as the reference product for validation.
- The Let-It-Snow logic was simplified for compatibility with binary masks.
---
## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

##  References

- Let-It-Snow Algorithm: [https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow](https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow)
- ARPA Lombardia Data Portal: [https://www.arpalombardia.it](https://www.arpalombardia.it)
- CNR Collaboration Guidelines (internal documentation)


-----



## Contact

For questions or support, please open an issue or contact the maintainer:

- **First Contributer:** Hadi Kheiri Gharajeh, [hadi.kheiri@mail.polimi.it](mailto\:hadi.kheiri@mail.polimi.it)
- **Second Contributer:** Ola Elwasila Abdelrahman Yousif, [olaelwasila@mail.polimi.it](mailto\:olaelwasila@mail.polimi.it)


