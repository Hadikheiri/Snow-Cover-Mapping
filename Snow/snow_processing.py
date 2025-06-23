# -------------------- extract_weeks_from_filenames --------------------
import pandas as pd
import os
from datetime import datetime
import re

def extract_weeks_from_filenames(folder, pattern_list=['.tif', '.img'], extract_from_folder=False):
    """
    Returns a sorted list of 'YYYY_Www' ISO-week strings found in `folder`.
    """
    week_keys = set()
    entries = os.listdir(folder)

    for entry in entries:
        name = entry if extract_from_folder else entry
        if not extract_from_folder and not any(name.endswith(ext) for ext in pattern_list):
            continue

        digits = ''.join([c for c in name if c.isdigit()])
        
        # 1️⃣ Try to parse full date (e.g., 20220315)
        if len(digits) >= 8:
            try:
                date = datetime.strptime(digits[:8], "%Y%m%d")
                iso = date.isocalendar()
                week_keys.add(f"{iso.year}_W{iso.week:02d}")
                continue
            except:
                pass

        # 2️⃣ Try to parse week-based names (e.g., 2022_W05)
        match = re.search(r"(20\d{2})_W(\d{2})", name)
        if match:
            year, week = match.groups()
            week_keys.add(f"{year}_W{week}")

    return sorted(week_keys)


# -------------------- check_all_products_side_by_side --------------------

import pandas as pd

def check_all_products_side_by_side(product_folders,
                                    start_year,
                                    end_year):
    """
    Compares coverage across multiple products side-by-side.
    product_folders = [
        ("MODIS", r"path\to\MODIS", False),
        ("GFSC", r"path\to\GFSC", True),  # True = use folder name as date
        ...
    ]
    """
    product_weeks = {}
    for name, path, extract_from_folder in product_folders:
        print(f"🔍 Scanning {name} → {path}")
        weeks = extract_weeks_from_filenames(path,
                                             extract_from_folder=extract_from_folder)
        if name not in product_weeks:
            product_weeks[name] = set()
        product_weeks[name].update(weeks)
        print(f"   ➤ Found {len(weeks)} unique weeks")

    # Full ISO-week calendar
    full_weeks = [f"{y}_W{w:02d}"
                  for y in range(start_year, end_year + 1)
                  for w in range(1, 54)]

    # Build table
    rows = []
    for wk in full_weeks:
        row = {"Week": wk}
        common = True
        for prod in product_weeks:
            present = wk in product_weeks[prod]
            row[prod] = "✅" if present else "❌"
            if not present:
                common = False
        row["Common_Week"] = "✅" if common else ""
        rows.append(row)

    return pd.DataFrame(rows)



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import rasterio
import numpy as np

def prepare_modis_mask(src_fp, clean_fp):
    """
    Read the raw MODIS snow mask (values 1=no-snow, 2=snow),
    remap to 0/1/255, and write out a GeoTIFF on the original MODIS grid.

    Mapping:
      • 1 → 0   (clear / no-snow)
      • 2 → 1   (snow)
      • any other value → 255 (nodata / cloud)
    """
    with rasterio.open(src_fp) as src:
        data = src.read(1)
        meta = src.meta.copy()

    # Initialize everything to nodata
    clean = np.full_like(data, 255, dtype=np.uint8)

    # Remap
    clean[data == 1] = 0   # clear → 0
    clean[data == 2] = 1   # snow  → 1

    # Write out with nodata=255
    meta.update(dtype='uint8', nodata=255)
    with rasterio.open(clean_fp, 'w', **meta) as dst:
        dst.write(clean, 1)

    return clean_fp



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

def reproject_resample_visualize(
    src_path: str,
    dst_path: str,
    dst_crs: str = 'EPSG:32632',
    dst_res: tuple = (60, 60),
    resampling_method=Resampling.nearest,
    visualize: bool = True,
    downsample_factor: int = 10
):
    """
    Reproject & resample a single-band raster to a target CRS/resolution,
    preserving nodata throughout, then optionally show a quick preview.
    """
    # Open source and compute the target grid & metadata
    with rasterio.open(src_path) as src:
        src_crs       = src.crs
        src_transform = src.transform
        nodata_val    = src.nodata if src.nodata is not None else 255
        dtype         = src.dtypes[0]
        band_count    = src.count

        # Compute new transform + shape
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs,
            src.width, src.height,
            *src.bounds,
            resolution=dst_res
        )

        # Build output metadata (including nodata and CRS)
        out_meta = src.meta.copy()
        out_meta.update({
            'driver'    : 'GTiff',
            'crs'       : dst_crs,
            'transform' : transform,
            'width'     : width,
            'height'    : height,
            'dtype'     : dtype,
            'count'     : band_count,
            'nodata'    : nodata_val,
            'compress'  : 'deflate'
        })

        # Now that src is still open, create & write into dst
        with rasterio.open(dst_path, 'w', **out_meta) as dst:
            for b in range(1, band_count + 1):
                # Pre-fill output array with nodata
                dest = np.full((height, width), nodata_val, dtype=dtype)

                reproject(
                    source        = rasterio.band(src, b),
                    destination   = dest,
                    src_transform = src_transform,
                    src_crs       = src_crs,
                    src_nodata    = nodata_val,
                    dst_transform = transform,
                    dst_crs       = dst_crs,
                    dst_nodata    = nodata_val,
                    resampling    = resampling_method
                )

                dst.write(dest, b)

    print(f"✅ Resampled & reprojected saved to: {dst_path}")

    # Optional downsampled preview
    if visualize:
        with rasterio.open(dst_path) as preview:
            arr = preview.read(1)[::downsample_factor, ::downsample_factor]
        plt.imshow(arr, cmap='gray')
        plt.title(f"Preview: {os.path.basename(dst_path)}")
        plt.colorbar()
        plt.axis('off')
        plt.show()



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def aggregate_weekly(
    directory: str,
    pattern: str = '.tif',
    output_dir: str | None = None,
    *,
    method: str = 'max',             # 'max' | 'mean' | 'median'
    threshold: float | None = None,  # e.g. 0.5 for binary; None → keep raw
    qc_suffix: str = '_QC.tif',      # companion QC files
    skip_existing: bool = True,
    summary: bool = True,
    parallel: bool = True,
    max_workers: int = 4
) -> list[str]:
    """
    Aggregate snow rasters into ISO-week composites.

    - Groups files by ISO-week (from filenames or S2 datemap).
    - Masks any original-nodata and (for threshold<=1) any >1 values to NaN.
    - Applies optional QC mask (only qc==0 kept).
    - Aggregates via max|mean|median.
    - Applies `threshold` to produce a final binary 0/1 mask.
    - Marks truly-NaN pixels (no valid obs) as GeoTIFF nodata (default 255).
    """
    # 0. collect candidate files
    all_files = [f for f in os.listdir(directory)
                 if f.endswith(pattern) and '_datemap' not in f]
    all_files.sort()
    # 1. group by ISO-week
    weekly_groups: dict[str, list[str]] = defaultdict(list)
    for fname in all_files:
        base = fname[:-len(pattern)]
        fpath = os.path.join(directory, fname)
        # S2 datemap special case
        dm = os.path.join(directory, base + '_datemap.tif')
        if os.path.exists(dm):
            try:
                date0 = datetime.strptime(''.join(c for c in base if c.isdigit())[:8], '%Y%m%d')
                with rasterio.open(dm) as dsrc:
                    datemap = dsrc.read(1)
                for day in range(1, int(np.nanmax(datemap)) + 1):
                    if np.any(datemap == day):
                        d = date0 + timedelta(days=day - 1)
                        key = d.isocalendar()
                        wk = f"{key.year}_W{key.week:02d}"
                        weekly_groups[wk].append(fpath)
                continue
            except:
                pass
        # default: parse date portion from filename
        try:
            date0 = datetime.strptime(''.join(c for c in fname if c.isdigit())[:8], '%Y%m%d')
            iso = date0.isocalendar()
            wk = f"{iso.year}_W{iso.week:02d}"
            weekly_groups[wk].append(fpath)
        except:
            print(f"⚠️ Could not parse date from {fname}")

    # ensure output dir exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed = skipped = 0

    def _build_week(week_key: str, paths: list[str]) -> str | None:
        nonlocal processed, skipped
        out_fp = os.path.join(output_dir, f"{week_key}.tif")
        if skip_existing and os.path.exists(out_fp):
            skipped += 1
            return None

        print(f"📦 Aggregating {len(paths)} → {week_key} ({method})")
        stack = []
        profile = None

        for p in paths:
            with rasterio.open(p) as src:
                band = src.read(1).astype('float32')
                nod = src.nodata
                # mask original nodata
                if nod is not None:
                    band = np.where(band == nod, np.nan, band)
                # mask any >1 for binary products (threshold ≤1)
                if threshold is not None and threshold <= 1:
                    band = np.where(band > 1, np.nan, band)
                # apply QC mask if present
                qc_p = p.replace(pattern, qc_suffix)
                if os.path.exists(qc_p):
                    with rasterio.open(qc_p) as qcsrc:
                        qc = qcsrc.read(1,
                            out_shape=band.shape,
                            resampling=Resampling.nearest
                        )
                    band = np.where(qc == 0, band, np.nan)

                stack.append(band)
                if profile is None:
                    profile = src.profile.copy()

        arr = np.stack(stack, axis=0)  # shape (n_dates, rows, cols)

        # 3. aggregate
        if method == 'max':
            result = np.nanmax(arr, axis=0)
        elif method == 'mean':
            result = np.nanmean(arr, axis=0)
        elif method == 'median':
            result = np.nanmedian(arr, axis=0)
        else:
            raise ValueError("method must be 'max', 'mean', or 'median'")

        # 4. threshold to binary if requested
        if threshold is not None:
            result = (result >= threshold).astype('uint8')
        else:
            if np.nanmax(result) <= 1:
                result = result.astype('uint8')

        # 5. preserve true nodata: any pixel always NaN across the stack
        nodata_mask = np.all(np.isnan(arr), axis=0)
        nodval = profile.get('nodata')
        if nodval is None:
            nodval = 255
        result = np.where(nodata_mask, nodval, result).astype('uint8')

        # 6. write out
        profile.update(dtype='uint8', count=1, nodata=nodval)
        with rasterio.open(out_fp, 'w', **profile) as dst:
            dst.write(result, 1)

        processed += 1
        return out_fp

    # run in parallel or serial
    outputs: list[str] = []
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_build_week, wk, paths): wk
                       for wk, paths in weekly_groups.items()}
            for f in as_completed(futures):
                out = f.result()
                if out:
                    outputs.append(out)
    else:
        for wk, paths in weekly_groups.items():
            out = _build_week(wk, paths)
            if out:
                outputs.append(out)

    if summary:
        print(f"✔️ Processed: {processed}, Skipped: {skipped}, Outputs: {len(outputs)}")

    return outputs




#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def resample_reproject_gfsc(
    input_dir: str,
    output_dir: str,
    dst_crs: str = 'EPSG:32632',
    dst_res: tuple = (60, 60),
    threshold: float = 20.0,
    gf_suffix: str = '_GF.tif',
    qc_suffix: str = '_QC.tif',
    acceptable_qc: set[int] = {0, 1, 2, 3}
) -> list[str]:
    """
    1) Find all *_GF.tif and *_QC.tif pairs in input_dir.
    2) Load GF and QC bands, masking out original nodata and fill (255) as NaN.
    3) Reproject & resample GF (bilinear) and QC (nearest) to (dst_crs, dst_res), pre-filling
       destination arrays to preserve nodata (GF→NaN, QC→invalid by default).
    4) Mask out pixels where QC is not in acceptable_qc or GF is NaN.
    5) Threshold GF >= threshold → 1, < threshold → 0, NaN → 255 (nodata).
    6) Write a uint8 GeoTIFF with nodata=255, returning list of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_files = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(gf_suffix):
            continue

        base = fname[:-len(gf_suffix)]
        gf_fp = os.path.join(input_dir, base + gf_suffix)
        qc_fp = os.path.join(input_dir, base + qc_suffix)
        if not os.path.exists(qc_fp):
            continue

        # 2a. Read GF
        with rasterio.open(gf_fp) as gf_src:
            gf = gf_src.read(1).astype('float32')
            gf_nodata = gf_src.nodata
            profile = gf_src.profile.copy()
            src_crs = gf_src.crs
            src_transform = gf_src.transform
            bounds = gf_src.bounds
            width = gf_src.width
            height = gf_src.height

        # Mask original nodata/fill
        if gf_nodata is not None:
            gf[gf == gf_nodata] = np.nan
        gf[gf == 255] = np.nan

        # 2b. Read QC
        with rasterio.open(qc_fp) as qc_src:
            qc = qc_src.read(1).astype('uint8')
            qc_nodata = qc_src.nodata if qc_src.nodata is not None else 255

        # 3. Compute target grid
        transform, w, h = calculate_default_transform(
            src_crs, dst_crs, width, height, *bounds, resolution=dst_res
        )

        # Pre-fill dest with NaN/QC_nodata
        gf_dst = np.full((h, w), np.nan, dtype='float32')
        qc_dst = np.full((h, w), qc_nodata, dtype='uint8')

        # Reproject
        reproject(
            source=gf,
            destination=gf_dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        reproject(
            source=qc,
            destination=qc_dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        # 4. Mask via QC whitelist + GF NaNs
        valid_qc = np.isin(qc_dst, list(acceptable_qc))
        valid = valid_qc & (~np.isnan(gf_dst))
        gf_dst[~valid] = np.nan

        # 5. Threshold to binary + set nodata
        bin_dst = np.full((h, w), 255, dtype='uint8')
        bin_dst[gf_dst >= threshold] = 1
        bin_dst[(gf_dst < threshold) & (~np.isnan(gf_dst))] = 0

        # 6. Write out
        profile.update({
            'driver': 'GTiff',
            'crs': dst_crs,
            'transform': transform,
            'width': w,
            'height': h,
            'dtype': 'uint8',
            'count': 1,
            'nodata': 255
        })
        out_fp = os.path.join(output_dir, f"{base}.tif")
        with rasterio.open(out_fp, 'w', **profile) as dst:
            dst.write(bin_dst, 1)

        out_files.append(out_fp)
        kept_pct = valid.sum() / valid.size * 100
        print(f"✅ Daily GFSC → {out_fp} (kept {kept_pct:.1f}% pixels via QC∈{sorted(acceptable_qc)})")

    return out_files



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



def aggregate_weekly_gfsc(
    input_dir: str,
    output_dir: str,
    *,
    method: str = 'max',       # 'max' | 'mean' | 'median'
    skip_existing: bool = True,
    summary: bool = True,
    parallel: bool = True,
    max_workers: int = 2
) -> list[str]:
    """
    Takes folder of binarized daily GFSC (0|1|255) and builds ISO-week composites:
     • groups by date in filename → YYYY_Www
     • stacks days, applies chosen method (nan-aware)
     • any pixel never observed → 255
     • writes uint8 {0,1,255}, nodata=255
    """
    from collections import defaultdict
    from datetime import datetime
    import os, numpy as np, rasterio
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # group by ISO-week
    groups = defaultdict(list)
    for fn in os.listdir(input_dir):
        if not fn.endswith('.tif'): continue
        digits = ''.join(c for c in fn if c.isdigit())
        if len(digits) < 8: continue
        dt = datetime.strptime(digits[:8], '%Y%m%d')
        wk = dt.isocalendar()
        key = f"{wk.year}_W{wk.week:02d}"
        groups[key].append(os.path.join(input_dir, fn))

    os.makedirs(output_dir, exist_ok=True)
    processed = skipped = 0
    outputs = []

    def _build(week_key, paths):
        nonlocal processed, skipped
        out_fp = os.path.join(output_dir, f"{week_key}.tif")
        if skip_existing and os.path.exists(out_fp):
            skipped += 1
            return None

        stack = []
        meta = None
        for p in paths:
            with rasterio.open(p) as src:
                arr = src.read(1).astype('float32')
                nod = src.nodata
                arr[arr == nod] = np.nan
                stack.append(arr)
                if meta is None:
                    meta = src.meta.copy()

        arr3d = np.stack(stack, axis=0)
        if method == 'max':
            res = np.nanmax(arr3d, axis=0)
        elif method == 'mean':
            res = np.nanmean(arr3d, axis=0)
        elif method == 'median':
            res = np.nanmedian(arr3d, axis=0)
        else:
            raise ValueError("method must be 'max','mean' or 'median'")

        # pixels never seen → nodata
        nodmask = np.all(np.isnan(arr3d), axis=0)
        res[nodmask] = meta.get('nodata', 255)
        res = res.astype('uint8')

        meta.update(dtype='uint8', count=1, nodata=255)
        with rasterio.open(out_fp, 'w', **meta) as dst:
            dst.write(res, 1)

        processed += 1
        return out_fp

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut = {ex.submit(_build, wk, paths): wk for wk, paths in groups.items()}
            for f in as_completed(fut):
                if out := f.result():
                    outputs.append(out)
    else:
        for wk, paths in groups.items():
            if out := _build(wk, paths):
                outputs.append(out)

    if summary:
        print(f"✔️ Weekly: processed={processed}, skipped={skipped}, outputs={len(outputs)}")
    return outputs


#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import os
import re
import numpy as np
import rasterio
from datetime import datetime, timedelta
from collections import defaultdict
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds

def process_s2_weekly(
    input_root: str,
    output_dir: str,
    dst_crs: str = 'EPSG:32632',
    resolution: float = 60.0
) -> list[str]:
    """
    Processes bi-weekly S2 snow products into weekly composites.

    1.  Walks input_root to find *_SnowMask_latest.tif and its paired *_datemap.tif.
    2.  Extracts the 14-day period from the filename (e.g., 20220103_20220116).
    3.  Uses the datemap (values 1-14) to split the snow mask into two weekly arrays.
        - Week A: Corresponds to the first 7 days of the period.
        - Week B: Corresponds to the second 7 days of the period.
    4.  Aggregates all data for the same ISO week into a single array, handling overlaps.
        - Merging rule: No-snow (0) only fills nodata pixels, but snow (1) always overwrites.
    5.  Reprojects each final weekly mask to the target CRS and resolution.
    """
    os.makedirs(output_dir, exist_ok=True)
    # The dictionary key is a tuple: (year, iso_week_number)
    # The value is another tuple: (numpy_array, rasterio_transform, rasterio_crs)
    weekly = defaultdict(lambda: (None, None, None))
    
    # This regex now correctly captures the first date as the start date.
    date_re = re.compile(r"(\d{8})_\d{8}")

    for root, _, files in os.walk(input_root):
        for fn in sorted(files): # Sorting ensures chronological processing
            if not fn.endswith('_SnowMask_latest.tif'):
                continue

            m = date_re.search(fn)
            if not m:
                print(f"⚠️  Skipping (no valid date pattern found): {fn}")
                continue
            
            # This is the START date of the 14-day period
            start_date = datetime.strptime(m.group(1), "%Y%m%d").date()

            sm_fp = os.path.join(root, fn)
            dm_fp = sm_fp.replace('_SnowMask_latest.tif', '_SnowMask_latest_datemap.tif')
            if not os.path.exists(dm_fp):
                print(f"⚠️ Missing datemap for {fn}")
                continue

            with rasterio.open(sm_fp) as sm_ds, rasterio.open(dm_fp) as dm_ds:
                sm = sm_ds.read(1).astype('uint8')
                dm = dm_ds.read(1).astype('uint8') # Datemap can be uint8
                transform = sm_ds.transform
                crs = sm_ds.crs
                nodata = sm_ds.nodata if sm_ds.nodata is not None else 255
                shape = sm.shape

            # --- CORRECT ISO WEEK CALCULATION ---
            # Week A is the ISO week of the start_date.
            # Week B is the ISO week of the start_date plus 7 days.
            iso_week_a = start_date.isocalendar()[:2]
            iso_week_b = (start_date + timedelta(days=7)).isocalendar()[:2]

            # Datemap values 1-7 belong to the first week (Week A)
            # Datemap values 8-14 belong to the second week (Week B)
            mappings = [
                ((dm >= 1) & (dm <= 7),  iso_week_a),
                ((dm >= 8) & (dm <= 14), iso_week_b),
            ]

            for mask_arr, iso in mappings:
                if not mask_arr.any():
                    continue
                
                # Retrieve the aggregated array for this ISO week, or None if it's the first time
                arr, tr, cr = weekly[iso]
                
                # If it's the first time we see this week, create a new blank array
                if arr is None:
                    arr = np.full(shape, nodata, dtype='uint8')
                    tr, cr = transform, crs

                # Apply the merging logic
                # No-snow (0) only fills existing nodata values
                arr[mask_arr & (sm == 0) & (arr == nodata)] = 0
                # Snow (1) always overwrites any existing value (nodata, 0)
                arr[mask_arr & (sm == 1)] = 1

                # Store the updated array back in the dictionary
                weekly[iso] = (arr, tr, cr)

    if not weekly:
        print("⚠️ No S2 weekly masks were generated.")
        return []

    # --- REPROJECTION TO COMMON GRID ---
    # Use the first processed raster to define the geographic bounds for the common grid
    _, (arr0, tr0, cr0) = next(iter(weekly.items()))
    h0, w0 = arr0.shape
    left, bottom, right, top = array_bounds(h0, w0, tr0)
    
    tgt_transform, tgt_width, tgt_height = calculate_default_transform(
        cr0, dst_crs, w0, h0, left, bottom, right, top, resolution=resolution
    )

    nodata_val = 255
    meta = {
        'driver': 'GTiff', 'dtype': 'uint8', 'count': 1,
        'crs': dst_crs, 'transform': tgt_transform,
        'width': tgt_width, 'height': tgt_height,
        'nodata': nodata_val, 'compress': 'deflate', 'tiled': True
    }

    out_files = []
    # Sort items by year, then week number for chronological output
    for (year, week), (arr_nat, tr_nat, cr_nat) in sorted(weekly.items()):
        dst_arr = np.full((tgt_height, tgt_width), nodata_val, dtype='uint8')
        reproject(
            source=arr_nat,
            destination=dst_arr,
            src_transform=tr_nat,
            src_crs=cr_nat,
            dst_transform=tgt_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=nodata_val,
            dst_nodata=nodata_val
        )
        
        # This is a good safety check
        bad = ~np.isin(dst_arr, [0, 1, nodata_val])
        if np.any(bad):
            dst_arr[bad] = nodata_val

        out_fp = os.path.join(output_dir, f"{year}_W{week:02d}.tif")
        with rasterio.open(out_fp, 'w', **meta) as dst:
            dst.write(dst_arr, 1)
        print(f"✅ Wrote {out_fp}")
        out_files.append(out_fp)

    return out_files


#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import os
import numpy as np
import rasterio
from datetime import datetime
from collections import defaultdict
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_s3_weekly(
    input_dir: str,
    output_dir: str,
    dst_crs: str = 'EPSG:32632',
    dst_res: tuple = (60, 60)
) -> list[str]:
    """
    Aggregates daily Sentinel-3 snow masks into weekly composites (binary),
    masks out any non-{0,1} codes (treating them as nodata), then
    reprojects each weekly raster to a common grid with explicit nodata=255.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Group daily files by ISO-week
    week_groups: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.tif'):
            continue
        low = fname.lower()
        if 'snowmask' not in low or '_datemap' in low:
            continue
        ds = ''.join(c for c in fname if c.isdigit())[:8]
        try:
            d0 = datetime.strptime(ds, '%Y%m%d').date()
        except ValueError:
            print(f"⚠️ Skipping (bad date): {fname}")
            continue
        y, w, _ = d0.isocalendar()
        week_groups[f"{y}_W{w:02d}"].append(os.path.join(input_dir, fname))

    if not week_groups:
        print("⚠️ No Sentinel-3 snowmask files found.")
        return []

    # 2) Build common target grid from first file
    first_fp = next(iter(week_groups.values()))[0]
    with rasterio.open(first_fp) as src0:
        src_crs        = src0.crs
        src_bounds     = src0.bounds
        width0, height0 = src0.width, src0.height
        src_transform0 = src0.transform

    tgt_transform, tgt_width, tgt_height = calculate_default_transform(
        src_crs, dst_crs,
        width0, height0,
        *src_bounds,
        resolution=dst_res
    )

    # 3) Output metadata template
    meta = {
        'driver'   : 'GTiff',
        'dtype'    : 'uint8',
        'count'    : 1,
        'crs'      : dst_crs,
        'transform': tgt_transform,
        'width'    : tgt_width,
        'height'   : tgt_height,
        'nodata'   : 255,
        'compress' : 'deflate',
        'tiled'    : True
    }

    out_files: list[str] = []

    # 4) For each ISO-week, composite + reproject
    for week_key, paths in sorted(week_groups.items()):
        # a) load each daily mask, mask >1 as NaN but keep 0 and 1
        stack = []
        for fp in paths:
            with rasterio.open(fp) as src:
                arr = src.read(1).astype('float32')
                # only values > 1 → nodata
                arr[arr > 1] = np.nan
                stack.append(arr)

        # b) temporal max (NaN-aware)
        weekly_float = np.nanmax(np.stack(stack, axis=0), axis=0)

        # c) cast to uint8: 0=no-snow, 1=snow, 255=nodata
        weekly_u = np.full(weekly_float.shape, 255, dtype='uint8')
        weekly_u[weekly_float == 0] = 0
        weekly_u[weekly_float == 1] = 1

        # d) prefill dest with nodata, then warp
        dst_arr = np.full((tgt_height, tgt_width), 255, dtype='uint8')
        reproject(
            source        = weekly_u,
            destination   = dst_arr,
            src_transform = src_transform0,
            src_crs       = src_crs,
            dst_transform = tgt_transform,
            dst_crs       = dst_crs,
            src_nodata    = 255,
            dst_nodata    = 255,
            resampling    = Resampling.nearest
        )

        # e) write out
        out_fp = os.path.join(output_dir, f"{week_key}.tif")
        with rasterio.open(out_fp, 'w', **meta) as dst:
            dst.write(dst_arr, 1)

        print(f"✅ Wrote weekly composite: {out_fp}")
        out_files.append(out_fp)

    return out_files


#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum

def match_raster_grid(reference_path: str,
                      target_path: str,
                      output_path: str,
                      resampling_method=ResamplingEnum.nearest) -> None:
    """
    Reprojects a raster (target) to match the exact grid of another (reference),
    preserving nodata by pre-filling the destination with nodata and
    passing src_nodata/dst_nodata to reproject.

    Parameters:
    - reference_path: raster to match (defines CRS, transform, shape, nodata)
    - target_path: input raster to reproject
    - output_path: where to save the reprojected raster
    - resampling_method: nearest, bilinear, etc.
    """
    # 1. Read reference metadata
    with rasterio.open(reference_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width
        dst_profile   = ref.profile.copy()
        dst_nodata    = ref.nodata if ref.nodata is not None else 255

    # 2. Read source data and determine src_nodata
    with rasterio.open(target_path) as src:
        src_data   = src.read(1)
        src_nodata = src.nodata if src.nodata is not None else dst_nodata
        src_transform = src.transform
        src_crs = src.crs

    # 3. Pre-fill destination array with nodata
    dst_array = np.full((dst_height, dst_width), dst_nodata, dtype=src_data.dtype)

    # 4. Reproject with explicit nodata handling
    reproject(
        source        = src_data,
        destination   = dst_array,
        src_transform = src_transform,
        src_crs       = src_crs,
        src_nodata    = src_nodata,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        dst_nodata    = dst_nodata,
        resampling    = resampling_method
    )

    # 5. Update profile and write
    dst_profile.update({
        'height': dst_height,
        'width': dst_width,
        'transform': dst_transform,
        'crs': dst_crs,
        'nodata': dst_nodata
    })
    with rasterio.open(output_path, 'w', **dst_profile) as dst:
        dst.write(dst_array, 1)

    print(f"✅ Reprojected to match grid: {output_path}")



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



# === 1. Visual Comparison Function with Side-by-Side, Overlay, and Difference Mode ===

import rasterio
import matplotlib.pyplot as plt
import numpy as np

def visual_compare_rasters(
    raster_path1,
    raster_path2,
    labels=("Product A", "Product B"),
    cmap='gray',
    overlay=False,
    difference=False,
    alpha=0.5,
    downsample=5,
    save_difference=False,
    difference_out_path=None
):
    """
    Compares two rasters visually using side-by-side, overlay, or difference view.
    Optionally saves the difference as a GeoTIFF file.

    Parameters:
    - raster_path1, raster_path2: Input raster file paths (must match in extent and resolution)
    - labels: Titles for the plots
    - overlay: Show image1 with image2 overlayed with transparency
    - difference: Show difference (image1 - image2)
    - downsample: Subsample factor for faster display
    - save_difference: If True, save difference raster as GeoTIFF
    - difference_out_path: Path to save difference raster if saving
    """

    with rasterio.open(raster_path1) as src1, rasterio.open(raster_path2) as src2:
        data1 = src1.read(1)[::downsample, ::downsample]
        data2 = src2.read(1)[::downsample, ::downsample]

        # after reading and downsampling...
        data1 = np.where(data1 == 255, np.nan, data1)
        data2 = np.where(data2 == 255, np.nan, data2)


        if data1.shape != data2.shape:
            print(f"⚠️ Shape mismatch: {data1.shape} vs {data2.shape}")
            return

        if difference:
            diff = data1.astype(float) - data2.astype(float)
            plt.figure(figsize=(8, 6))
            plt.imshow(diff, cmap='coolwarm', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
            plt.title(f"Difference: {labels[0]} - {labels[1]}")
            plt.colorbar(label='Difference')
            plt.axis('off')
            plt.show()

            if save_difference and difference_out_path:
                with rasterio.open(raster_path1) as ref:
                    profile = ref.profile.copy()
                    profile.update(dtype='float32')
                    with rasterio.open(difference_out_path, 'w', **profile) as dst:
                        dst.write(diff.astype('float32'), 1)
                print(f"💾 Difference saved to: {difference_out_path}")

        elif overlay:
            plt.figure(figsize=(8, 8))
            plt.imshow(data1, cmap=cmap)
            plt.imshow(data2, cmap='hot', alpha=alpha)
            plt.title(f"Overlay: {labels[0]} + {labels[1]}")
            plt.colorbar()
            plt.axis('off')
            plt.show()

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(data1, cmap=cmap)
            axes[0].set_title(labels[0])
            axes[0].axis('off')
            axes[1].imshow(data2, cmap=cmap)
            axes[1].set_title(labels[1])
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from rasterio.warp    import reproject
from rasterio.enums   import Resampling

def _bbox_intersection(bb1, bb2):
    """Return (minx, miny, maxx, maxy) of the intersection or None if empty."""
    minx = max(bb1.left,  bb2.left)
    miny = max(bb1.bottom, bb2.bottom)
    maxx = min(bb1.right, bb2.right)
    maxy = min(bb1.top,   bb2.top)
    if minx >= maxx or miny >= maxy:
        return None
    return (minx, miny, maxx, maxy)

def visual_compare_rasters_strict(
    raster_path1: str,
    raster_path2: str,
    labels       = ("Raster A", "Raster B"),
    difference   = False,
    downsample   = 5,
    cmap         = 'gray',
    resampling   = Resampling.nearest
):
    """
    Compare two binary snow‐mask rasters on their common footprint only,
    even if they have different extents/CRS.
    """
    # 1. Compute the common bounding box in raster1’s CRS
    with rasterio.open(raster_path1) as src1, rasterio.open(raster_path2) as src2:
        inter = _bbox_intersection(src1.bounds, src2.bounds)
        if inter is None:
            raise ValueError("Datasets do not overlap!")
        win1 = from_bounds(*inter, transform=src1.transform)
        win1 = win1.round_offsets().round_lengths()

        # 2. Read full‐resolution data from raster1’s intersection
        arr1_full = src1.read(1, window=win1)

        # 3. Allocate a matching full‐resolution array for raster2
        arr2_full = np.empty((win1.height, win1.width), dtype=src2.dtypes[0])

        # 4. Reproject raster2 into that exact window/grid
        reproject(
            source      = rasterio.band(src2, 1),
            destination = arr2_full,
            src_transform = src2.transform,
            src_crs       = src2.crs,
            dst_transform = src1.window_transform(win1),
            dst_crs       = src1.crs,
            dst_width     = win1.width,
            dst_height    = win1.height,
            resampling    = resampling
        )

    # 5. Downsample for display
    if downsample > 1:
        arr1 = arr1_full[::downsample, ::downsample]
        arr2 = arr2_full[::downsample, ::downsample]
    else:
        arr1, arr2 = arr1_full, arr2_full

    # 6. Mask out the common zero‐pixels (no‐snow background)
    mask = ~((arr1 == 0) & (arr2 == 0))
    a1 = np.where(mask, arr1, np.nan)
    a2 = np.where(mask, arr2, np.nan)

    # 7. Plot side‑by‑side or difference
    if difference:
        diff = a1.astype(float) - a2.astype(float)
        v = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1
        plt.figure(figsize=(6,5))
        plt.imshow(diff, cmap='coolwarm', vmin=-v, vmax=v)
        plt.title(f"Difference: {labels[0]} – {labels[1]}")
        plt.colorbar(label='Difference')
        plt.axis('off')
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.imshow(a1, cmap=cmap); ax1.set_title(labels[0]); ax1.axis('off')
        ax2.imshow(a2, cmap=cmap); ax2.set_title(labels[1]); ax2.axis('off')
        plt.tight_layout()
        plt.show()



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



import os
import rasterio
from rasterio.mask import mask
from typing import List, Dict

def clip_weekly_to_roi(
    weekly_dirs: Dict[str, str],
    shapes: List[dict],
    common_weeks: List[str],
    clipped_root: str
) -> None:
    """
    For each product in `weekly_dirs`, clips the weekly TIFFs (common_weeks)
    to the vector ROI (`shapes`), writing outputs under `clipped_root/<product>/<week>.tif`.
    """
    os.makedirs(clipped_root, exist_ok=True)

    for prod, src_folder in weekly_dirs.items():
        dst_folder = os.path.join(clipped_root, prod)
        os.makedirs(dst_folder, exist_ok=True)

        for wk in common_weeks:
            src_path = os.path.join(src_folder, f"{wk}.tif")
            dst_path = os.path.join(dst_folder,   f"{wk}.tif")

            if not os.path.exists(src_path):
                # no such week for this product
                continue
            if os.path.exists(dst_path):
                # already clipped
                continue

            with rasterio.open(src_path) as src:
                out_img, out_transform = mask(
                    dataset = src,
                    shapes  = shapes,
                    crop    = True,           # tight crop to ROI envelope
                    nodata  = src.nodata      # fill outside ROI with your nodata
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver"   : "GTiff",
                    "height"   : out_img.shape[1],
                    "width"    : out_img.shape[2],
                    "transform": out_transform,
                    "nodata"   : src.nodata
                })

            with rasterio.open(dst_path, "w", **out_meta) as dst:
                dst.write(out_img)

            print(f"🏷  Clipped {prod} {wk} → {dst_path}")



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



# === 2. Calculate Pixel-by-Pixel Agreement Percentage ===

import numpy as np
import rasterio

def calculate_agreement(raster_path1, raster_path2):
    """
    Calculates the % of matching pixels between two rasters,
    **excluding** any pixels equal to each file's nodata value.

    Returns:
      • agreement_pct
      • total_pixels  (non-nodata in both)
      • agree_pixels  (where data1==data2 and both valid)
    """
    with rasterio.open(raster_path1) as src1, rasterio.open(raster_path2) as src2:
        # read into float32 so we can use NaN
        d1 = src1.read(1).astype("float32")
        d2 = src2.read(1).astype("float32")
        nod1 = src1.nodata
        nod2 = src2.nodata

        # turn each nodata value into NaN
        if nod1 is not None:
            d1[d1 == nod1] = np.nan
        if nod2 is not None:
            d2[d2 == nod2] = np.nan

        if d1.shape != d2.shape:
            print("⚠️ Shape mismatch for agreement check.")
            return None

        # only pixels where neither is NaN
        valid = (~np.isnan(d1)) & (~np.isnan(d2))
        total = int(np.count_nonzero(valid))
        agree = int(np.count_nonzero((d1 == d2) & valid))

        pct = 100.0 * agree / total if total else 0.0
        return {
            "agreement_pct": pct,
            "total_pixels": total,
            "agree_pixels": agree
        }



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



# ── in snow_processing.py (or a new file e.g. stats.py) ────────────────────
import os
import glob
import numpy as np
import pandas as pd
import rasterio

def compute_weekly_statistics(clipped_root: str,
                              products: list[str],
                              pixel_size: float = 60.0) -> pd.DataFrame:
    """
    Walks through clipped weekly TIFFs for each product, computes:
      • total_pixels   (arr != nodata)
      • snow_pixels    (arr == 1)
      • missing_pixels (arr == nodata)
      • snow_area_km2  (snow_pixels * pixel_size^2 / 1e6)
      • coverage_pct   (100 * snow_pixels / total_pixels)

    Parameters
    ----------
    clipped_root : str
        Root folder containing one subfolder per product, each with weekly .tif files.
    products : list[str]
        List of product names matching subfolder names under clipped_root.
    pixel_size : float
        Side length of a pixel in meters (default 60.0 → 400 m²).

    Returns
    -------
    pd.DataFrame
        Columns: product, week, total_pixels, snow_pixels, missing_pixels,
                 snow_area_km2, coverage_pct
    """
    records = []
    pixel_area = pixel_size * pixel_size

    for prod in products:
        folder = os.path.join(clipped_root, prod)
        for path in glob.glob(os.path.join(folder, "*.tif")):
            week = os.path.splitext(os.path.basename(path))[0]
            with rasterio.open(path) as src:
                arr = src.read(1)
                nod = src.nodata

            valid_mask   = (arr != nod)
            snow_mask    = (arr == 1)
            missing_mask = (arr == nod)

            total_pixels   = int(np.count_nonzero(valid_mask))
            snow_pixels    = int(np.count_nonzero(snow_mask))
            missing_pixels = int(np.count_nonzero(missing_mask))

            snow_area_km2 = snow_pixels * pixel_area / 1e6
            coverage_pct  = (100.0 * snow_pixels / total_pixels
                             if total_pixels else np.nan)

            records.append({
                "product"        : prod,
                "week"           : week,
                "total_pixels"   : total_pixels,
                "snow_pixels"    : snow_pixels,
                "missing_pixels" : missing_pixels,
                "snow_area_km2"  : snow_area_km2,
                "coverage_pct"   : coverage_pct
            })

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["week", "product"]).reset_index(drop=True)



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import os
import itertools
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def compute_pairwise_agreement(
    aligned_root: str,
    products: list[str],
    common_weeks: list[str],
    pixel_size: float = 60.0,
    downsample: int = 10      # factor for plotting to reduce memory
) -> pd.DataFrame:
    """
    Compute per-week pairwise agreement stats and show difference maps between snow-mask products.

    For each product pair:
      - Returns a DataFrame of agreement metrics.
      - Plots a multi-panel figure of all weekly difference maps (downsampled) with agreement % in titles.
    Finally, produces a single bar chart of *overall* agreement % for each product pair.
    """
    records = []
    pixel_area_km2 = (pixel_size ** 2) / 1e6

    # Prepare storage
    diff_store = {pair: [] for pair in itertools.combinations(products, 2)}
    agree_store = {pair: [] for pair in itertools.combinations(products, 2)}

    # Gather weekly stats and difference arrays
    for week in sorted(common_weeks):
        for p1, p2 in itertools.combinations(products, 2):
            f1 = os.path.join(aligned_root, p1, f"{week}.tif")
            f2 = os.path.join(aligned_root, p2, f"{week}.tif")
            if not (os.path.isfile(f1) and os.path.isfile(f2)):
                continue
            with rasterio.open(f1) as src1, rasterio.open(f2) as src2:
                if (src1.width, src1.height) != (src2.width, src2.height):
                    continue
                a1 = src1.read(1)
                a2 = src2.read(1)
                nod1, nod2 = src1.nodata, src2.nodata

            valid = np.ones_like(a1, bool)
            if nod1 is not None:
                valid &= (a1 != nod1)
            if nod2 is not None:
                valid &= (a2 != nod2)
            total = int(valid.sum())
            if total == 0:
                continue

            agree = int(((a1 == a2) & valid).sum())
            pct   = 100.0 * agree / total
            only1 = int(((a1 == 1) & (a2 == 0) & valid).sum())
            only2 = int(((a2 == 1) & (a1 == 0) & valid).sum())

            records.append({
                'week': week,
                'prod1': p1,
                'prod2': p2,
                'agreement_pct': pct,
                'total_pixels': total,
                'agree_pixels': agree,
                'p1_only_pixels': only1,
                'p2_only_pixels': only2,
                'p1_only_area_km2': only1 * pixel_area_km2,
                'p2_only_area_km2': only2 * pixel_area_km2,
                'area_bias_km2': (only1 - only2) * pixel_area_km2,
            })

            # create int8 diff to save memory
            diff_arr = a1.astype(np.int8) - a2.astype(np.int8)
            diff_masked = np.ma.masked_where(~valid, diff_arr)
            if np.ma.count(diff_masked) > 0:
                diff_store[(p1, p2)].append((week, diff_masked))
                agree_store[(p1, p2)].append(pct)

    # If no data, return empty
    if not records:
        print("⚠️ No valid data found for any product pair.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Plot difference maps per pair
    for pair, diffs in diff_store.items():
        p1, p2 = pair
        if not diffs:
            print(f"No overlapping data for {p1} vs {p2}.")
            continue

        stats = agree_store[pair]
        n = len(diffs)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        cmap = ListedColormap(['blue', 'white', 'red'])
        cmap.set_bad(color='lightgrey')
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        for ax, ((wk, diff), pct) in zip(axes.flat, zip(diffs, stats)):
            disp = diff[::downsample, ::downsample]
            ax.imshow(disp, cmap=cmap, norm=norm)
            ax.set_title(f"{wk}\nAgree: {pct:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')

        # Turn off any unused axes
        for i in range(n, rows * cols):
            axes.flat[i].axis('off')

        fig.suptitle(f"Difference maps: {p1} vs {p2}", y=0.98)
        fig.tight_layout(rect=[0, 0.1, 1, 0.96])
        legend_patches = [
            Patch(color='red',   label=f'{p1} only'),
            Patch(color='white', label='agreement'),
            Patch(color='blue',  label=f'{p2} only'),
            Patch(color='lightgrey', label='no data')
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4, frameon=False)
        plt.show()

    # Plot overall agreement bar chart
    pairs = list(agree_store.keys())
    overall = [np.mean(agree_store[p]) for p in pairs]
    labels = [f"{p1} vs {p2}" for p1, p2 in pairs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, overall, color='gray')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Average Agreement (%)')
    ax.set_title('Overall Pairwise Agreement')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return df.sort_values(['prod1', 'prod2', 'week']).reset_index(drop=True)



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________


import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def compute_multisensor_snow_agreement(
    inputs: dict[str, str],  # product_name -> weekly_folder_path
    output_dir: str,
    common_weeks: list[str] | None = None
) -> None:
    """
    Build multisensor snow consensus maps across multiple weeks,
    then plot them all in a single mosaic figure with balanced side margins.

    Each output GeoTIFF per week has values:
      0 = all no-snow
      1 = exactly one says snow
      2 = exactly two say snow
      3 = exactly three say snow
      4 = all say snow
      255 = nodata (no product had data)

    Parameters
    ----------
    inputs : dict[str, str]
        Mapping from product name to the folder containing its weekly TIFFs.
    output_dir : str
        Directory where consensus TIFFs and the combined plot will be saved.
    common_weeks : list[str], optional
        If provided, the exact list of week IDs (e.g. ['2022_W01', '2022_W02',...]);
        otherwise, intersection of weeks present across all products is used.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Determine weeks to process
    if common_weeks is None:
        weeks_per_prod = []
        for week_dir in inputs.values():
            files = [os.path.splitext(f)[0]
                     for f in os.listdir(week_dir)
                     if f.lower().endswith('.tif')]
            weeks_per_prod.append(set(files))
        common = set.intersection(*weeks_per_prod) if weeks_per_prod else set()
        common_weeks = sorted(common)

    if not common_weeks:
        print("⚠️  No common weeks found. Exiting.")
        return

    # Prepare to collect arrays for plotting
    consensus_store = []  # list of (week_id, 2D array)
    meta = None
    nod = None

    # 2) Loop through each week and build consensus
    for week in common_weeks:
        arrays = []
        nodata_vals = []
        for prod, week_dir in inputs.items():
            fp = os.path.join(week_dir, f"{week}.tif")
            if not os.path.isfile(fp):
                break
            with rasterio.open(fp) as src:
                arr = src.read(1).astype('uint8')
                nodata_val = src.nodata if src.nodata is not None else 255
                arrays.append(arr)
                nodata_vals.append(nodata_val)
                if meta is None:
                    meta = src.meta.copy()
        else:
            # only if all products had this week
            stack = np.stack(arrays, axis=0)
            nod = nodata_vals[0]
            valid = (stack != nod)
            snow = (stack == 1)
            count = snow.sum(axis=0).astype('uint8')
            all_nodata = ~valid.any(axis=0)
            consensus = np.where(all_nodata, nod, count)

            # write GeoTIFF
            out_meta = meta.copy()
            out_meta.update({
                'dtype': 'uint8',
                'count': 1,
                'nodata': int(nod),
                'compress': 'deflate',
                'tiled': True
            })
            out_fp = os.path.join(output_dir, f"{week}.tif")
            with rasterio.open(out_fp, 'w', **out_meta) as dst:
                dst.write(consensus, 1)
            print(f"✅ Wrote consensus map: {out_fp}")

            consensus_store.append((week, consensus))

    # 3) Plot all weeks in a single mosaic with balanced margins and frames
    if not consensus_store:
        print("⚠️ No consensus arrays to plot.")
        return

    n = len(consensus_store)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False
    )

    # Discrete colormap: 0–4, with nodata masked as white
    cmap = ListedColormap(['#440154', '#30678D', '#35B779', '#FDE725', '#FAF925'])
    cmap.set_bad(color='white')
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    for ax, (week, arr) in zip(axes.flat, consensus_store):
        masked = np.ma.masked_equal(arr, nod)
        ax.imshow(masked, cmap=cmap, norm=norm)
        ax.set_title(week)
        ax.set_xticks([])
        ax.set_yticks([])
        # Draw black frame around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    # turn off any unused axes
    for idx in range(len(consensus_store), nrows * ncols):
        axes.flat[idx].axis('off')

    # Legend patches
    from matplotlib.patches import Patch
    patches = [
        Patch(color=cmap(i), label=str(i)) for i in range(5)
    ] + [Patch(color='white', label='nodata')]
    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=6,
        frameon=False
    )

    fig.suptitle('Multisensor Snow Consensus', fontsize=16, y=0.98)
    # Ensure equal left/right margins
    fig.subplots_adjust(left=0.2, right=0.9, wspace=0.1, hspace=0.1)
    plt.show()



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



import os
import re
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from snow_processing import calculate_agreement

def plot_agreement_series(s2_dir: str, s3_dir: str):
    """
    Scan two folders of weekly two snow‐mask tiffs (named 'YYYY_Www.tif'),
    compute S2-vs-S3 agreement for each common week, then plot agreement (%) 
    as a time series. Weeks for which one of the products is missing remain gaps.
    """
    # regex to pull out year & ISO‐week
    pattern = re.compile(r"(\d{4})_W(\d{2})\.tif$")
    
    records = []
    for fn in os.listdir(s2_dir):
        m = pattern.match(fn)
        if not m:
            continue
        year, week = int(m.group(1)), int(m.group(2))
        s2_fp = os.path.join(s2_dir, fn)
        s3_fp = os.path.join(s3_dir, fn)
        if not os.path.exists(s3_fp):
            # skip weeks where S3 is missing
            continue
        
        # compute pixel‐wise agreement
        stats = calculate_agreement(s2_fp, s3_fp)
        if stats is None:
            continue
        
        # turn ISO‐week into a real date (Monday)
        week_start = date.fromisocalendar(year, week, 1)
        records.append((week_start, stats["agreement_pct"]))
    
    # assemble into a time‐indexed Series
    df = pd.DataFrame(records, columns=["date", "agreement"]) \
           .set_index("date") \
           .sort_index()

    # reindex to a full weekly Monday sequence, leaving NaNs where missing
    full_idx = pd.date_range(start=df.index.min(),
                             end=df.index.max(),
                             freq="W-MON")
    df = df.reindex(full_idx)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["agreement"], marker="o", linestyle="-")
    ax.set_title("S2 ↔ S3 Weekly Agreement Over Time")
    ax.set_xlabel("Week starting")
    ax.set_ylabel("Agreement (%)")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
