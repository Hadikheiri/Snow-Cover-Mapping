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



import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

def reproject_resample_visualize(
    src_path,
    dst_path,
    dst_crs='EPSG:32632',
    dst_res=(20, 20),
    resampling_method=Resampling.nearest,    # Use Resampling.nearest for binary products (MODIS, S2, S3)   /   Use Resampling.bilinear for continuous products (GFSC, EURAC)
    visualize=True,
    downsample_factor=10
):
    """
    Reprojects and resamples a raster to a target CRS and resolution,
    then optionally visualizes the result.

    Parameters:
    - src_path: str, path to input raster file
    - dst_path: str, path to save output raster
    - dst_crs: str, target CRS (default: 'EPSG:32632')
    - dst_res: tuple, target resolution in meters (default: 20x20)
    - resampling_method: rasterio.enums.Resampling, method to use
    - visualize: bool, whether to plot a preview of the result
    - downsample_factor: int, how much to reduce resolution for preview
    """

    # Load input raster and calculate transform for reprojection
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': src.dtypes[0],
            'count': src.count
        })

        # Perform reprojection and save output
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )

    print(f"✅ Resampled and reprojected saved to: {dst_path}")

    # Optional: Visualization (downsampled for speed/memory)
    if visualize:
        with rasterio.open(dst_path) as dst:
            data = dst.read(1)[::downsample_factor, ::downsample_factor]

        plt.imshow(data, cmap='gray')
        plt.title(f"Preview: Resampled Snow Mask\n({os.path.basename(dst_path)})")
        plt.colorbar()
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
    dst_res: tuple = (20, 20),
    threshold: float = 20.0,
    gf_suffix: str = '_GF.tif',
    qc_suffix: str = '_QC.tif'
) -> list[str]:
    """
    1) Find all *_GF.tif / *_QC.tif in input_dir.
    2) Read GF, QC; treat both GF.nodata and GF==255 as np.nan.
    3) Reproject/resample GF (bilinear) & QC (nearest) to (dst_crs, dst_res).
    4) Mask out (QC != 0) or np.isnan(GF) → nan.
    5) Binarize GF ≥ threshold → 1, else 0, then set nan→255.
    6) Write uint8 raster with nodata=255.
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

        with rasterio.open(gf_fp) as gf_src:
            gf = gf_src.read(1).astype('float32')
            nod = gf_src.nodata
            profile = gf_src.profile.copy()
            src_crs = gf_src.crs
            src_transform = gf_src.transform
            bounds = gf_src.bounds
            width = gf_src.width
            height = gf_src.height

        # treat both file nodata _and_ 255 as missing
        if nod is not None:
            gf[gf == nod] = np.nan
        gf[gf == 255] = np.nan

        with rasterio.open(qc_fp) as qc_src:
            qc = qc_src.read(1)

        # build target grid
        transform, w, h = calculate_default_transform(
            src_crs, dst_crs, width, height, *bounds, resolution=dst_res
        )

        gf_dst = np.empty((h, w), dtype='float32')
        qc_dst = np.empty((h, w), dtype=qc.dtype)

        # reproject
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

        # mask out clouds / missing
        invalid = (qc_dst != 0) | np.isnan(gf_dst)
        gf_dst[invalid] = np.nan

        # threshold → binary
        bin_dst = (gf_dst >= threshold).astype('uint8')
        # set all masked → 255
        bin_dst[np.isnan(gf_dst)] = 255

        # write out
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
        print(f"✅ Daily GFSC → {out_fp}")

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
import numpy as np
import rasterio
from datetime import datetime, timedelta
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_to_target(src_arr, src_transform, src_crs,
                        dst_transform, dst_crs, dst_shape, nodata_val):
    """Unchanged from your version."""
    h, w = dst_shape
    dst = np.full((h, w), nodata_val, dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=nodata_val,
        dst_nodata=nodata_val
    )
    return dst

def process_all_biweekly(input_root, output_dir):
    dst_crs = 'EPSG:32632'
    weekly_accum = {}
    target_transform = None
    target_shape = None

    mask_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.endswith('_SnowMask_latest.tif'):
                mask_files.append(os.path.join(root, f))

    print(f"Found {len(mask_files)} biweekly SnowMask files.")
    for idx, mask_fp in enumerate(sorted(mask_files), 1):
        dm_fp = mask_fp.replace('_SnowMask_latest.tif','_SnowMask_latest_datemap.tif')
        if not os.path.exists(dm_fp):
            print(f"[{idx}/{len(mask_files)}] Missing datemap, skipping.")
            continue

        print(f"[{idx}/{len(mask_files)}] {os.path.basename(mask_fp)}")
        with rasterio.open(mask_fp) as sm_ds, rasterio.open(dm_fp) as dm_ds:
            # ───────────────────────────────────────────────────────────────────────
            # 1) establish output grid once
            if target_transform is None:
                tform, w, h = calculate_default_transform(
                    sm_ds.crs, dst_crs,
                    sm_ds.width, sm_ds.height,
                    *sm_ds.bounds,
                    resolution=20
                )
                target_transform = tform
                target_shape = (h, w)

            # 2) force snow‐mask nodata to 255, keep 0 as valid
            nod_sm = 255
            nod_dm = dm_ds.nodatavals[0] if dm_ds.nodatavals[0] is not None else 0

            # 3) reproject both rasters
            sm_reproj = reproject_to_target(
                sm_ds.read(1), sm_ds.transform, sm_ds.crs,
                target_transform, dst_crs, target_shape, nod_sm
            )
            dm_reproj = reproject_to_target(
                dm_ds.read(1).astype(np.int32), dm_ds.transform, dm_ds.crs,
                target_transform, dst_crs, target_shape, nod_dm
            )

            # 4) valid = “observed” pixels per datemap only
            valid = (dm_reproj != nod_dm)

            # 5) parse the biweekly period from the filename
            base = os.path.basename(mask_fp).split('_SnowMask')[0]
            d0 = datetime.strptime(base.split('_')[0], "%Y%m%d").date()
            d1 = datetime.strptime(base.split('_')[1], "%Y%m%d").date()

            # 6) decide if datemap is day-index (1–14) or YYYYMMDD values
            vals = dm_reproj[valid]
            day_index = bool(vals.size and vals.max() <= 31)

            # 7) build two boolean masks + get their ISO‐week keys
            if day_index:
                offs   = dm_reproj - 1       # 0–13
                w1_map = valid & (offs <= 6)
                w2_map = valid & (offs >= 7)
                iso1   = d0.isocalendar()[:2]
                iso2   = (d0 + timedelta(days=7)).isocalendar()[:2]
            else:
                # map each unique YYYYMMDD → ISO and build year/week arrays
                uniq   = np.unique(vals)
                iso_map = {d: datetime.strptime(str(d), "%Y%m%d").isocalendar()[:2]
                        for d in uniq}
                years  = np.zeros(target_shape, np.int32)
                weeks  = np.zeros(target_shape, np.int32)
                for d,(y,wk) in iso_map.items():
                    m = (dm_reproj == d)
                    years[m] = y
                    weeks[m] = wk
                # now mask out the two periods
                iso1 = d0.isocalendar()[:2]
                iso2 = (d0 + timedelta(days=7)).isocalendar()[:2]
                w1_map = valid & (years == iso1[0]) & (weeks == iso1[1])
                w2_map = valid & (years == iso2[0]) & (weeks == iso2[1])

            # 8) ACCUMULATE into weekly_accum (this must come *after* you have w1_map/w2_map)
            for wk_map, iso in ((w1_map, iso1), (w2_map, iso2)):
                key = tuple(iso)
                if key not in weekly_accum:
                    weekly_accum[key] = np.full(target_shape, 255, np.uint8)
                arr = weekly_accum[key]
                # snow wins
                arr[wk_map & (sm_reproj == 1)] = 1
                # no‐snow only where still nodata
                arr[wk_map & (sm_reproj == 0) & (arr == 255)] = 0

    # 8) write outputs
    os.makedirs(output_dir, exist_ok=True)
    for (y, w), arr in sorted(weekly_accum.items()):
        out_fp = os.path.join(output_dir, f"{y}_W{w:02d}.tif")
        with rasterio.open(
            out_fp, 'w', driver='GTiff',
            height=arr.shape[0], width=arr.shape[1],
            count=1, dtype='uint8',
            crs=dst_crs, transform=target_transform,
            nodata=255, compress='deflate', tiled=True
        ) as dst:
            dst.write(arr, 1)
        print("→ Wrote", out_fp)

    print("Done.")



#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________

import os
import numpy as np
import rasterio
from datetime import datetime
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_s3_weekly(
    input_dir: str,
    output_dir: str,
    dst_crs: str = 'EPSG:32632',
    dst_res: tuple = (20, 20)
) -> list[str]:
    """
    Reprojects Sentinel-3 weekly snow masks into a common grid (EPSG:32632, 20m)
    and names them by ISO-week (YYYY_Www.tif).  No prefix is added.

    Parameters:
    - input_dir: folder containing files like 'YYYYMMDD_YYYYMMDD_SnowMask.tif'
    - output_dir: where to save reprojected weekly rasters
    - dst_crs: target CRS (default EPSG:32632)
    - dst_res: target resolution (default 20m)

    Returns:
    - List of output file paths (unique ISO-week filenames)
    """
    os.makedirs(output_dir, exist_ok=True)
    out_files = []

    # Loop through each Sentinel-3 weekly source
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.tif'):
            continue
        # skip any QC or unwanted files
        if 'snowmask' not in fname.lower():
            continue

        src_fp = os.path.join(input_dir, fname)
        # Extract the starting date (first 8 digits)
        digits = ''.join(c for c in fname if c.isdigit())
        try:
            date0 = datetime.strptime(digits[:8], '%Y%m%d')
        except ValueError:
            print(f"⚠️ Cannot parse date from '{fname}', skipping.")
            continue
        iso = date0.isocalendar()
        week_key = f"{iso.year}_W{iso.week:02d}"  # e.g. '2022_W07'
        out_fp = os.path.join(output_dir, f"{week_key}.tif")

        # Reproject + resample to target grid
        with rasterio.open(src_fp) as src:
            data = src.read(1)
            profile = src.profile.copy()
            src_crs = src.crs
            src_transform = src.transform
            src_width = src.width
            src_height = src.height
            src_bounds = src.bounds

        # compute transform and new dims using bounds unpack
        transform, w, h = calculate_default_transform(
            src_crs, dst_crs,
            src_width, src_height,
            *src_bounds,
            resolution=dst_res
        )
        meta = profile.copy()
        meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': w,
            'height': h,
            'dtype': 'uint8',
            'count': 1,
            'nodata': 255
        })

        dst_arr = np.empty((h, w), dtype='uint8')
        reproject(
            source=data,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        with rasterio.open(out_fp, 'w', **meta) as dst:
            dst.write(dst_arr, 1)
        print(f"✅ S3 week {week_key} → {out_fp}")
        out_files.append(out_fp)

    # Remove duplicates: if multiple source tiles produce same week_key, keep one
    unique = []
    seen = set()
    for fp in out_files:
        week = os.path.splitext(os.path.basename(fp))[0]
        if week not in seen:
            seen.add(week)
            unique.append(fp)
    return unique


#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________



import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum

def match_raster_grid(reference_path, target_path, output_path,
                      resampling_method=ResamplingEnum.nearest):
    """
    Reprojects a raster (target) to match the exact grid of another (reference).

    Parameters:
    - reference_path: str, path to raster whose shape/resolution/grid you want to match
    - target_path: str, input raster to reproject
    - output_path: str, path to save reprojected result
    - resampling_method: rasterio Resampling method (e.g., nearest, bilinear)
    """

    with rasterio.open(reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)
        dst_profile = ref.profile.copy()

    with rasterio.open(target_path) as src:
        src_data = src.read(1)  # assumes single band
        dst_array = np.empty(dst_shape, dtype=src_data.dtype)

        reproject(
            source=src_data,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_method
        )

    # Save to file
    dst_profile.update({
        'height': dst_shape[0],
        'width': dst_shape[1],
        'transform': dst_transform,
        'crs': dst_crs
    })

    with rasterio.open(output_path, 'w', **dst_profile) as dst:
        dst.write(dst_array, 1)

    print(f"✅ Reprojected to match: {output_path}")



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
                              pixel_size: float = 20.0) -> pd.DataFrame:
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
        Side length of a pixel in meters (default 20.0 → 400 m²).

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



from typing import List
import os
import itertools
import glob
import numpy as np
import pandas as pd
import rasterio
from snow_processing import calculate_agreement

def compute_pairwise_agreement(
    aligned_root: str,
    products: List[str],
    common_weeks: List[str],
    pixel_size: float = 20.0
) -> pd.DataFrame:
    """
    For each week in common_weeks and each pair of products, computes:
      • agreement_pct, total_pixels, agree_pixels
      • p1_only_pixels, p2_only_pixels
      • p1_only_area_km2, p2_only_area_km2
      • area_bias_km2 = (p2_only - p1_only) * pixel_area_km2

    aligned_root : root folder containing subfolders named after each product,
                   where each subfolder holds weekly .tif files named 'YYYY_Wxx.tif'
    products     : list of product names (must match subfolder names)
    common_weeks : list of week strings, e.g. ['2022_W07', '2022_W09', …]
    pixel_size   : side length of a pixel (meters), default 20 → pixel_area 400 m²
    """
    records = []
    pixel_area_km2 = (pixel_size * pixel_size) / 1e6

    for week in common_weeks:
        for p1, p2 in itertools.combinations(products, 2):
            f1 = os.path.join(aligned_root, p1, week + ".tif")
            f2 = os.path.join(aligned_root, p2, week + ".tif")
            if not (os.path.exists(f1) and os.path.exists(f2)):
                continue

            stats = calculate_agreement(f1, f2)
            if stats is None:
                # either a shape mismatch or zero‐pixel overlap
                continue

            # reopen the rasters to count exclusive snow pixels
            with rasterio.open(f1) as src1, rasterio.open(f2) as src2:
                a1 = src1.read(1)
                a2 = src2.read(1)

            # boolean masks for snow
            p1_snow = (a1 == 1)
            p2_snow = (a2 == 1)

            only1 = int(np.count_nonzero(p1_snow & ~p2_snow))
            only2 = int(np.count_nonzero(p2_snow & ~p1_snow))

            records.append({
                "week"             : week,
                "prod1"            : p1,
                "prod2"            : p2,
                "agreement_pct"    : stats["agreement_pct"],
                "total_pixels"     : stats["total_pixels"],
                "agree_pixels"     : stats["agree_pixels"],
                "p1_only_pixels"   : only1,
                "p2_only_pixels"   : only2,
                "p1_only_area_km2" : only1 * pixel_area_km2,
                "p2_only_area_km2" : only2 * pixel_area_km2,
                "area_bias_km2"    : (only2 - only1) * pixel_area_km2
            })

    df = pd.DataFrame.from_records(records)

    if df.empty:
        print("⚠️  No overlapping pairs found—check your paths, week list, or that files exist.")
        return df

    return df.sort_values(["week", "prod1", "prod2"]).reset_index(drop=True)
