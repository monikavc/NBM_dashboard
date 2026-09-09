#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import os
import boto3
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
import rasterio
from rasterstats import zonal_stats
import requests
import xarray as xr
from herbie import Herbie
import re
from shapely.geometry import Point


# In[2]:


# loading shapefile
print("Loading NWS Public Zones...")
zones_gdf = gpd.read_file(
    "https://www.weather.gov/source/gis/Shapefiles/WSOM/z_16ap26.zip"
)

zones_gdf["unique_zone_str"] = zones_gdf["STATE"] + "_" + zones_gdf["ZONE"]
values_to_drop = ["AK", "AS", "FM", "GU", "HI", "MH", "MP", "PR", "PW", "VI"]
zones_gdf = zones_gdf[~zones_gdf["STATE"].isin(values_to_drop)]

zones_gdf = zones_gdf.dissolve(by="unique_zone_str", as_index=False)
zones_gdf = zones_gdf.to_crs(epsg=4269)
print(f"Filtered to CONUS: {len(zones_gdf)} zones ready.")


# MINT

# In[3]:


START_DATE = date.today() - timedelta(days=61)
END_DATE = date.today()
VARIABLE = "tmin"
NUM_WORKERS = 16
RESOLUTION = "4km"

# fetch data, then zonal stats for each zone
def process_single_day(target_date: date):
    ###Fetches PRISM zip directly, extracts rasters, computes zonal stats, and returns a daily df ##
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://services.nacse.org/prism/data/get/us/{RESOLUTION}/{VARIABLE}/{date_str}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Skipping {date_str}: HTTP status {response.status_code}")
            return None

        # Unpack the response zip in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # PRISM web services deliver .tif files inside the zip
            raster_names = [
                n for n in z.namelist() if n.endswith((".bil", ".tif"))
            ]
            if not raster_names:
                print(f"No raster found in zip for {date_str}")
                return None

            raster_bytes = z.read(raster_names[0])

        # Open raster in memory without saving to disk
        with rasterio.io.MemoryFile(raster_bytes) as memfile:
            with memfile.open() as src:
                affine = src.transform
                array = src.read(1)
                nodata = src.nodata if src.nodata is not None else -9999

        # Run zonal stats on the in-memory array
        stats = zonal_stats(
            zones_gdf, array, affine=affine, stats="mean", nodata=nodata
        )

        df = pd.DataFrame(stats)

        raw_datetime = pd.to_datetime(date_str, format="%Y%m%d")
        df["date"] = raw_datetime # bc/ tmin is the morning low for the date ending convention
        df["zone_id"] = zones_gdf["ZONE"]
        df["state"] = zones_gdf["STATE"]
        df["name"] = zones_gdf["NAME"]
        df = df.rename(columns={"mean": "temp_c"})

        return df

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return None

if __name__ == "__main__":
    date_list = []
    VAR = 'mint'
    curr = START_DATE
    while curr <= END_DATE:
        date_list.append(curr)
        curr += timedelta(days=1)

    print(
        f"Processing {len(date_list)} days across {NUM_WORKERS} worker processes..."
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_single_day, date_list))

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError("No data was successfully retrieved.")

    final_df = pd.concat(valid_results, ignore_index=True)

    # Post-processing & Unit Conversion
    final_df["unique_zone_str"] = (
        final_df["state"] + "_" + final_df["zone_id"]
    )
    final_df["public_zone"] = (
        final_df["unique_zone_str"].astype("category").cat.codes + 1
    )

    final_df = final_df.rename(columns={"temp_c": f"{VAR}_value"})
    final_df[f"{VAR}_value"] = (final_df[f"{VAR}_value"] * (9 / 5)) + 32     # C to F conversion

    final_df = final_df.sort_values(by=["date", "public_zone"])
    final_df["week"] = final_df["date"].dt.isocalendar().week

    column_order = ["public_zone", "date", "week", f"{VAR}_value", "unique_zone_str", "zone_id", "state", "name"]
    final_df = final_df[column_order].reset_index(drop=True)


# In[4]:


today = datetime.utcnow().date()
date_str = today.strftime("%Y-%m-%d")

print(f'getting urma data for {today}')
hourly_datasets = []

for hour in range(12): # only up until 12Z
    dt_str = f"{date_str} {hour:02d}:00"

    # Uuse "urma" since it's likely post-processed already, as opposed to rtma
    H = Herbie(dt_str, model="rtma", product="anl", verbose=False)

    # retrieving 2m maxT
    ds = H.xarray("TMP:2 m")
    hourly_datasets.append(ds)

ds_day = xr.concat(hourly_datasets, dim="time")

# get maxT across the entire day, plus temp conversions
max_temp_k = ds_day["t2m"].min(dim="time")
max_temp_c = max_temp_k - 273.15
max_temp_f = (max_temp_c * 9/5) + 32

# Attach variables back to an xarray DataArray / Dataset
max_temp_f.name = "max_temperature_2m_F"
max_temp_f.attrs["units"] = "Fahrenheit"
max_temp_f.attrs["description"] = f"CONUS minT for {date_str}"

print("urma minT complete")


# In[5]:


# convert xarray max_temp_f grid to flatter DF
df_grid = max_temp_f.to_dataframe().reset_index()

# URMA longitudes are typically 0 to 360, so need to ajust to -180 to 180 for nbm
if (df_grid["longitude"] > 180).any():
    df_grid["longitude"] = df_grid["longitude"] - 360

# creating Point geometries for each grid cell coordinate
geometry = [
    Point(xy) for xy in zip(df_grid["longitude"], df_grid["latitude"])
]

# convert to GeoDF
grid_gdf = gpd.GeoDataFrame(
    df_grid[["max_temperature_2m_F"]], geometry=geometry, crs="EPSG:4326"
)

# Reprojecting grid to match the rest of the PRISM zones' coordinate systesm
grid_gdf = grid_gdf.to_crs(zones_gdf.crs)

# spatial join, and aggregate maxT by zone like for prism
joined = gpd.sjoin(grid_gdf, zones_gdf, how="inner", predicate="within")

# grouping by zone attributes, finding max
rtma_minT = (
    joined.groupby(["unique_zone_str", "NAME", "STATE"])[
        "max_temperature_2m_F"
    ]
    .max()
    .reset_index()
)

# renaming columns and adding necessary columns to soon concatenate PRISM and URMA/RTMA ###
rtma_minT = rtma_minT.rename(
    columns={"max_temperature_2m_F": "mint_value"}
)

rtma_minT['date'] = datetime.utcnow().date()
rtma_minT['date'] = pd.to_datetime(rtma_minT['date'])
rtma_minT['week'] = rtma_minT['date'].dt.isocalendar().week
rtma_minT["public_zone"] = (
        rtma_minT["unique_zone_str"].astype("category").cat.codes + 1
    )

rtma_minT['zone_id'] = rtma_minT['unique_zone_str'].str.extract(r'(\d+)').astype(int)
rtma_minT = rtma_minT.rename(columns={"STATE": 'state', 'NAME': 'name'})

column_order = ["public_zone", "date", "week", "mint_value", "unique_zone_str", "zone_id", "state", "name"]
rtma_minT = rtma_minT[column_order].reset_index(drop=True)


# In[6]:


result_mint = pd.concat([final_df, rtma_minT], axis=0, ignore_index=True)
result_mint['zone_id'] = result_mint['zone_id'].astype(str)
result_mint # should just have 3850 more rows than final_df, if not there's an issue


# In[7]:


# Save & Upload
var = 'mint'
BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'
result_mint.to_parquet(f'prism_{var}_DEPLOYMENT.parquet', index=False)
s3 = boto3.client('s3')
s3.upload_file(f'prism_{var}_DEPLOYMENT.parquet', BUCKET_NAME, f'prism_{var}_DEPLOYMENT.parquet')
print("PARQUET completed/uploaded for " + var)


# MAXT

# In[8]:


START_DATE = date.today() - timedelta(days=61)
END_DATE = date.today()
VARIABLE = "tmax"
NUM_WORKERS = 16
RESOLUTION = "4km"

# fetch data, then zonal stats for each zone
def process_single_day(target_date: date):
    ###Fetches PRISM zip directly, extracts rasters, computes zonal stats, and returns a daily df ##
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://services.nacse.org/prism/data/get/us/{RESOLUTION}/{VARIABLE}/{date_str}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Skipping {date_str}: HTTP status {response.status_code}")
            return None

        # unpack the response zip in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # PRISM web services deliver .bil or .tif files inside the zip
            raster_names = [
                n for n in z.namelist() if n.endswith((".bil", ".tif"))
            ]
            if not raster_names:
                print(f"No raster found in zip for {date_str}")
                return None

            raster_bytes = z.read(raster_names[0])

        # open raster in memory without saving to disk
        with rasterio.io.MemoryFile(raster_bytes) as memfile:
            with memfile.open() as src:
                affine = src.transform
                array = src.read(1)
                nodata = src.nodata if src.nodata is not None else -9999

        # run zonal stats on the in-memory array
        stats = zonal_stats(
            zones_gdf, array, affine=affine, stats="mean", nodata=nodata
        )

        df = pd.DataFrame(stats)

        # Shift date backward by 1 dayy to match PRISM 12Z-12Z
        raw_datetime = pd.to_datetime(date_str, format="%Y%m%d")
        df["date"] = raw_datetime - pd.DateOffset(days=1)
        df["zone_id"] = zones_gdf["ZONE"]
        df["state"] = zones_gdf["STATE"]
        df["name"] = zones_gdf["NAME"]
        df = df.rename(columns={"mean": "temp_c"})

        return df

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return None

if __name__ == "__main__":
    date_list = []
    VAR = 'maxt'
    curr = START_DATE
    while curr <= END_DATE:
        date_list.append(curr)
        curr += timedelta(days=1)

    print(
        f"Processing {len(date_list)} days across {NUM_WORKERS} worker processes..."
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_single_day, date_list))

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError("No data was successfully retrieved.")

    final_df = pd.concat(valid_results, ignore_index=True)

    # Post-processing & Unit Conversion
    final_df["unique_zone_str"] = (
        final_df["state"] + "_" + final_df["zone_id"]
    )
    final_df["public_zone"] = (
        final_df["unique_zone_str"].astype("category").cat.codes + 1
    )

    final_df = final_df.rename(columns={"temp_c": f"{VAR}_value"})
    final_df[f"{VAR}_value"] = (final_df[f"{VAR}_value"] * (9 / 5)) + 32      # conversion from C to F

    final_df = final_df.sort_values(by=["date", "public_zone"])
    final_df["week"] = final_df["date"].dt.isocalendar().week

    column_order = ["public_zone", "date", "week", f"{VAR}_value", "unique_zone_str", "zone_id", "state", "name"]
    final_df = final_df[column_order].reset_index(drop=True)


# In[9]:


yesterday = datetime.utcnow().date() - timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")

print(f'getting urma data for {yesterday}')
hourly_datasets = []

for hour in range(24):
    dt_str = f"{date_str} {hour:02d}:00"

    # Uuse "urma" since it's likely post-processed already, as opposed to rtma
    H = Herbie(dt_str, model="urma", product="anl", verbose=False)

    # retrieving 2m maxT
    ds = H.xarray("TMP:2 m")
    hourly_datasets.append(ds)

ds_day = xr.concat(hourly_datasets, dim="time")

# get maxT across the entire day, plus temp conversions
max_temp_k = ds_day["t2m"].max(dim="time")
max_temp_c = max_temp_k - 273.15
max_temp_f = (max_temp_c * 9/5) + 32

# Attach variables back to an xarray DataArray / Dataset
max_temp_f.name = "max_temperature_2m_F"
max_temp_f.attrs["units"] = "Fahrenheit"
max_temp_f.attrs["description"] = f"CONUS maxT for {date_str}"

print("urma maxT complete")


# In[10]:


# convert xarray max_temp_f grid to flatter DF
df_grid = max_temp_f.to_dataframe().reset_index()

# URMA longitudes are typically 0 to 360, so need to ajust to -180 to 180 for nbm
if (df_grid["longitude"] > 180).any():
    df_grid["longitude"] = df_grid["longitude"] - 360

# creating Point geometries for each grid cell coordinate
geometry = [
    Point(xy) for xy in zip(df_grid["longitude"], df_grid["latitude"])
]

# convert to GeoDF
grid_gdf = gpd.GeoDataFrame(
    df_grid[["max_temperature_2m_F"]], geometry=geometry, crs="EPSG:4326"
)

# Reprojecting grid to match the rest of the PRISM zones' coordinate systesm
grid_gdf = grid_gdf.to_crs(zones_gdf.crs)

# spatial join, and aggregate maxT by zone like for prism
joined = gpd.sjoin(grid_gdf, zones_gdf, how="inner", predicate="within")

# grouping by zone attributes, finding max
urma_maxT = (
    joined.groupby(["unique_zone_str", "NAME", "STATE"])[
        "max_temperature_2m_F"
    ]
    .max()
    .reset_index()
)

# renaming columns and adding necessary columns to soon concatenate PRISM and URMA/RTMA ###
urma_maxT = urma_maxT.rename(
    columns={"max_temperature_2m_F": "maxt_value"}
)

urma_maxT['date'] = datetime.utcnow().date() - timedelta(days=1)
urma_maxT['date'] = pd.to_datetime(urma_maxT['date'])
urma_maxT['week'] = urma_maxT['date'].dt.isocalendar().week
urma_maxT["public_zone"] = (
        urma_maxT["unique_zone_str"].astype("category").cat.codes + 1
    )

urma_maxT['zone_id'] = urma_maxT['unique_zone_str'].str.extract(r'(\d+)').astype(int)
urma_maxT = urma_maxT.rename(columns={"STATE": 'state', 'NAME': 'name'})

column_order = ["public_zone", "date", "week", "maxt_value", "unique_zone_str", "zone_id", "state", "name"]
urma_maxT = urma_maxT[column_order].reset_index(drop=True)


# In[11]:


result_maxt = pd.concat([final_df, urma_maxT], axis=0, ignore_index=True)
result_maxt['zone_id'] = result_maxt['zone_id'].astype(str)
result_maxt # should just have 3850 more rows than final_df, if not there's an issue


# In[12]:


# Save & Upload
var = 'maxt'
BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'
result_maxt.to_parquet(f'prism_{var}_DEPLOYMENT.parquet', index=False)
s3 = boto3.client('s3')
s3.upload_file(f'prism_{var}_DEPLOYMENT.parquet', BUCKET_NAME, f'prism_{var}_DEPLOYMENT.parquet')
print("PARQUET completed/uploaded for " + var)


# PCP

# In[13]:


START_DATE = date.today() - timedelta(days=61)
END_DATE = date.today()
VARIABLE = "ppt"

# fetch data, then zonal stats for each zone
def process_single_day(target_date: date):
    ###Fetches PRISM zip directly, extracts rasters, computes zonal stats, and returns a daily df ##
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://services.nacse.org/prism/data/get/us/{RESOLUTION}/{VARIABLE}/{date_str}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Skipping {date_str}: HTTP status {response.status_code}")
            return None

        # unpack the response zip in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # PRISM web services deliver .bil or .tif files inside the zip
            raster_names = [
                n for n in z.namelist() if n.endswith((".bil", ".tif"))
            ]
            if not raster_names:
                print(f"No raster found in zip for {date_str}")
                return None

            raster_bytes = z.read(raster_names[0])

        # open raster in memory, not saving to disk
        with rasterio.io.MemoryFile(raster_bytes) as memfile:
            with memfile.open() as src:
                affine = src.transform
                array = src.read(1)
                nodata = src.nodata if src.nodata is not None else -9999

        # run zonal stats on in-memory array
        stats = zonal_stats(
            zones_gdf, array, affine=affine, stats="mean", nodata=nodata
        )

        df = pd.DataFrame(stats)

        # shifting date backward by 1 dayt to match PRISM's 12Z-12Z
        raw_datetime = pd.to_datetime(date_str, format="%Y%m%d")
        df["date"] = raw_datetime - pd.DateOffset(days=1)
        df["zone_id"] = zones_gdf["ZONE"]
        df["state"] = zones_gdf["STATE"]
        df["name"] = zones_gdf["NAME"]
        df = df.rename(columns={"mean": "temp_c"})

        return df

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return None

# main
if __name__ == "__main__":
    date_list = []
    VAR = 'qpf24pmean'
    curr = START_DATE
    while curr <= END_DATE:
        date_list.append(curr)
        curr += timedelta(days=1)

    print(
        f"Processing {len(date_list)} days across {NUM_WORKERS} cpus"
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_single_day, date_list))

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError("No data was successfully retrieved.")

    final_df = pd.concat(valid_results, ignore_index=True)

    # post-processing part
    final_df["unique_zone_str"] = (
        final_df["state"] + "_" + final_df["zone_id"]
    )
    final_df["public_zone"] = (
        final_df["unique_zone_str"].astype("category").cat.codes + 1
    )

    final_df = final_df.rename(columns={"temp_c": f"{VAR}_value"})
    final_df[f"{VAR}_value"] = (final_df[f"{VAR}_value"] / 25.4)    # conversion from mm to inches to match NBM units

    final_df = final_df.sort_values(by=["date", "public_zone"])
    final_df["week"] = final_df["date"].dt.isocalendar().week

    column_order = ["public_zone", "date", "week", f"{VAR}_value", "unique_zone_str", "zone_id", "state", "name"]
    final_df = final_df[column_order].reset_index(drop=True)


# In[14]:


# define valid end date/time at 12Z today
yesterday = datetime.utcnow().date()
end_12z = datetime(yesterday.year, yesterday.month, yesterday.day, 12, 0)

# Generate hourly timestamps for the 24-hour window (13Z yesterday to 12Z today)
start_time = end_12z - timedelta(hours=23)
hourly_times = [start_time + timedelta(hours=i) for i in range(24)]

print(f"Fetching URMA 1-hr APCP from {hourly_times[0]} to {hourly_times[-1]}...")

pcp_datasets = []
for dt in hourly_times:
    dt_str = dt.strftime("%Y-%m-%d %H:00")

    # URMA precipitation product
    H = Herbie(dt_str, model="rtma", product="pcp", verbose=False)

    # load 1-hr accumulated ppt
    ds = H.xarray()
    pcp_datasets.append(ds)

# 2. Combine and sum all 24 hours
ds_pcp = xr.concat(pcp_datasets, dim="time")

# Sum total precipitation in millimeters and convert to inches (1 mm = 1/25.4 in)
total_qpf_mm = ds_pcp["tp"].sum(dim="time")
total_qpf_in = total_qpf_mm / 25.4

# Rename variable for clarity
qpf_24h = total_qpf_in.to_dataset(name="qpf_24h_in")


# In[15]:


df_grid = qpf_24h.to_dataframe().reset_index()

# fixing longitudinal formatting (0-360 to -180 to 180)
if (df_grid["longitude"] > 180).any():
    df_grid["longitude"] = df_grid["longitude"] - 360

geometry = [
    Point(xy) for xy in zip(df_grid["longitude"], df_grid["latitude"])
]

grid_gdf = gpd.GeoDataFrame(
    df_grid[["qpf_24h_in"]], geometry=geometry, crs="EPSG:4326"
)

# reproject to match PRISM"S zones coordinate system
grid_gdf = grid_gdf.to_crs(zones_gdf.crs)

joined = gpd.sjoin(grid_gdf, zones_gdf, how="inner", predicate="within")

# choosing to use mean for ppt zonal statistics across zones here. could potentially use max too, just in case
rtma_qpf = (
    joined.groupby(["unique_zone_str", "NAME", "STATE"])["qpf_24h_in"]
    .agg(
        qpf_max="max", qpf_mean="mean"
    )  # Change to .max() or .mean() as needed
    .reset_index()
)
# post processing dataframe to concat with prism ppt soon!!
rtma_qpf = rtma_qpf.rename(
    columns={
        "qpf_mean": "qpf24pmean_value",  #  switch to MAX if necessary
        "STATE": "state",
        "NAME": "name",
    }
)

rtma_qpf["date"] = datetime.utcnow().date() - timedelta(days=1)
rtma_qpf["date"] = pd.to_datetime(rtma_qpf["date"])
rtma_qpf["week"] = rtma_qpf["date"].dt.isocalendar().week
rtma_qpf["public_zone"] = (
    rtma_qpf["unique_zone_str"].astype("category").cat.codes + 1
)

rtma_qpf["zone_id"] = (
    rtma_qpf["unique_zone_str"].str.extract(r"(\d+)").astype(int)
)

column_order = ["public_zone", "date", "week", "qpf24pmean_value", "unique_zone_str", "zone_id", "state", "name"]
rtma_qpf = rtma_qpf[column_order].reset_index(drop=True)


# In[16]:


result_qpf = pd.concat([final_df, rtma_qpf], axis=0, ignore_index=True)
result_qpf['zone_id'] = result_qpf['zone_id'].astype(str)
result_qpf # should just have 3850 more rows than final_df, if not there's an issue


# In[17]:


# Save & Upload
var = 'qpf24pmean'
BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'
result_qpf.to_parquet(f'prism_{var}_DEPLOYMENT.parquet', index=False)
s3 = boto3.client('s3')
s3.upload_file(f'prism_{var}_DEPLOYMENT.parquet', BUCKET_NAME, f'prism_{var}_DEPLOYMENT.parquet')
print(f"PARQUET completed/uploaded for {var}")

