## in this script, all 7 forecast runs for every day (for the past 60 days) are retrieved from the NBM 
## found on https://noaa-nbm-pds.s3.amazonaws.com/index.html and saved to parquet
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np 
import os
import boto3
from rasterstats import zonal_stats
import rioxarray
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta 
#import datetime

var = 'mint' #'maxt'

# loading NWS public zones shapefile
zones_gdf = gpd.read_file("https://www.weather.gov/source/gis/Shapefiles/WSOM/z_16ap26.zip")

# removing duplicates and non-conus zones, 3850 zones expected to remain
values_to_drop = ['AK', 'AS', 'FM', 'GU', 'HI', 'MH', 'MP', 'PR', 'PW', 'VI']
zones_gdf = zones_gdf[~zones_gdf['STATE'].isin(values_to_drop)]
zones_gdf['STATE_ZONE'] = zones_gdf['STATE'] + '_' + zones_gdf['ZONE']
zones_gdf = zones_gdf.dissolve(by='STATE_ZONE', as_index=False)
zones_gdf['global_zone_id'] = zones_gdf['STATE_ZONE'].astype('category').cat.codes + 1

os.environ["AWS_NO_SIGN_REQUEST"] = "YES" # to ensure u can anonymously use data

# pre-project shapefile to nbm projection to harmonize the two
# sample an example file from AWS to align the CRS globally before anything
sample_url = f"s3://noaa-nbm-pds/blendv4.0/conus/2020/10/20/0100/{var}/blendv4.0_conus_{var}_2020-10-20T01:00_2020-10-21T18:00.tif" #maxt:06:00 ending window time

print(f"we have variable named '{var}'")  
print('aligning shapefile CRS to NBM standard grid model')
with rioxarray.open_rasterio(sample_url, chunks=True) as sample_rst:
    zones_projected = zones_gdf.to_crs(sample_rst.rio.crs)

def get_nbm_version(target_date):
    # converting datetime.date object if it arrives as a string or pandas timestamp
    if isinstance(target_date, str):
        target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    elif hasattr(target_date, 'date'):
        target_date = target_date.date()
        
    # NOAA operational NBM windows
    if target_date > datetime.date(2020, 9, 30) and target_date < datetime.date(2023, 1, 18):
        return "blendv4.0"
    elif target_date < datetime.date(2024, 5, 17):
        return "blendv4.1"
    elif target_date < datetime.date(2025, 5, 7):
        return "blendv4.2"
    elif target_date < datetime.date(2026, 5, 5):
        return "blendv4.3"
    else:
        return "blendv5.0"

# processing 1 model run date across 7 forecast days
def process_nbm_multi_day(current_date):
    blend = get_nbm_version(current_date)
    
    date_str = current_date.strftime("%Y%m%d")
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")   
    
    # list to store all 7 days of records for this specific model run date
    model_run_records = []
    
    #  iterating through forecast days 1-7
    for f_day in range(1, 8):

        # for df record keeping, for maxT
        forecast_day_file = f_day - 1
        delta_file = datetime.timedelta(days=forecast_day_file)
        forecasted_date_file = current_date + delta_file
        #forecasted_date_str_df = forecasted_date_file.strftime("%Y%m%d")  #comment out for minT, uncomment for maxT

        # for url regex
        delta = datetime.timedelta(days=f_day)
        forecasted_date = current_date + delta
        forecasted_year = forecasted_date.strftime("%Y")
        forecasted_month = forecasted_date.strftime("%m")
        forecasted_day = forecasted_date.strftime("%d")
        forecasted_date_str = forecasted_date.strftime("%Y%m%d")
        forecasted_date_str_df = forecasted_date.strftime("%Y%m%d") #comment out for maxT, uncomment for minT
        
        s3_url = f"s3://noaa-nbm-pds/{blend}/conus/{year}/{month}/{day}/0100/{var}/{blend}_conus_{var}_{year}-{month}-{day}T01:00_{forecasted_year}-{forecasted_month}-{forecasted_day}T18:00.tif" # 18 for minT, 06 for maxT
        
        try:           
            with rioxarray.open_rasterio(s3_url, chunks=True) as rst:
                maxt_f = rst.sel(band=1)
                
                raster_array = maxt_f.values
                affine_transform = rst.rio.transform()

                stats = zonal_stats(
                    zones_projected, 
                    raster_array, 
                    affine=affine_transform, 
                    stats=["mean"], 
                    nodata=-99
                )

                # appending every datapoint for all active zones for this specific forecast day
                for idx, row in zones_projected.iterrows():
                    mean_val = stats[idx]['mean']

                    if mean_val is not None and not np.isnan(mean_val):
                        model_run_records.append({
                            'public_zone': row['global_zone_id'],
                            "unique_zone_str": row['STATE_ZONE'],
                            "model_run_date": pd.to_datetime(date_str), # day model was initialized
                            "date": pd.to_datetime(forecasted_date_str_df), # day the weather actually happens
                            "forecast_day": f_day,
                            f"{var}_value": mean_val,
                            'zone_id': row['ZONE'],
                            'state': row['STATE'],
                            'name': row['NAME']
                        })
                        
                print(f"initialization day is {date_str} (forecast day {f_day}) for {forecasted_date_str}")
                
        except Exception as e:
            print(f"Skipping NBM run {date_str} forecast day {f_day}: {str(e)}")
            continue

    return model_run_records

# main function
if __name__ == '__main__':
    # define past 60 days
    start_date = date.today() - timedelta(days=61)
    end_date = date.today()
    print(f'today is {end_date}')  # should be whatever date it is in zulu time
    
    date_list = []
    curr = start_date
    while curr <= end_date:
        date_list.append(curr)
        curr += datetime.timedelta(days=1)
        
    num_cores = 16  # change depending on the machine
    
    print(f"creating parallel multi-day grid across {num_cores} CPU cores for {len(date_list)} model runs, variable='{var}'")

    # Process dates concurrently 
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_nbm_multi_day, date_list))
        
    print("data parallel processing now complete, putting together df")
    
    all_flattened_records = []
    for r in results:
        if r is not None:
            all_flattened_records.extend(r)
            
    final_df = pd.DataFrame(all_flattened_records)
    
    # syncing some info
    final_df['week'] = final_df['date'].dt.isocalendar().week
    final_df['public_zone'] = final_df['unique_zone_str'].astype('category').cat.codes + 1
        
    # re-ordering columns to include new feature columns
    column_order = [
        'public_zone', 'date', 'week', f"{var}_value", 'forecast_day', 'unique_zone_str', 
        'model_run_date', 'zone_id', 'state', 'name'
    ]
    final_df = final_df[column_order]

final_df = final_df.sort_values(by=['public_zone', 'date', 'week', 'forecast_day']).reset_index(drop=True)

# saving to parquet
BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'
var = 'mint'
PARQUET_NAME = f'nbm_{var}_DEPLOYMENT.parquet'

final_df.to_parquet(PARQUET_NAME, index=False)
s3 = boto3.client('s3')
s3.upload_file(PARQUET_NAME, BUCKET_NAME, PARQUET_NAME)
print("PARQUET completed/uploaded")