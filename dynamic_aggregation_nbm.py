#!/usr/bin/env python
# coding: utf-8

# In[1]:


# retrieving live nbm data
import os
import pandas as pd
import datetime
import geopandas as gpd
import boto3
import numpy as np 
from rasterstats import zonal_stats
import rioxarray
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# In[2]:


var = 'maxt'   # depending on 'maxt', 'mint', 'qpf24pmean', 'snowamt24mean'

# dictionaries for each variable and their associated model run initialization times & window ending times
VARS = ['mint', 'maxt', 'qpf24pmean']
VARS_URL_AWS_NBM = {'mint': '0100', 'maxt':'0100', 'qpf24pmean': '0600', 'snowamt24mean':'0100'}
VARS_WINDOW_END_TIME = {'mint': '18', 'maxt':'06', 'qpf24pmean': '12', 'snowamt24mean':'12'}
VARS_INTIALIZATION_TIME = {'mint': '01', 'maxt':'01', 'qpf24pmean': '06', 'snowamt24mean':'01'}

# load NWS public zones shapefile like in previous noteboks
zones_gdf = gpd.read_file("https://www.weather.gov/source/gis/Shapefiles/WSOM/z_16ap26.zip")

# removing duplicates and non-conus zones, 3850 zones expected to remain
values_to_drop = ['AK', 'AS', 'FM', 'GU', 'HI', 'MH', 'MP', 'PR', 'PW', 'VI']
zones_gdf = zones_gdf[~zones_gdf['STATE'].isin(values_to_drop)]
zones_gdf['STATE_ZONE'] = zones_gdf['STATE'] + '_' + zones_gdf['ZONE']
zones_gdf = zones_gdf.dissolve(by='STATE_ZONE', as_index=False)
zones_gdf['global_zone_id'] = zones_gdf['STATE_ZONE'].astype('category').cat.codes + 1

os.environ["AWS_NO_SIGN_REQUEST"] = "YES" # to ensure we can anonymously use data

# need to sample an example file from AWS to align the CRS globally
sample_url = f"s3://noaa-nbm-pds/blendv4.0/conus/2020/10/20/{VARS_URL_AWS_NBM[var]}/{var}/blendv4.0_conus_{var}_2020-10-20T{VARS_INTIALIZATION_TIME[var]}:00_2020-10-21T{VARS_WINDOW_END_TIME[var]}:00.tif"

print(f"We have variable named '{var}'")
print("Aligning shapefile CRS to NBM standard grid model")
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


# In[3]:


# processing 1 model run date across 7 forecast days
def process_nbm_multi_day(current_date, var, initialization_time, initialization_time_url, window_ending_time_url):

    blend = get_nbm_version(current_date)

    date_str = current_date.strftime("%Y%m%d")
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")   

    # list to store all 7 days of records for this specific model run date
    model_run_records = []

    #  iterating through forecast days 1-7
    for f_day in range(1, 8):

        # for df record keeping
        forecast_day_file = f_day - 1
        delta_file = datetime.timedelta(days=forecast_day_file)
        forecasted_date_file = current_date + delta_file
        forecasted_date_str_df = forecasted_date_file.strftime("%Y%m%d")

        # for url regex
        delta = datetime.timedelta(days=f_day)
        forecasted_date = current_date + delta
        forecasted_date_str = forecasted_date.strftime("%Y%m%d")
        forecasted_year = forecasted_date.strftime("%Y")
        forecasted_month = forecasted_date.strftime("%m")
        forecasted_day = forecasted_date.strftime("%d")        
        if (var=='mint'):
            forecasted_date_str_df = forecasted_date.strftime("%Y%m%d")

        s3_url = f"s3://noaa-nbm-pds/{blend}/conus/{year}/{month}/{day}/{initialization_time}/{var}/{blend}_conus_{var}_{year}-{month}-{day}T{initialization_time_url}:00_{forecasted_year}-{forecasted_month}-{forecasted_day}T{window_ending_time_url}:00.tif"                                                                                                               

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

                print(f"initialization day is {date_str} (forecast day {f_day}) for {forecasted_date_str_df}, var: {var}")

        except Exception as e:
            print(f"Skipping NBM run {date_str} forecast day {f_day}: {str(e)}")
            continue

    return model_run_records


# In[ ]:


# main
if __name__ == '__main__':
    var='maxt'
    # because we are running for tomorrow's forecast
    if (var=='qpf24pmean'):
        start_date = datetime.date.today() #- datetime.timedelta(days=1)   #bc/ qpf init time is 06z. not 01z so it's later 
        end_date = datetime.date.today() #- datetime.timedelta(days=1)
    else:
        start_date = datetime.date.today()
        end_date = datetime.date.today()

    num_cores = 16
    date_list = []
    curr = start_date

    while curr <= end_date:
        date_list.append(curr)
        curr += datetime.timedelta(days=1)

    func = partial(
        process_nbm_multi_day,
        var=var,
        initialization_time=VARS_URL_AWS_NBM[var], # for every time and day
        initialization_time_url=VARS_INTIALIZATION_TIME[var],
        window_ending_time_url=VARS_WINDOW_END_TIME[var],
    )

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(func, date_list))

    print("data parallel processing now complete, putting together df")

    all_flattened_records = []
    for r in results:
        if r is not None:
            all_flattened_records.extend(r)

    newest_data = pd.DataFrame(all_flattened_records)

    # syncing some info
    newest_data['week'] = newest_data['date'].dt.isocalendar().week
    newest_data['public_zone'] = newest_data['unique_zone_str'].astype('category').cat.codes + 1

    # re-ordering columns
    column_order = [
        'public_zone', 'date', 'week', f"{var}_value", 'forecast_day', 'unique_zone_str', 
        'model_run_date', 'zone_id', 'state', 'name'
    ]
    newest_data = newest_data[column_order]

newest_data = newest_data.sort_values(by=['public_zone', 'date', 'week', 'forecast_day']).reset_index(drop=True)


# In[8]:


def process_daily_update(PARQUET_PATH, newest_data):
    CUTOFF_DATE = datetime.date.today() - datetime.timedelta(days=61)
    CUTOFF_DATE = np.datetime64(CUTOFF_DATE)             # convert to datetime64 object

    if os.path.exists(PARQUET_PATH):
        #df_history = pd.read_parquet(PARQUET_PATH)
        s3 = boto3.client('s3')
        df_history = pd.read_parquet(f's3://processed-data-809918852303-us-east-1-an/{PARQUET_PATH}')
        # combine existing history with today's new data
        df = pd.concat([df_history, newest_data], ignore_index=True)
    else:
        df = newest_data

    # ensure sorting by date & drop duplicate entries if script runs twice on the same day for same exact data
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "public_zone", 'week', 'forecast_day', 'model_run_date'], keep="last")

    # take only the last 60 daily model runs across all zones and forecast days, creating this sliding window of kept data
    df = df[df["model_run_date"] >= CUTOFF_DATE]
    df = df.sort_values(by=['public_zone', 'date', 'week', 'forecast_day']).reset_index(drop=True)
    print(df)

    # overwrite parquet with updated 60 day window
    BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'
    df.to_parquet(PARQUET_PATH, index=False)
    s3 = boto3.client('s3')
    s3.upload_file(PARQUET_PATH, BUCKET_NAME, PARQUET_PATH)
    print("parquet completed/uploaded")


# In[9]:


if __name__ == "__main__":

    start_date = datetime.date.today()
    end_date = datetime.date.today()
    PARQUETS = ['nbm_mint_DEPLOYMENT.parquet', 'nbm_maxt_DEPLOYMENT.parquet', 'nbm_qpf24pmean_DEPLOYMENT.parquet']
    num_cores = 16
    date_list = []
    curr = start_date
    i = 0

    for var in VARS:
        # because we are running for tomorrow's forecast
        if (var=='qpf24pmean'):
            start_date = datetime.date.today() #- datetime.timedelta(days=1)   #bc/ qpf init time is 06z. not 01z so it's later 
            end_date = datetime.date.today() #- datetime.timedelta(days=1)
        else:
            start_date = datetime.date.today()
            end_date = datetime.date.today()

        while curr <= end_date:
            date_list.append(curr)
            curr += datetime.timedelta(days=1)

        func = partial(
            process_nbm_multi_day,
            var=var,
            initialization_time=VARS_URL_AWS_NBM[var], # for every time and day
            initialization_time_url=VARS_INTIALIZATION_TIME[var],
            window_ending_time_url=VARS_WINDOW_END_TIME[var],
        )

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(func, date_list))

        print("data parallel processing now complete, putting together df")

        all_flattened_records = []
        for r in results:
            if r is not None:
                all_flattened_records.extend(r)

        newest_data = pd.DataFrame(all_flattened_records)

        # syncing some info
        newest_data['week'] = newest_data['date'].dt.isocalendar().week
        newest_data['public_zone'] = newest_data['unique_zone_str'].astype('category').cat.codes + 1

        # re-ordering columns
        column_order = [
            'public_zone', 'date', 'week', f"{var}_value", 'forecast_day', 'unique_zone_str', 
            'model_run_date', 'zone_id', 'state', 'name'
        ]
        newest_data = newest_data[column_order]

        newest_data = newest_data.sort_values(by=['public_zone', 'date', 'week', 'forecast_day']).reset_index(drop=True)

        process_daily_update(PARQUETS[i], newest_data)
        i += 1
    print('appended freshest model runs')

