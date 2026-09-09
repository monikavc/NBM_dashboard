#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### in this script, calculate rolling averages/merging model and observations ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import boto3
import datetime


# In[2]:


nbm_s3_url = "s3://processed-data-809918852303-us-east-1-an/nbm_maxt_DEPLOYMENT.parquet"
nbm_df = pd.read_parquet(nbm_s3_url)


# In[3]:


nbm_df['date'] = pd.to_datetime(nbm_df['date'])
nbm_df = nbm_df.sort_values(by=['public_zone', 'date', 'week', 'forecast_day'])


# In[4]:


prism_s3_url = 's3://processed-data-809918852303-us-east-1-an/prism_maxt_DEPLOYMENT.parquet'
prism_df = pd.read_parquet(prism_s3_url)
prism_df = prism_df.sort_values(["unique_zone_str", "date"])


# In[5]:


# shift by 1 day to exclude the current day!!
prism_df["60daywindow_prism"] = prism_df.groupby("unique_zone_str")["maxt_value"].transform(lambda x: x.shift(1).rolling(window=60, min_periods=1).mean())

prism_df["7daywindow_prism"] = prism_df.groupby("unique_zone_str")["maxt_value"].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())


# In[6]:


# forward fill future dates to create full date spine for observations per zone up to the max forecast date in df_model
# so NaNs don't propagate
max_forecast_date = nbm_df["date"].max()
zones = prism_df["unique_zone_str"].unique()

full_idx = pd.MultiIndex.from_product(
    [zones, pd.date_range(prism_df["date"].min(), max_forecast_date)],
    names=["unique_zone_str", "date"],
)


# In[7]:


prism_df_extended = (
    prism_df.set_index(["unique_zone_str", "date"])
    .reindex(full_idx)
    .groupby("unique_zone_str")
    .ffill()  # Forward-fills latest valid rolling window metrics into future dates, 6 days in advance
    .reset_index()
)


# In[9]:


merged_df = pd.merge(
    nbm_df,
    prism_df_extended,
    on=["date", "unique_zone_str"],
    suffixes=("_nbm", "_prism"),
    how="left",
)


# In[11]:


merged_df = merged_df.drop(columns=['week_nbm', 'zone_id_nbm', 'state_nbm', 'name_nbm', 'zone_id_nbm', 'state_nbm', 'name_nbm', 'unique_zone_str', 'public_zone_nbm'])
merged_df = merged_df.rename(columns={'week_prism': 'week', 'state_prism': 'state', 'name_prism':'name', 'zone_id_prism':'zone_id', 'public_zone_prism':'public_zone'})


# In[12]:


merged_df['nbm_minus_obs'] = merged_df['maxt_value_nbm'] - merged_df['maxt_value_prism']


# In[13]:


column_order = [
        'public_zone', 'date', 'week', 'forecast_day', 'state', 'zone_id', 'name', 
        'model_run_date', 'maxt_value_nbm', 'maxt_value_prism', '60daywindow_prism', '7daywindow_prism','nbm_minus_obs'
]
merged_df = merged_df[column_order]


# In[15]:


merged_df['nbm_minus_obs_window_60d'] = merged_df['maxt_value_nbm'] - merged_df['60daywindow_prism']
merged_df['nbm_minus_obs_window_7d'] = merged_df['maxt_value_nbm'] - merged_df['7daywindow_prism']


# In[17]:


# print(merged_df['nbm_minus_obs'].describe().round(2))
# print(merged_df['nbm_minus_obs_window_60d'].describe().round(2))
# print(merged_df['nbm_minus_obs_window_7d'].describe().round(2))


# In[18]:


BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'

merged_df.to_parquet('merged_maxt_DEPLOYMENT.parquet', index=False)
s3 = boto3.client('s3')
s3.upload_file('merged_maxt_DEPLOYMENT.parquet', BUCKET_NAME, 'merged_maxt_DEPLOYMENT.parquet')
print("Parquet complete /uploaded  now")


# In[24]:


# # attempting to plot NBM biases

# import geopandas as gpd

# plot_day = '2026-09-06'
# day_diff_df = merged_df[merged_df['date'] == plot_day]

# zones_gdf = gpd.read_file("https://www.weather.gov/source/gis/Shapefiles/WSOM/z_16ap26.zip")
# zones_gdf['unique_zone_str'] = zones_gdf['STATE'] + '_' + zones_gdf['ZONE']
# zones_gdf['STATE_ZONE'] = zones_gdf['STATE'] + '_' + zones_gdf['ZONE']

# zones_display = zones_gdf.to_crs(epsg=4269)
# map_gdf = zones_display.merge(day_diff_df, left_on='STATE_ZONE', right_on='unique_zone_str')

# #initialize the plot figure
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# zones_display.plot(ax=ax, color='lightgrey', edgecolor='none')
# plt.grid(linestyle='--', color='grey', alpha=0.2)

# plt.xlim(-126, -65)
# plt.ylim(24, 50)

# max_error_bound = 10

# map_gdf.plot(
#     column='temp_diff_f',
#     ax=ax,
#     legend=True,
#     legend_kwds={'label': 'temperature diff in °F', 'orientation': 'horizontal', 'extend': 'both','pad': 0.05},
#     cmap='RdBu_r',
#     vmin=-max_error_bound,
#     vmax=max_error_bound
# )

# plt.title('NBM bias for ' + plot_day + ' forecast day 1', fontsize=13, fontweight='bold') # this is model - obs
# plt.tight_layout()
# plt.show()

