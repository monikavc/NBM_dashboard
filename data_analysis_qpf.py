#!/usr/bin/env python
# coding: utf-8

# In[25]:


from io import StringIO
import boto3
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime


# In[26]:


nbm_s3_url = "s3://processed-data-809918852303-us-east-1-an/nbm_qpf24pmean_DEPLOYMENT.parquet"
nbm_df = pd.read_parquet(nbm_s3_url)


# In[27]:


nbm_df['date'] = pd.to_datetime(nbm_df['date'])
nbm_df = nbm_df.sort_values(by=['public_zone', 'date', 'week', 'forecast_day'])


# In[28]:


prism_s3_url = 's3://processed-data-809918852303-us-east-1-an/prism_qpf24pmean_DEPLOYMENT.parquet'
prism_df = pd.read_parquet(prism_s3_url)
prism_df = prism_df.sort_values(["unique_zone_str", "date"])


# In[29]:


# ensure data sorted chronologically
prism_df['date'] = pd.to_datetime(prism_df['date'])
prism_df = prism_df.sort_index()

# separate wet day column
prism_df["rain_only"] = prism_df["qpf24pmean_value"].where(prism_df["qpf24pmean_value"] > 0)


# In[30]:


def calc_rolling_stats(group, window_num):
    mean = group.rolling(window=window_num, min_periods=window_num).mean().shift(1)
    var = group.rolling(window=window_num, min_periods=window_num).var().shift(1)
    return pd.DataFrame({"mean": mean, "var": var}, index=group.index)

# rolling calculations, shifted by 1 day so as to not take current day into account
stats_df_60d = prism_df.groupby("public_zone", group_keys=False)['qpf24pmean_value'].apply(calc_rolling_stats, window_num=60)
stats_df_7d = prism_df.groupby("public_zone", group_keys=False)['qpf24pmean_value'].apply(calc_rolling_stats, window_num=7)


# In[31]:


# Method of Moments with a protection clip against 0 variance, to prevent zero deivionn errors in dry periods
stats_df_60d["var"] = stats_df_60d["var"].replace(0, np.nan)

prism_df["gamma_shape_60d"] = (stats_df_60d["mean"] ** 2) / stats_df_60d["var"]
prism_df["gamma_scale_60d"] = stats_df_60d["var"] / stats_df_60d["mean"]

# same thing for 3 day
stats_df_7d["var"] = stats_df_7d["var"].replace(0, np.nan)

prism_df["gamma_shape_7d"] = (stats_df_7d["mean"] ** 2) / stats_df_7d["var"]
prism_df["gamma_scale_7d"] = stats_df_7d["var"] / stats_df_7d["mean"]


# In[32]:


max_forecast_date = nbm_df["date"].max()
zones = prism_df["unique_zone_str"].unique()

full_idx = pd.MultiIndex.from_product(
    [zones, pd.date_range(prism_df["date"].min(), max_forecast_date)],
    names=["unique_zone_str", "date"],
)


# In[33]:


prism_df_extended = (
    prism_df.set_index(["unique_zone_str", "date"])
    .reindex(full_idx)
    .groupby("unique_zone_str")
    .ffill()  # forward-fill latest data, 6 days in advance
    .reset_index()
)


# In[34]:


prism_df_extended = prism_df_extended.replace([np.inf, -np.inf], np.nan).fillna(0)


# Merging

# In[35]:


# merge prism_df and nbm_df on date and zone string to ensure match

merged_df = pd.merge(
    nbm_df, 
    prism_df_extended, 
    on=['date', 'unique_zone_str'], 
    suffixes=('_nbm', '_prism'),
    how='left'
)


# In[36]:


merged_df = merged_df.drop(columns=['week_nbm', 'zone_id_nbm', 'state_nbm', 'name_nbm', 'zone_id_nbm', 'state_nbm', 'name_nbm', 
                                    'unique_zone_str', 'public_zone_nbm'])


# In[37]:


# Calculate the temp difference in F (NBM minus PRISM)
merged_df['nbm_minus_obs'] = merged_df['qpf24pmean_value_nbm'] - merged_df['qpf24pmean_value_prism']

merged_df = merged_df.rename(columns={'week_prism': 'week', 'state_prism': 'state', 'name_prism':'name', 'zone_id_prism':'zone_id', 'public_zone_prism':'public_zone'})

column_order = [
        'public_zone', 'date', 'week', 'forecast_day', 'nbm_minus_obs', 'state', 'zone_id', 'name', 
        'model_run_date', 'qpf24pmean_value_nbm', 'qpf24pmean_value_prism', 'rain_only', 'gamma_shape_60d', 'gamma_scale_60d',
        'gamma_shape_7d', 'gamma_scale_7d'
]
merged_df = merged_df[column_order]


# In[38]:


# calculate CDF. temporarily fill NaN shapes/scales with 1 just so the math function runs,
# then overwrite the invalid days afterward.
safe_shape_60d = merged_df["gamma_shape_60d"].fillna(1)
safe_scale_60d = merged_df["gamma_scale_60d"].fillna(1)

safe_shape_7d = merged_df["gamma_shape_7d"].fillna(1)
safe_scale_7d = merged_df["gamma_scale_7d"].fillna(1)


# In[39]:


merged_df["forecast_probability"] = stats.gamma.cdf(
    merged_df["qpf24pmean_value_nbm"], safe_shape_60d, scale=safe_scale_60d
)


# In[40]:


# clean up the edges and dry periods
# if there weren't enough rainy days to fit a Gamma distribution, the probability that a '0' forecast is normal is 100% (or 0 baseline)
merged_df.loc[merged_df["gamma_shape_60d"].isna(), "forecast_probability"] = 0.0

# If the forecast itself is 0, the probability is 0 (Gamma is strictly > 0)
merged_df.loc[merged_df["qpf24pmean_value_nbm"] == 0, "forecast_probability"] = 0.0


# In[41]:


# expected rain amount on a wet day
merged_df["past_60d_gamma_mean"] = merged_df["gamma_shape_60d"] * merged_df["gamma_scale_60d"]

# fill NaNs with 0 for completely dry periods
merged_df["past_60d_gamma_mean"] = merged_df["past_60d_gamma_mean"].fillna(0)


# expected rain amount on a wet day
merged_df["past_7d_gamma_mean"] = merged_df["gamma_shape_7d"] * merged_df["gamma_scale_7d"]

# fill NaNs with 0 for completely dry periods
merged_df["past_7d_gamma_mean"] = merged_df["past_7d_gamma_mean"].fillna(0)


# In[42]:


merged_df['nbm_minus_obs_window_60d'] = merged_df['qpf24pmean_value_nbm'] - merged_df['past_60d_gamma_mean']
merged_df['nbm_minus_obs_window_7d'] = merged_df['qpf24pmean_value_nbm'] - merged_df['past_7d_gamma_mean']


# In[44]:


# short-term volatility, checks if error spikes right now
merged_df['nbm_minus_obs_7d_std'] = merged_df['nbm_minus_obs_window_7d'].rolling(window=7).std()


# In[45]:


column_order = [
        'public_zone', 'date', 'week', 'forecast_day', 'nbm_minus_obs', 'nbm_minus_obs_window_60d', 'nbm_minus_obs_window_7d',
        'nbm_minus_obs_7d_std', 'past_60d_gamma_mean', 'past_7d_gamma_mean', 'state', 'zone_id', 'name', 'model_run_date', 'qpf24pmean_value_nbm', 
        'qpf24pmean_value_prism'
]

merged_df = merged_df[column_order]


# In[46]:


merged_df = merged_df.sort_values(by=['public_zone', 'model_run_date', 'week', 'forecast_day'])


# In[47]:


BUCKET_NAME = 'processed-data-809918852303-us-east-1-an'

merged_df.to_parquet('merged_qpf24pmean_DEPLOYMENT.parquet', index=False)
s3 = boto3.client('s3')
s3.upload_file('merged_qpf24pmean_DEPLOYMENT.parquet', BUCKET_NAME, 'merged_qpf24pmean_DEPLOYMENT.parquet')
print("PARQUET complete / uploaded now")


# done

# In[48]:


# import matplotlib.pyplot as plt
# import geopandas as gpd

# plot_day = '2026-07-12'
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
#     column='nbm_minus_obs',
#     ax=ax,
#     legend=True,
#     legend_kwds={'label': ' QPF diff in inches', 'orientation': 'horizontal','extend': 'both', 'pad': 0.05},
#     cmap='RdBu_r',
#     vmin=-max_error_bound,
#     vmax=max_error_bound
# )

# plt.title('NBM bias for ' + plot_day + ' forecast day 1', fontsize=13, fontweight='bold') # this is model - obs
# plt.tight_layout()
# plt.show()


# In[ ]:




