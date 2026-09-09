#!/usr/bin/env python
# coding: utf-8

# In[1]:


# in this script, arcGIS is accessed and feature layer is updated
import lightgbm as lgb
import pandas as pd
import numpy as np
import boto3
import geopandas as gpd
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection


# In[2]:


# retrieve LGBM text file.....
var='maxt'
model_maxt_thresh_1 = lgb.Booster(model_file=f'lgbm_production_model_{var}_2F_window_nozones.txt')
model_maxt_thresh_2 = lgb.Booster(model_file=f'lgbm_production_model_{var}_3F_window_nozones.txt')
model_maxt_thresh_3 = lgb.Booster(model_file=f'lgbm_production_model_{var}_4F_window.txt')
model_maxt_thresh_4 = lgb.Booster(model_file=f'lgbm_production_model_{var}_5F_window.txt')

var='mint'
model_mint_thresh_1 = lgb.Booster(model_file=f'lgbm_production_model_{var}_2F_window.txt')
model_mint_thresh_2 = lgb.Booster(model_file=f'lgbm_production_model_{var}_3F_window.txt')
model_mint_thresh_3 = lgb.Booster(model_file=f'lgbm_production_model_{var}_4F_window.txt')
model_mint_thresh_4 = lgb.Booster(model_file=f'lgbm_production_model_{var}_5F_window.txt')

var='qpf'
model_qpf_thresh_1 = lgb.Booster(model_file=f'lgbm_production_model_{var}_0.25_window.txt')
model_qpf_thresh_2 = lgb.Booster(model_file=f'lgbm_production_model_{var}_0.75_window.txt')


# In[3]:


import datetime # get current date, UTC time
date = datetime.date.today()


# In[4]:


s3 = boto3.client('s3')
df = pd.read_parquet(f's3://processed-data-809918852303-us-east-1-an/merged_maxt_DEPLOYMENT.parquet')

df['model_run_date'] = pd.to_datetime(df['model_run_date']).dt.date
df = df[df['model_run_date'] == date]

df = df.drop(columns=['state', 'zone_id', 'name'])
df = df.sort_values(by="date").reset_index(drop=True)

# extracting year for each row
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# extracting total weeks for each year
max_weeks = df.groupby('year')['week'].transform('max')
max_weeks = np.where(max_weeks.isin([52, 53]), max_weeks, 52)

# decomposing discrete week into continuous sin/cos component
df['week_sin'] = np.sin(2 * np.pi * df['week'] / max_weeks)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / max_weeks)

df = df.drop(columns=['year'])

X_features_maxt = df[['public_zone', 'week_sin', 'week_cos', 'forecast_day', 'nbm_minus_obs_window_7d', 'nbm_minus_obs_window_60d']] # matching features
X_features_maxt2 = df[['week_sin', 'week_cos', 'forecast_day', 'nbm_minus_obs_window_7d', 'nbm_minus_obs_window_60d']] # matching features


# In[6]:


s3 = boto3.client('s3')
df = pd.read_parquet(f's3://processed-data-809918852303-us-east-1-an/merged_mint_DEPLOYMENT.parquet')

df['model_run_date'] = pd.to_datetime(df['model_run_date']).dt.date
df = df[df['model_run_date'] == date]

df = df.drop(columns=['state', 'zone_id', 'name'])
df = df.sort_values(by="date").reset_index(drop=True)

# extracting year for each row
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# extracting total weeks for each year
max_weeks = df.groupby('year')['week'].transform('max')
max_weeks = np.where(max_weeks.isin([52, 53]), max_weeks, 52)

# decomposing discrete week into continuous sin/cos component
df['week_sin'] = np.sin(2 * np.pi * df['week'] / max_weeks)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / max_weeks)

df = df.drop(columns=['year'])

X_features_mint = df[['public_zone', 'week_sin', 'week_cos', 'forecast_day', 'nbm_minus_obs_window_7d', 'nbm_minus_obs_window_60d']] # matching features


# In[7]:


s3 = boto3.client('s3')
df = pd.read_parquet(f's3://processed-data-809918852303-us-east-1-an/merged_qpf24pmean_DEPLOYMENT.parquet')

df['model_run_date'] = pd.to_datetime(df['model_run_date']).dt.date
df = df[df['model_run_date'] == date]

df = df.drop(columns=['state', 'zone_id', 'name'])
df = df.sort_values(by="date").reset_index(drop=True)

# extracting year for each row
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# extracting total weeks for each year
max_weeks = df.groupby('year')['week'].transform('max')
max_weeks = np.where(max_weeks.isin([52, 53]), max_weeks, 52)

# decomposing discrete week into continuous sin/cos component
df['week_sin'] = np.sin(2 * np.pi * df['week'] / max_weeks)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / max_weeks)

df = df.drop(columns=['year'])

X_features_qpf = df[['public_zone', 'week_sin', 'week_cos', 'forecast_day', 'nbm_minus_obs_window_7d', 'nbm_minus_obs_window_60d']] # matching features


# In[9]:


# run predictions on all var for all thresh
preds_maxt1 = model_maxt_thresh_1.predict(X_features_maxt2)  
preds_maxt2 = model_maxt_thresh_2.predict(X_features_maxt2)  
preds_maxt3 = model_maxt_thresh_3.predict(X_features_maxt)  
preds_maxt4 = model_maxt_thresh_4.predict(X_features_maxt)  

preds_mint1 = model_mint_thresh_1.predict(X_features_mint)  
preds_mint2 = model_mint_thresh_2.predict(X_features_mint)  
preds_mint3 = model_mint_thresh_3.predict(X_features_mint)  
preds_mint4 = model_mint_thresh_4.predict(X_features_mint)  

preds_qpf1 = model_qpf_thresh_1.predict(X_features_qpf)
preds_qpf2 = model_qpf_thresh_2.predict(X_features_qpf)

argmax_maxt1 = np.argmax(preds_maxt1, axis=1)
argmax_maxt2 = np.argmax(preds_maxt2, axis=1)
argmax_maxt3 = np.argmax(preds_maxt3, axis=1)
argmax_maxt4 = np.argmax(preds_maxt4, axis=1)

argmax_mint1 = np.argmax(preds_maxt1, axis=1)
argmax_mint2 = np.argmax(preds_maxt2, axis=1)
argmax_mint3 = np.argmax(preds_maxt3, axis=1)
argmax_mint4 = np.argmax(preds_maxt4, axis=1)

argmax_qpf1 = np.argmax(preds_qpf1, axis=1)
argmax_qpf2 = np.argmax(preds_qpf2, axis=1)

# extract every class column using slice notation [:, column_index]
df_predictions = pd.DataFrame({
    'public_zone': df['public_zone'],
    'model_run_date': df['model_run_date'],
    'forecast_day': df['forecast_day'],
    'date': df['date'],

    'maxt1_predicted_class': argmax_maxt1,
    'maxt1_prob_class0': preds_maxt1[:, 0],  # reasonable forecast
    'maxt1_prob_class1': preds_maxt1[:, 1],  # likely overforecast
    'maxt1_prob_class2': preds_maxt1[:, 2],  # likely underforecast

    'maxt2_predicted_class': argmax_maxt2,
    'maxt2_prob_class0': preds_maxt2[:, 0],
    'maxt2_prob_class1': preds_maxt2[:, 1],
    'maxt2_prob_class2': preds_maxt2[:, 2],

    'maxt3_predicted_class': argmax_maxt3,
    'maxt3_prob_class0': preds_maxt3[:, 0],
    'maxt3_prob_class1': preds_maxt3[:, 1],
    'maxt3_prob_class2': preds_maxt3[:, 2],

    'maxt4_predicted_class': argmax_maxt4,
    'maxt4_prob_class0': preds_maxt4[:, 0],
    'maxt4_prob_class1': preds_maxt4[:, 1],
    'maxt4_prob_class2': preds_maxt4[:, 2],

    'mint1_predicted_class': argmax_mint1,
    'mint1_prob_class0': preds_mint1[:, 0],  # reasonable forecast
    'mint1_prob_class1': preds_mint1[:, 1],  # likely overforecast
    'mint1_prob_class2': preds_mint1[:, 2],  # likely underforecast etc etc..

    'mint2_predicted_class': argmax_mint2,
    'mint2_prob_class0': preds_mint2[:, 0],
    'mint2_prob_class1': preds_mint2[:, 1],
    'mint2_prob_class2': preds_mint2[:, 2],

    'mint3_predicted_class': argmax_mint3,
    'mint3_prob_class0': preds_mint3[:, 0],
    'mint3_prob_class1': preds_mint3[:, 1],
    'mint3_prob_class2': preds_mint3[:, 2],

    'mint4_predicted_class': argmax_mint4,
    'mint4_prob_class0': preds_mint4[:, 0],
    'mint4_prob_class1': preds_mint4[:, 1],
    'mint4_prob_class2': preds_mint4[:, 2],

    'qpf1_predicted_class': argmax_qpf1,
    'qpf1_prob_class0': preds_qpf1[:, 0],
    'qpf1_prob_class1': preds_qpf1[:, 1],
    'qpf1_prob_class2': preds_qpf1[:, 2],

    'qpf2_predicted_class': argmax_qpf2,
    'qpf2_prob_class0': preds_qpf2[:, 0],
    'qpf2_prob_class1': preds_qpf2[:, 1],
    'qpf2_prob_class2': preds_qpf2[:, 2],
})


# In[12]:


# get columns of highest probabilities for confidence mapping later on
df_predictions['maxt1_confidence'] = df_predictions[['maxt1_prob_class0', 'maxt1_prob_class1', 'maxt1_prob_class2']].max(axis=1)
df_predictions['maxt2_confidence'] = df_predictions[['maxt2_prob_class0', 'maxt2_prob_class1', 'maxt2_prob_class2']].max(axis=1)
df_predictions['maxt3_confidence'] = df_predictions[['maxt3_prob_class0', 'maxt3_prob_class1', 'maxt3_prob_class2']].max(axis=1)
df_predictions['maxt4_confidence'] = df_predictions[['maxt4_prob_class0', 'maxt4_prob_class1', 'maxt4_prob_class2']].max(axis=1)

df_predictions['mint1_confidence'] = df_predictions[['mint1_prob_class0', 'mint1_prob_class1', 'mint1_prob_class2']].max(axis=1)
df_predictions['mint2_confidence'] = df_predictions[['mint2_prob_class0', 'mint2_prob_class1', 'mint2_prob_class2']].max(axis=1)
df_predictions['mint3_confidence'] = df_predictions[['mint3_prob_class0', 'mint3_prob_class1', 'mint3_prob_class2']].max(axis=1)
df_predictions['mint4_confidence'] = df_predictions[['mint4_prob_class0', 'mint4_prob_class1', 'mint4_prob_class2']].max(axis=1)

df_predictions['qpf1_confidence'] = df_predictions[['qpf1_prob_class0', 'qpf1_prob_class1', 'qpf1_prob_class2']].max(axis=1)
df_predictions['qpf2_confidence'] = df_predictions[['qpf2_prob_class0', 'qpf2_prob_class1', 'qpf2_prob_class2']].max(axis=1)


# Getting training set from merged df

# In[14]:


# load raw spatial data to append to test set
gdf_map = gpd.read_file("https://www.weather.gov/source/gis/Shapefiles/WSOM/z_16ap26.zip")

values_to_drop = ['AK', 'AS', 'FM', 'GU', 'HI', 'MH', 'MP', 'PR', 'PW', 'VI']
gdf_map = gdf_map[~gdf_map['STATE'].isin(values_to_drop)]

gdf_map['unique_zone_str'] = gdf_map['STATE'] + '_' + gdf_map['ZONE']
gdf_map['public_zone'] = gdf_map['unique_zone_str'].astype('category').cat.codes + 1

gdf_map = gdf_map.dissolve(by='unique_zone_str', as_index=False)
gdf_map = gdf_map.to_crs(epsg=4269)


# In[15]:


gdf_map = gdf_map.drop(columns=[
 'CWA',
 'TIME_ZONE',
 'FE_AREA',
 'ZONE',
 'STATE_ZONE',
 'SHORTNAME'])


# In[16]:


gdf = gdf_map.merge(df_predictions, on='public_zone', how='inner')


# In[17]:


# append 12 hours ahead for arcgis purposes
gdf['date'] = gdf['date'] + pd.Timedelta(hours=12)
gdf['model_run_date'] = gdf['model_run_date'] + pd.Timedelta(hours=12)


# In[ ]:


gis = GIS(
    os.getenv("ARCGIS_URL"), os.getenv("ARCGIS_USERNAME"), os.getenv("ARCGIS_PASSWORD"),
)


# In[19]:


# get feature layer item using feature layer item ID
item_id = "ce78b0087377409ca8eb3b9caef40117"
feature_layer_item = gis.content.get(item_id)

# save updated df to same geojson (MUST BE SAME NAME !!)
gjson_path = f"daily_update.geojson"
gdf.to_file(gjson_path, driver="GeoJSON")

# access the feature layer  collection manager, this is where we overwrite
flc = FeatureLayerCollection.fromitem(feature_layer_item)
flc.manager.overwrite(gjson_path)

print("feature layer overwritten ,,")

