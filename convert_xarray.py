import xarray as xr
import matplotlib.pyplot as plt
#import tensorflow as tf

path = "INSERT"
# Paths to our data
temperature_path = path+"2m_temperature/resolution_100/era5_2m_temperature_daily_*.nc"
snow_path = path+"snow_depth/resolution_100/era5_snow_depth_daily_*.nc"
precipitation_path = path+"total_precipitation/resolution_100/era5_total_precipitation_daily_*.nc" 
# Load data
# data_temp = xr.open_mfdataset(temperature_path, combine = 'by_coords')

def xarray_load(paths,lon,lat):
    '''
    paths: list of paths to datasets with * notation (e.g. ["2m_temperature/era5_2m_temperature_daily_*.nc"])
    lon: longitude, tuple (0,360) -> for europe (331,50)
    lat: latitude, tuple (-90,90) -> for europe (21,80)
    '''
    sets = []
    for i in paths:
        data = xr.open_mfdataset(i, combine = 'by_coords')
        data["time"] = data.time.dt.floor("1d")
        data = data.drop("time_bnds")
        data = data.sel(time=~data.get_index("time").duplicated())
        data = data.resample(time="W-MON").mean()
        if lon[0]>lon[1]:
            data1 = data.sel(longitude=slice(None,lon[1]),latitude=slice(lat[1],lat[0]))
            data2 = data.sel(longitude=slice(lon[0],None),latitude=slice(lat[1],lat[0]))
            data = xr.concat([data2,data1],dim="longitude")
        else:
            data = data.sel(longitude=slice(*lon),latitude=slice(lat[1],lat[0])) 
        sets.append(data)
    return xr.merge(sets)

cmb_data = xarray_load([temperature_path, snow_path, precipitation_path], (0,360), (-90,90))

df = cmb_data.to_dataframe()
x = df.unstack().values

df.to_hdf(path+"data_full.h5", key='df', mode='w')
