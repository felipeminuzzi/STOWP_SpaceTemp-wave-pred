import os
import glob
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
from matplotlib import animation

def plot2map(lon, lat, dados):
    """
    Create a 2D map consisting of the data specified in the lat x 
    lon dimension.

    *******************************************
   
    input:
        - lon: longitude
        - lat: latitude
        - dados: data to plot

    output:
        - m: the figure
    """
    map = Basemap(projection='cyl', llcrnrlon=lon.min(), 
                  llcrnrlat=lat.min(), urcrnrlon=lon.max(), 
                  urcrnrlat=lat.max(), resolution='h')
    
    map.fillcontinents(color=(0.55, 0.55, 0.55))

    map.drawcoastlines(color=(0.3, 0.3, 0.3))

    map.drawstates(color=(0.3, 0.3, 0.3))
    
    map.drawparallels(np.linspace(lat.min(), lat.max(), 6), labels=[1,0,0,0],
                      rotation=90, dashes=[1, 2], color=(0.3, 0.3, 0.3))
    map.drawmeridians(np.linspace(lon.min(), lon.max(), 6), labels=[0,0,0,1], 
                      dashes=[1, 2], color=(0.3, 0.3, 0.3))

    llons, llats = np.meshgrid(lon, lat)

    lons, lats = map(lon, lat)
    m = map.contourf(llons, llats, dados, 120, latlon=True)

    return m

def create_fig(data, lon, lat, tmp):
    """
    Save the figure created by 'plot2map' for each time step of the
    dataset.

    *******************************************
   
    input:
        - lon: longitude
        - lat: latitude
        - data: numpy array of data to create plot
        - tmp: index of the dataset representing the time step

    output:
        - saved figure in the specified path.
    """
    
    plt.figure()
    graph = plot2map(lon,lat,data)
    plt.colorbar()
    name  = str(pd.to_datetime(tmp)).replace(' ','_').split('_')
    date  = name[0]
    hour  = name[1]
    plt.savefig(f'./processed/figs/variable_{date}_{hour}.png', bbox_inches='tight')
    plt.close()

os.chdir('./data/')
raw_path  = './raw/'
raw_file  = glob.glob(raw_path + '*.nc')

if not os.path.exists('./processed/figs/'):
    os.mkdir('./processed/figs/')
else:
    shutil.rmtree('./processed/figs/')
    os.mkdir('./processed/figs/')


full_data = xr.open_dataset(raw_file[0])

lati      = full_data.latitude.values
long      = full_data.longitude.values
time      = full_data.valid_time.values
swh       = full_data.swh.values

[create_fig(swh[i], long, lati, time[i]) for i in tqdm(range(5000))]
#breakpoint()
