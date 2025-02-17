import os
import glob
import numpy as np
from numpy.core.numeric import full 
import pandas as pd
import xarray as xr
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
from matplotlib import animation

def plot2map(lon, lat, dados):
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

if not os.path.exists('/Users/felipeminuzzi/Documents/OCEANO/Simulations/Wind_data/'+'figures'):
    os.mkdir('/Users/felipeminuzzi/Documents/OCEANO/Simulations/Wind_data/'+'figures')
path = '/Users/felipeminuzzi/Documents/OCEANO/Simulations/Wind_data/era5_reanalysis/'
files = sorted(glob.glob(path + '*'))

full_data = xr.concat((xr.open_dataset(file) for file in files), dim='time') 
lat = full_data.latitude.values
lon = full_data.longitude.values
swh = full_data.swh.values #significant heigh
time = full_data.time.values
wind = full_data.wind.values #10m wind speed
pp1d = full_data.pp1d.values #peak wave period
dwi = full_data.dwi.values #10m wind direction

# for i in tqdm(range(10)): #range(time.shape[0]):
#     plt.figure(i+1)
#     graph = plot2map(lon, lat, wind[0])
#     plt.colorbar()
#     plt.title(f'10m wind speed at {pd.to_datetime(time[0])}')
#     name = str(pd.to_datetime(time[i])).replace(' ','_').replace(':','_')
#     plt.savefig(f'./figures/{name}.png', bbox_inches='tight')

fig = plt.figure()
#llons, llats = np.meshgrid(lon, lat)
cont = plt.contourf(lon, lat, wind[0])    # first image on screen
plt.colorbar()

# animation function
def animate(i):
    global cont
    z = wind[i]
    for c in cont.collections:
        c.remove()  
    cont = plt.contourf(lon, lat, z)
    return cont

anim = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
writervideo = animation.FFMpegWriter(fps=60)
#anim.save('mymovie.mp4', writer=writervideo)
plt.show()