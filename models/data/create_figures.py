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
    Escrever
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
    Escrever
    """
    
    plt.figure()
    graph = plot2map(lon,lat,data)
    plt.colorbar()
    name  = str(pd.to_datetime(tmp)).replace(' ','_').split('_')
    date  = name[0]
    hour  = name[1]
    plt.savefig(f'./processed/figs/variable_{date}_{hour}.png', bbox_inches='tight')

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



# full_data = xr.concat((xr.open_dataset(file) for file in files), dim='time') 
# lat = full_data.latitude.values
# lon = full_data.longitude.values
# swh = full_data.swh.values #significant heigh
# time = full_data.time.values
# wind = full_data.wind.values #10m wind speed
# pp1d = full_data.pp1d.values #peak wave period
# dwi = full_data.dwi.values #10m wind direction

# # for i in tqdm(range(10)): #range(time.shape[0]):
    # plt.figure(i+1)
    # graph = plot2map(lon, lat, wind[0])
    # plt.colorbar()
    # plt.title(f'10m wind speed at {pd.to_datetime(time[0])}')
    # name = str(pd.to_datetime(time[i])).replace(' ','_').replace(':','_')
    # plt.savefig(f'./figures/{name}.png', bbox_inches='tight')

# fig = plt.figure()
# cont = plt.contourf(long, lati, swh[0])    # first image on screen
# plt.colorbar()

# animation function
# def animate(i):
#     global cont
#     z = swh[i]
#     for c in cont.collections:
#         c.remove()  
#     cont = plt.contourf(long, lati, z)
#     return cont

# anim = animation.FuncAnimation(fig, animate, frames=1000, repeat=False)
# writervideo = animation.FFMpegWriter(fps=60)
#anim.save('mymovie.mp4', writer=writervideo)
# plt.show()