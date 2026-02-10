# save as: src/10_plot_domains.py
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import numpy as np

# Domain bounds (degrees)
SA = dict(name="South Atlantic",  lat_min=-35.0, lat_max=-20.0, lon_min=-25.0,  lon_max=-10.0)
NP = dict(name="North Pacific",   lat_min= 25.0, lat_max= 40.0, lon_min=-172.0, lon_max=-157.0)

def add_domain_box(m, ax, dom, edgecolor, facealpha=0.18, lw=2.2):
    # corners in lon/lat
    lons = [dom["lon_min"], dom["lon_max"], dom["lon_max"], dom["lon_min"]]
    lats = [dom["lat_min"], dom["lat_min"], dom["lat_max"], dom["lat_max"]]
    x, y = m(lons, lats)
    poly = Polygon(list(zip(x, y)), closed=True, fill=True,
                   facecolor=edgecolor, alpha=facealpha,
                   edgecolor=edgecolor, linewidth=lw, joinstyle="round")
    ax.add_patch(poly)

    # label near top-left corner (slightly offset)
    tx_lon = dom["lon_min"] + 0.3*(dom["lon_max"] - dom["lon_min"])
    tx_lat = dom["lat_max"] + 2.0
    tx, ty = m(tx_lon, tx_lat)
    ax.text(tx, ty, dom["name"], fontsize=11, fontweight="semibold",
            ha="center", va="bottom")

def main():
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)

    # Aesthetic-ish global projection
    m = Basemap(projection="robin", lon_0=-90, resolution="c", ax=ax)


    # Background
    m.drawmapboundary(fill_color="#f4f7fb", linewidth=0.6)
    m.fillcontinents(color="#e6e6e6", lake_color="#f4f7fb", zorder=1)
    m.drawcoastlines(color="#5a5a5a", linewidth=0.6, zorder=2)

    # Subtle graticule
    par = m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0],
                      color="#b7b7b7", linewidth=0.25, dashes=[2,2], fontsize=8)
    mer = m.drawmeridians(np.arange(-180, 181, 30), labels=[0,0,0,1],
                      color="#b7b7b7", linewidth=0.25, dashes=[2,2], fontsize=8)

    
    

    # Domain boxes
    add_domain_box(m, ax, SA, edgecolor="#1f77b4")  # blue-ish
    add_domain_box(m, ax, NP, edgecolor="#d62728")  # red-ish

    # Title (optional; keep minimal for paper)
    ax.set_title("ERA5 domains used for training (South Atlantic) and transfer testing (North Pacific)",
                 fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig("results/domains_era5_basins.pdf")   # vector, best for paper
    fig.savefig("results/domains_era5_basins.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()

