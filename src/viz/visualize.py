import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Load the file
ds = xr.open_dataset("data/raw/AQUA_MODIS.20200715.L3m.DAY.CHL.chlor_a.4km.nc", engine="netcdf4")

# Print what's inside
print(ds)

# Grab chlorophyll variable
chl = ds['chlor_a']

# Crop to Long Island Sound bounding box
chl_lis = chl.sel(lat=slice(41.5, 40.5), lon=slice(-73.8, -71.8))

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

chl_lis.plot(
    ax=ax,
    norm=colors.LogNorm(vmin=0.1, vmax=20),
    cmap='YlGn',
    cbar_kwargs={'label': 'Chlorophyll-a (mg/m³)'}
)

ax.set_title('Long Island Sound - Chlorophyll-a - July 15, 2020')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('figures/eda/lis_chlorophyll_20200715.png', dpi=150)
plt.show()
print("Map saved.")