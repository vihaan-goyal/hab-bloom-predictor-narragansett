import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Long Island Sound bounding box
NORTH, SOUTH = 41.5, 40.5
EAST, WEST = -71.8, -73.8

def load_day(filepath):
    ds = xr.open_dataset(filepath, engine="netcdf4")
    chl = ds['chlor_a']
    chl_lis = chl.sel(lat=slice(NORTH, SOUTH), lon=slice(WEST, EAST))
    mean_chl = float(chl_lis.mean(skipna=True))
    ds.close()
    return mean_chl

# Load all July 2020 files
data_dir = "data/raw"
records = []

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".nc"):
        continue
    if "DAY" not in fname or "CHL" not in fname:
        continue
    
    # Extract date from filename e.g. AQUA_MODIS.20200715.L3m...
    try:
        date_str = fname.split(".")[1]
        date = datetime.strptime(date_str, "%Y%m%d")
    except:
        continue
    
    filepath = os.path.join(data_dir, fname)
    mean_chl = load_day(filepath)
    records.append((date, mean_chl))
    print(f"{date.strftime('%Y-%m-%d')}: mean CHL = {mean_chl:.3f} mg/m³")

records.sort()
dates = [r[0] for r in records]
values = [r[1] for r in records]

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, values, color='green', linewidth=1.5, marker='o', markersize=4)
ax.fill_between(dates, values, alpha=0.2, color='green')
ax.set_title('Long Island Sound — Mean Chlorophyll-a over time')
ax.set_ylabel('Mean Chlorophyll-a (mg/m³)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/eda/lis_timeseries.png', dpi=150)
plt.show()
print("Saved.")