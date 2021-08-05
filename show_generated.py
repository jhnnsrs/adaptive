#%%
import zarr as zr
import xarray as xr
from sklearn.metrics import mean_squared_error
import numpy as np
import napari
#%% 
def open_array(filepath):
    file = zr.DirectoryStore(filepath)
    dataset= xr.open_zarr(file)
    return dataset["data"]


def mean_squared(a,b):
    mse = ((a - b)**2).max()
    return mse.compute()







# %%
psf_100 = np.abs(open_array("psf_100"))
psf_200 = np.abs(open_array("psf_200"))
psf_50 = np.abs(open_array("psf_50"))
psf_20 = np.abs(open_array("psf_20"))






#%%
mean_squared(psf_200, psf_20)
#%%
mean_squared(psf_200, psf_100)

#%%

mse = ((psf_100 - psf_200)**2).mean()
mse.compute()
#%%
np.array(psf_100.sel(d=1))

#%%
dif_norm.compute().max()

#%%
psf_100
#%%
mean_squared_error(np.abs(psf_100), np.abs(psf_200))

# %%
array = dataset["a1-0.75_a2-0.28_a3-0.92_a4-0.29_a5-0.64_a6-0.17_a7-0.5_a8-0.06_a9-0.52_a10-0.75_a11-0.16"]

# %%
import napari 

napari.view_image(array)
# %%
