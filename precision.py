#%%
from run import Aberration
from model.psf import generate_psf, Settings
import zarr as zr
import xarray as xr
import matplotlib.pyplot as plt


experiment = f"loss_experiment_2"
ranges =  [1,2,10,20,30,40,50,60,80,90,100,120]


#%%
for i in ranges:
    psf = generate_psf(Settings(Ntheta=i, Nphi=i, aberration= Aberration(a1=0.4)))
    psf.to_dataset(name="data").to_zarr(zr.DirectoryStore(f"{experiment}/psf_{i}"))


#%%
def open_array(filepath):
    file = zr.DirectoryStore(filepath)
    dataset= xr.open_zarr(file)
    return dataset["data"]

def mean_squared(a,b):
    mse = ((a - b)**2).max()
    return mse.compute()


psf_ref = open_array(f"loss_experiment/psf_200")
mses = [mean_squared(psf_ref, open_array(f"{experiment}/psf_{i}")) for i in ranges]
comptime = [0,0,1,6,14,24,39,56,100,127,157,325]

plt.plot(ranges,comptime, drawstyle='steps',color='grey', alpha=0.7)
plt.plot(ranges,comptime, 'o--', )
plt.ylabel("Time in Seconds")
plt.xlabel("NTheta|NRho")
#plt.plot(ranges,com_time)
plt.show()
# %%
