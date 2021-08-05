from model.psf import generate_psf,Settings, Aberration
import napari
import matplotlib.pyplot
from multiprocessing import Process, freeze_support
from dask.distributed import Client
from random import randint
import zarr as zr
import dask.array as da




if __name__ == '__main__':

    psf = generate_psf(Settings(a3=0.5))
    psf2 = generate_psf(Settings(a3=0.5, Ntheta=40, Nphi=40))
    psf3 = generate_psf(Settings(a3=0.5, Ntheta=60, Nphi=60))

    napari.view_image(da.stack([psf3, psf2, psf]))
    napari.run()