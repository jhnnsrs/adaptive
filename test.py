from types import LambdaType
from model.psf import Settings, Aberration
import napari
import matplotlib.pyplot
from multiprocessing import Process, freeze_support
from dask.distributed import Client
import dask
import dask.array as da
import time

def inc(NX, NY, NZ, theta):

    x = da.ones((NX, NY, NZ))
    time.sleep(2)

    return da.array([x*da.sin(theta),x*da.cos(theta),x*da.sin(theta)])







incdelayed = dask.delayed(inc, pure=True)



def generate_psf(s, client):

    x = inc(100,100,100,0)

    arrays = [da.from_delayed(incdelayed(100,100,100, i), shape=x.shape, dtype=x.dtype) for i in range(40000)]

    image = da.stack(arrays, axis=0)
    x = image.sum(axis=0)

    return x.compute()



if __name__ == '__main__':
    client = Client()
    psf = generate_psf(Settings(aberration=Aberration()), client)
    print(psf)