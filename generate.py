from model.psf import generate_psf,Settings, Aberration
import napari
import matplotlib.pyplot
from multiprocessing import Process, freeze_support
from dask.distributed import Client
from random import randint
import zarr as zr
from tqdm import tqdm
import os



if __name__ == '__main__':



    base_path = "first_night_run"

    SETTINGS_LIST = [
        Settings(abberation=Aberration(**{
            "a1": randint(0,100)/100,
            "a2": randint(0,100)/100,
            "a3": randint(0,100)/100,
            "a4": randint(0,100)/100,
            "a5": randint(0,100)/100,
            "a6": randint(0,100)/100,
            "a7": randint(0,100)/100,
            "a8": randint(0,100)/100,
            "a9": randint(0,100)/100,
            "a10": randint(0,100)/100,
            "a11": randint(0,100)/100,
        })) for i in range(100)]


    with tqdm(total=len(SETTINGS_LIST)) as pbar:
        for i, settings in enumerate(SETTINGS_LIST):
            store = zr.DirectoryStore(os.path.join(base_path,f"run_{i}"))
            psf = generate_psf(settings)
            psf.to_dataset(name=settings.abberation.to_name()).to_zarr(store)
            pbar.update(1)

    #psf = generate_psf(Settings(aberration=Aberration(a1=1)))

    