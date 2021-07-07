from model.psf import generate_psf,Settings, Aberration
import napari

psf = generate_psf(Settings(aberration=Aberration()))

napari.view_image(psf)
napari.run()