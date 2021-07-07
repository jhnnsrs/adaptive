import cProfile
from pstats import Stats, SortKey

from model.psf import generate_psf, Settings, Aberration


if __name__ == '__main__':
    with cProfile.Profile() as pr:
        generate_psf(Settings(aberration=Aberration(a11=1)))
        filename = 'profile.prof'  # You can change this if needed
        pr.dump_stats(filename)