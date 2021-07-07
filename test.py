import cProfile
from pstats import Stats, SortKey

from model.psf import generate_psf, Settings, Aberration


if __name__ == '__main__':
    with cProfile.Profile() as pr:
         generate_psf(Settings(aberration=Aberration(a11=1)))

    with open('profiling_stats.txt', 'w') as stream:
        stats = Stats(pr, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('time')
        stats.dump_stats('.prof_stats')
        stats.print_stats()