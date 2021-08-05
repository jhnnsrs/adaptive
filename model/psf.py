from enum import Enum
from re import X
from distributed import client
from pydantic import BaseModel
import numpy as np
import numba
from tqdm import tqdm
import dask
import concurrent.futures
from itertools import islice
from threadedprocess import ThreadedProcessPoolExecutor
from dask.distributed import as_completed
from numba.pycc import CC
import xarray as xr
import dask.array as da

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


class Mode(int, Enum):
    GAUSSIAN= 1
    DONUT=2
    BOTTLE=3



class Polarization(int, Enum):
    X_LINEAR = 1
    Y_LINEAR = 2
    LEFT_CIRCULAR = 3
    RIGHT_CIRCULAR = 4
    ELLIPTICAL = 5
    RADIAL = 6
    AZIMUTHAL = 7


class Aberration(BaseModel):
    a1: float = 0
    a2: float = 0
    a3: float = 0
    a4: float = 0
    a5: float = 0
    a6: float = 0
    a7: float = 0
    a8: float = 0
    a9: float = 0
    a10: float = 0
    a11: float = 0

    def to_name(self) -> str:
        return "_".join(map(lambda value: f"{value[0]}-{value[1]}",self.dict().items()))


class Settings(BaseModel):
    mode: Mode = Mode.DONUT
    polarization: Polarization = Polarization.LEFT_CIRCULAR
    # Geometry parameters
    numerical_aperature=1.0                                                                               #numerical aperture of objective lens
    working_distance=3e-3                                                                              #working distance of the objective in meter
    refractive_index=1.33                                                                               #refractive index of immersion medium
    radius_window =1.3                                                                           #radius of the cranial window (in mm)
    thicknes_window = 2.23

    # Beam parameters
    wavelength= 592e-9                                                                   #wavelength of light in meter
    unit_phase_radius = 0.5                                                              #radius of the ring phase mask (on unit pupil)
    beam_waist= 0.008 

    aberration: Aberration = Aberration()

    #sampling parameters
    Lfocal=2.5e-6                                                                        # observation scale
    Nx=64                                                                                #discretization of image plane
    Ny=64
    Nz=64
    Ntheta=20
    Nphi=20

@numba.njit(nogil=True)
def zernike(rho, theta, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11):
    Z1=1                                            
    Z2=2*rho*np.cos(theta)                                                                              #Tip
    Z3=2*rho*np.sin(theta)                                                                              #Tilt
    Z4=np.sqrt(3)*(2*rho**2-1)                                                                          #Defocus
    Z5=np.sqrt(6)*(rho**2)*np.cos(2*theta)                                                              #Astigmatisme
    Z6=np.sqrt(6)*(rho**2)*np.sin(2*theta)                                                              #Astigmatisme
    Z7=np.sqrt(8)*(3*rho**3-2*rho)*np.cos(theta)                                                        #coma
    Z8=np.sqrt(8)*(3*rho**3-2*rho)*np.sin(theta)                                                        #coma
    Z9=np.sqrt(8)*(rho**3)*np.cos(3*theta)                                                              #Trefoil
    Z10=np.sqrt(8)*(rho**3)*np.sin(3*theta)                                                             #Trefoil
    Z11=np.sqrt(5)*(6*rho**4-6*rho**2+1)                                                                #Spherical
    zer= a1*Z1+a2*Z2+a3*Z3+a4*Z4+a5*Z5+a6*Z6+a7*Z7+a8*Z8+a9*Z9+a10*Z10+a11*Z11
    return zer

@numba.njit(nogil=True)
def zernike2(rho, theta, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11):
    Z1=1.                                            
    Z2=2*rho*np.cos(theta)                                                                              #Tip
    Z3=2*rho*np.sin(theta)                                                                              #Tilt
    Z4=np.sqrt(3)*(2*rho**2-1)                                                                          #Defocus
    Z5=np.sqrt(6)*(rho**2)*np.cos(2*theta)                                                              #Astigmatisme
    Z6=np.sqrt(6)*(rho**2)*np.sin(2*theta)                                                              #Astigmatisme
    Z7=np.sqrt(8)*(3*rho**3-2*rho)*np.cos(theta)                                                        #coma
    Z8=np.sqrt(8)*(3*rho**3-2*rho)*np.sin(theta)                                                        #coma
    Z9=np.sqrt(8)*(rho**3)*np.cos(3*theta)                                                              #Trefoil
    Z10=np.sqrt(8)*(rho**3)*np.sin(3*theta)                                                             #Trefoil
    Z11=np.sqrt(5)*(6*rho**4-6*rho**2+1)                                                                #Spherical
    zer= a1*Z1+a2*Z2+a3*Z3+a4*Z4+a5*Z5+a6*Z6+a7*Z7+a8*Z8+a9*Z9+a10*Z10+a11*Z11
    return zer

@numba.njit(nogil=True)
def calculate_polar_matrix(phi, theta):
    return np.array([[1+(np.cos(phi)**2)*(np.cos(theta)-1), np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), -np.sin(theta)*np.cos(phi)],
    [np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), 1+np.sin(phi)**2*(np.cos(theta)-1), -np.sin(theta)*np.sin(phi)],
    [np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi), np.cos(theta)]]) 
#phase mask function

@numba.njit(nogil=True)
def heavy(phi, theta, X2, Y2, Z2, Ai, B, PM, W, wavenumber, deltatheta, deltaphi):
    term1=X2*np.cos(phi)+Y2*np.sin(phi)
    term2=np.multiply(np.sin(theta),term1)
    temp=np.exp(1j*wavenumber*(Z2*np.cos(theta)+term2))*deltatheta*deltaphi   # element by element
    return np.sin(theta)*Ai*PM*B*np.exp(1j*W)*temp 

@numba.njit(nogil=True)
def superheavy(phi, theta, X2, Y2, Z2, Lfocal, polarization, working_distance, beam_waist, pupil_radius, aberrations, unit_phase_radius, mode_val, wavenumber, deltatheta, deltaphi):

    
    T = calculate_polar_matrix(phi, theta)         # Pola matrix

    #incident beam polarization cases
    p0x=[complex(1),complex(0),1./np.sqrt(2),1j/np.sqrt(2),2/np.sqrt(5),np.cos(phi),-np.sin(phi)]
    p0y=[complex(1),complex(0),1j/np.sqrt(2),1/np.sqrt(2),1j/np.sqrt(5),np.sin(phi),np.cos(phi)]
    p0z=[complex(0)]
    
    #selected incident beam polarization   
    P0=np.array([[p0x[polarization-1]],[p0y[polarization-1]],p0z])   # needs to be a colone vector
    #polarization in focal region
    P= np.multiply(T,P0)

    # Cylindrical coordinates on pupil
    rho_pup=working_distance*np.sin(theta)
   
    
    #Incident intensity profile
    Ai=np.exp(-rho_pup**2/(beam_waist**2))
    #Apodization factor
    B=np.sqrt(np.cos(theta))

    theta_pup=phi
    #Phase mask
    PM= phase_mask(rho_pup,theta_pup,unit_phase_radius*pupil_radius, mode_val)
    #Wavefront     
    a1, a2, a3, a4, a5,a6,a7,a8,a9,a10,a11 = aberrations 
    W=  zernike2(rho_pup/pupil_radius,theta_pup, a1, a2, a3, a4, a5,a6,a7,a8,a9,a10,a11)
    #numerical calculation of field distribution in focal region
    factored = heavy(phi, theta, X2, Y2, Z2, Ai, B, PM, W, wavenumber, deltatheta, deltaphi)
    return np.stack((factored*P[0,0],factored*P[1,0],factored*P[2,0]))

@numba.njit(nogil=True)
def phase_mask(rho, theta, cutoff_radius: float, mode: int):
    return np.exp(1j*theta)



def chunk_function(pairs,
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client):
    E = np.zeros((3, X2,Y2,Z2), dtype=np.complex128)

    x2=np.linspace(-Lfocal,Lfocal,X2)
    y2=np.linspace(-Lfocal,Lfocal,Y2)
    z2=np.linspace(-Lfocal,Lfocal,Z2)
    [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)

    test = superheavy(*pairs[0],
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi)

    the_chunks = list(chunks(pairs, 800))

    with tqdm(total=len(pairs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Start the load operations and mark each future with its URL

                    for key, chunk in enumerate(the_chunks):
                        pbar.set_description(f"{key}/{len(the_chunks)}")

                        future_to_url = [executor.submit(superheavy,
                        *pair,   
                        X2,
                        Y2,
                        Z2,
                        Lfocal,
                        polarization, 
                        working_distance,
                        beam_waist, 
                        pupil_radius,
                        aberrations, 
                        unit_phase_radius,
                        mode_val, 
                        wavenumber,
                        deltatheta, 
                        deltaphi) for pair in chunk]
                        

                        for future in concurrent.futures.as_completed(future_to_url):
                            try:
                                E += future.result()
                                future._result = None #Free memory my boy
                                pbar.update(1)
                            except Exception as exc:
                                print(exc)


    return E



def chunk_function_small(pairs,
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client):
    E = np.zeros((3, X2,Y2,Z2), dtype=np.complex128)

    x2=np.linspace(-Lfocal,Lfocal,X2)
    y2=np.linspace(-Lfocal,Lfocal,Y2)
    z2=np.linspace(-Lfocal,Lfocal,Z2)
    [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)

    test = superheavy(*pairs[0],
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi)

    the_chunks = list(chunks(pairs, len(pairs)))

    with tqdm(total=len(pairs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Start the load operations and mark each future with its URL

                    for key, chunk in enumerate(the_chunks):
                        pbar.set_description(f"{key}/{len(the_chunks)}")

                        future_to_url = [executor.submit(superheavy,
                        *pair,   
                        X2,
                        Y2,
                        Z2,
                        Lfocal,
                        polarization, 
                        working_distance,
                        beam_waist, 
                        pupil_radius,
                        aberrations, 
                        unit_phase_radius,
                        mode_val, 
                        wavenumber,
                        deltatheta, 
                        deltaphi) for pair in chunk]
                        

                        for future in concurrent.futures.as_completed(future_to_url):
                            try:
                                E += future.result()
                                future._result = None #Free memory my boy
                                pbar.update(1)
                            except Exception as exc:
                                print(exc)


    return E




def no_parallelfunction(chunk,
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client):
    E = np.zeros((3, X2,Y2,Z2), dtype=np.complex128)

    x2=np.linspace(-Lfocal,Lfocal,X2)
    y2=np.linspace(-Lfocal,Lfocal,Y2)
    z2=np.linspace(-Lfocal,Lfocal,Z2)
    [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)

    with tqdm(total=len(chunk)) as pbar:
        # Start the load operations and mark each future with its URL
        for pair in chunk:
            E += superheavy(
                *pair,   
                X2,
                Y2,
                Z2,
                Lfocal,
                polarization, 
                working_distance,
                beam_waist, 
                pupil_radius,
                aberrations, 
                unit_phase_radius,
                mode_val, 
                wavenumber,
                deltatheta, 
                deltaphi)


            pbar.update(1)

    return E



def add(x,y):
    return np.array(x) + np.array(y)

def add3(x,y,z):
    return np.array(x) + np.array(y) + np.array(z)

def dask_function(chunk,
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client):
    E = np.zeros((3, X2,Y2,Z2), dtype=np.complex128)

    x2=np.linspace(-Lfocal,Lfocal,X2)
    y2=np.linspace(-Lfocal,Lfocal,Y2)
    z2=np.linspace(-Lfocal,Lfocal,Z2)
    [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)
    # Start the load operations and mark each future with its URL

    zs = [client.submit(superheavy,
                    *pair,   
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi) for pair in chunk]


    L = zs
    while len(L) > 1:
        print(len(L))
        new_L = []
        if len(L) % 2 == 1:
            for i in range(0, len(L) - 3 , 2):
                lazy = dask.delayed(add)(L[i], L[i + 1])  # add neighbors
                new_L.append(lazy)
            new_L.append(dask.delayed(add3)(L[i], L[i + 1], L[i+2]))
        else:
            for i in range(0, len(L), 2):
                lazy = dask.delayed(add)(L[i], L[i + 1])  # add neighbors
                new_L.append(lazy)
            

        L = new_L     

    res = L[0].compute()
    print(res)
              
    return res 

def dask_precomputed_function(chunk,
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client):


    Eshape = (3, X2,Y2,Z2 )

    @dask.delayed()
    def parallelizable(rho, theta,X2,Y2,Z2,Lfocal, *args):

        x2=np.linspace(-Lfocal,Lfocal,X2)
        y2=np.linspace(-Lfocal,Lfocal,Y2)
        z2=np.linspace(-Lfocal,Lfocal,Z2)
        [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)

        return superheavy(rho, theta, X2, Y2, Z2, Lfocal, *args)


    arrays = [da.from_delayed(parallelizable(
                    *pair,   
                    X2,
                    Y2,
                    Z2,
                    Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi), shape=Eshape, dtype=np.complex128) for pair in chunk]


    zs = da.stack(arrays)
    
    graph = zs.sum(axis=0)

    return graph.compute()





def generate_psf(s: Settings, client=None):

    #Calulcated Parameters

    focusing_angle= np.arcsin(s.numerical_aperature/s.refractive_index)                                                                #maximum focusing angle of the objective
    #effective_focusing_angle=min(atan(radius_window/thicknes_window),focusing_angle);                                           # effective focalization angle in presence of the cranial window
    effective_focusing_angle = focusing_angle   

    #effective_numerical_aperture= min(refractive_index*np.sin(effective_focusing_angle),numerical_aperature)    # Effective NA in presence of the cranial window
    effective_numerical_aperture = s.numerical_aperature
    wavenumber= 2*np.pi*s.refractive_index/s.wavelength  # wavenumber

    pupil_radius = s.working_distance*np.tan(focusing_angle)
    effective_pupil_radius=s.working_distance*np.tan(effective_focusing_angle)


    # Sample Space
    x1=np.linspace(-pupil_radius,pupil_radius,s.Nx)
    y1=np.linspace(-pupil_radius,pupil_radius,s.Ny)
    [X1,Y1]=np.meshgrid(x1,y1)

    
    x2=np.linspace(-s.Lfocal,s.Lfocal,s.Nx)
    y2=np.linspace(-s.Lfocal,s.Lfocal,s.Ny)
    z2=np.linspace(-s.Lfocal,s.Lfocal,s.Nz)
    [X2,Y2,Z2]=np.meshgrid(x2,y2,z2)

    rho_pupil= np.sqrt(np.square(X1)+np.square(Y1))                                                     #cylindrical coordinates on pupil plane
    theta_pupil= np.arctan2(Y1,X1)

    A_pupil= np.zeros(rho_pupil.shape)                                                   
    mask_pupil= np.zeros(rho_pupil.shape)                                               
    mask_pupil_eff= np.zeros(rho_pupil.shape)                                          
    W_pupil= np.zeros(rho_pupil.shape)

    aberrations = [s.aberration.a1,s.aberration.a2, s.aberration.a3, s.aberration.a4, s.aberration.a5, s.aberration.a6, s.aberration.a7, s.aberration.a8, s.aberration.a9, s.aberration.a10, s.aberration.a11]

    A_pupil[rho_pupil<pupil_radius] = np.exp(-((np.square(X1[rho_pupil<pupil_radius])+np.square(Y1[rho_pupil<pupil_radius]))/s.beam_waist**2))           #Amplitude profile              
    mask_pupil[rho_pupil<pupil_radius]=np.angle(phase_mask(rho_pupil[rho_pupil<pupil_radius],theta_pupil[rho_pupil<pupil_radius],s.unit_phase_radius*pupil_radius, s.mode.value))                           #phase mask
    #W_pupil[rho_pupil<=pupil_radius]= np.angle(np.exp(1j*zernike(rho_pupil[rho_pupil<=pupil_radius]/pupil_radius,theta_pupil[rho_pupil<=pupil_radius],*aberrations)))                             #Wavefront
    W_pupil[rho_pupil<pupil_radius]= np.angle(np.exp(1j*zernike(rho_pupil[rho_pupil<pupil_radius]/pupil_radius,theta_pupil[rho_pupil<=pupil_radius],*aberrations)))                             #Wavefront


    
    #Step of integral
    deltatheta=effective_focusing_angle/s.Ntheta
    deltaphi=2*np.pi/s.Nphi  

    E = 0 

    polarization = s.polarization
    beam_waist = s.beam_waist
    unit_phase_radius = s.unit_phase_radius
    mode_val = s.mode.value
    working_distance = s.working_distance


    all_pairs = [
        (q*deltaphi,
        step*deltatheta) for q in range(s.Nphi) for step in range(s.Ntheta)]

    pairs = list(chunks(all_pairs, 1000))

   
    E = chunk_function_small(all_pairs,
                    s.Nx,
                    s.Ny,
                    s.Nz,
                    s.Lfocal,
                    polarization, 
                    working_distance,
                    beam_waist, 
                    pupil_radius,
                    aberrations, 
                    unit_phase_radius,
                    mode_val, 
                    wavenumber,
                    deltatheta, 
                    deltaphi,
                    client,)
   
    I =np.abs(E)**2
    PSF = I.sum(axis=0)
    return xr.DataArray(PSF, dims=list("xyz"), attrs={**s.dict(exclude={"aberration"}), **s.aberration.dict()})
