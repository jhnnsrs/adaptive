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
    a1: int = 0
    a2: int = 0
    a3: int = 0
    a4: int = 0
    a5: int = 0
    a6: int = 0
    a7: int = 0
    a8: int = 0
    a9: int = 0
    a10: int = 0
    a11: int = 0


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

    abberation: Aberration = Aberration()

    #sampling parameters
    Lfocal=2.5e-6                                                                        # observation scale
    Nx=100                                                                                #discretization of image plane
    Ny=100
    Nz=100
    Ntheta=50
    Nphi=50


@numba.jit()
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


def calculate_polar_matrix(phi, theta):
    return np.array([[1+(np.cos(phi)**2)*(np.cos(theta)-1), np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), -np.sin(theta)*np.cos(phi)],
    [np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), 1+np.sin(phi)**2*(np.cos(theta)-1), -np.sin(theta)*np.sin(phi)],
    [np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi), np.cos(theta)]]) 
#phase mask function

def heavy(phi, theta, X2, Y2, Z2, Ai, B, PM, W, wavenumber, deltatheta, deltaphi):
    term1=X2*np.cos(phi)+Y2*np.sin(phi)
    term2=np.multiply(np.sin(theta),term1)
    temp=np.exp(1j*wavenumber*(Z2*np.cos(theta)+term2))*deltatheta*deltaphi   # element by element
    return np.sin(theta)*Ai*PM*B*np.exp(1j*W)*temp 

def superheavy(phi, theta, X2, Y2, Z2, polarization, working_distance, beam_waist, pupil_radius, aberrations, unit_phase_radius, mode_val, wavenumber, deltatheta, deltaphi):
    
    T = calculate_polar_matrix(phi, theta)         # Pola matrix

    #incident beam polarization cases
    p0x=[1,0,1/np.sqrt(2),1j/np.sqrt(2),2/np.sqrt(5),np.cos(phi),-np.sin(phi)]
    p0y=[0,1,1j/np.sqrt(2),1/np.sqrt(2),1j/np.sqrt(5),np.sin(phi),np.cos(phi)]
    p0z= 0
    
    #selected incident beam polarization   
    P0=np.array([[p0x[polarization-1]],[p0y[polarization-1]],[p0z]])    # needs to be a colone vector
    #polarization in focal region
    P= np.multiply(T,P0)

    # Cylindrical coordinates on pupil
    rho_pup=working_distance*np.sin(theta)
    theta_pup=phi
    
    #Incident intensity profile
    Ai=np.exp(-rho_pup**2/(beam_waist**2))
    #Apodization factor
    B=np.sqrt(np.cos(theta))
    #Phase mask
    PM= phase_mask(rho_pup,theta_pup,unit_phase_radius*pupil_radius, mode_val)
    #Wavefront      
    W=  zernike(rho_pup/pupil_radius,theta_pup, *aberrations)
    #numerical calculation of field distribution in focal region
    factored = heavy(phi, theta, X2, Y2, Z2, Ai, B, PM, W, wavenumber, deltatheta, deltaphi)
    return np.array([factored*P[0,0],factored*P[1,0],factored*P[2,0]])

def phase_mask(rho, theta, cutoff_radius: float, mode: int):
    return np.exp(1j*theta)



def chunk_function(chunk,
                    X2,
                    Y2,
                    Z2,
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
    E = 0

    with tqdm(total=len(chunk)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    # Start the load operations and mark each future with its URL
                    future_to_url = [executor.submit(superheavy,
                    *pair,   
                    X2,
                    Y2,
                    Z2,
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
                            pbar.update(1)
                            future._result = None #Free memory my boy
                        except Exception as exc:
                            print("Exception")

    return E


def add(x,y):
    return x + y

def dask_function(chunk,
                    X2,
                    Y2,
                    Z2,
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
    E = 0

    print(chunk)

    # Start the load operations and mark each future with its URL
    zs = [client.submit(superheavy,
                    *pair,   
                    X2,
                    Y2,
                    Z2,
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


    print(zs)
    L = zs
    while len(L) > 1:
        new_L = []
        for i in range(0, len(L), 2):
            lazy = add(L[i], L[i + 1])  # add neighbors
            new_L.append(lazy)

        L = new_L     
        
    print(L)        
    return client.compute(L)





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

    aberrations = [s.abberation.a1,s.abberation.a2, s.abberation.a3, s.abberation.a4, s.abberation.a5, s.abberation.a6, s.abberation.a7, s.abberation.a8, s.abberation.a9, s.abberation.a10, s.abberation.a11]

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
        step*deltatheta) for q in range(s.Nphi+1) for step in range(s.Ntheta+1)]

    pairs = list(chunks(all_pairs, 1000))

   
    E = dask_function(all_pairs,
                    X2,
                    Y2,
                    Z2,
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
    return I.sum(axis=0)
