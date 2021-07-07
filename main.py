
from enum import Enum
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import LogNorm
from pydantic.main import BaseModel




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






# %%
#zernike function

def zernike(rho,theta, a: Aberration):
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
    zer=a.a1*Z1+a.a2*Z2+a.a3*Z3+a.a4*Z4+a.a5*Z5+a.a6*Z6+a.a7*Z7+a.a8*Z8+a.a9*Z9+a.a10*Z10+a.a11*Z11
    return zer


class Mode(str, Enum):
    GAUSSIAN= "GAUSSIAN"
    DONUT= "DONUT"
    BOTTLE= "BOTTLE"



class Polarization(int, Enum):
    X_LINEAR = 1
    Y_LINEAR = 2
    LEFT_CIRCULAR = 3
    RIGHT_CIRCULAR = 4
    ELLIPTICAL = 5
    RADIAL = 6
    AZIMUTHAL = 7





#phase mask function
def phase_mask(rho: np.array,theta: np.array, cutoff_radius: float, mode: Mode):
    if mode==Mode.GAUSSIAN:                          #guassian 
        mask=1
    elif mode==Mode.DONUT:                        #donut
        mask=np.exp(1j*theta)
    elif mode==Mode.BOTTLE:                        #bottleMo
        if rho<cutoff_radius:
            mask=np.exp(1j*np.pi)
        else :
            mask=np.exp(1j*0)
    else :
        raise NotImplementedError("Please use a specified Mode")
    return mask

                                                                          #polarization case (1:xlinear, 2:ylinear, 3:leftcircular, 4:rightcircular, 5:elliptical, 6:radial, 7:azimuthal)



# Geometry parameters
numerical_aperature=1.0                                                                               #numerical aperture of objective lens
working_distance=3e-3                                                                              #working distance of the objective in meter
refractive_index=1.33                                                                               #refractive index of immersion medium
radius_window =1.3                                                                           #radius of the cranial window (in mm)
thicknes_window = 2.23 #thichkness of the window (in mm)

polynomials = Aberration(a7=1, a8=1)                                                                      

# Beam parameters
wavelength= 592e-9                                                                   #wavelength of light in meter
unit_phase_radius = 0.5                                                              #radius of the ring phase mask (on unit pupil)
beam_waist= 0.008                                                                              #Gaussian beam waist (in mm)
mode= Mode.DONUT                                                                       
polar= Polarization.LEFT_CIRCULAR                                                                             


#%%
#Calculated parameters

# CRANIAL WINDOW ADJUSTMENTS


focusing_angle= np.arcsin(numerical_aperature/refractive_index)                                                                #maximum focusing angle of the objective



#effective_focusing_angle=min(atan(radius_window/thicknes_window),focusing_angle);                                           # effective focalization angle in presence of the cranial window
effective_focusing_angle = focusing_angle   


#effective_numerical_aperture= min(refractive_index*np.sin(effective_focusing_angle),numerical_aperature)    # Effective NA in presence of the cranial window
effective_numerical_aperture = numerical_aperature
wavenumber= 2*np.pi*refractive_index/wavelength  # wavenumber

pupil_radius = working_distance*np.tan(focusing_angle)
effective_pupil_radius=working_distance*np.tan(effective_focusing_angle)


#sampling parameters
Lfocal=2.5e-6                                                                        # observation scale
Nx=50                                                                                #discretization of image plane
Ny=50
Nz=50
Ntheta=50
Nphi=50



x1=np.linspace(-pupil_radius,pupil_radius,Nx)
y1=np.linspace(-pupil_radius,pupil_radius,Ny)
[X1,Y1]=np.meshgrid(x1,y1)



x2=np.linspace(-Lfocal,Lfocal,Nx)
y2=np.linspace(-Lfocal,Lfocal,Ny)
z2=np.linspace(-Lfocal,Lfocal,Nz)
[X2,Y2,Z2]=np.meshgrid(x2,y2,z2)


rho_pupil=np.sqrt(np.square(X1)+np.square(Y1))                                                     #cylindrical coordinates on pupil plane
theta_pupil=np.arctan2(Y1,X1)


#Initialization
Ex2=0                                                                               #Ex?component in focal
Ey2=0                                                                               #Ey?component in focal
Ez2=0     


                                                                         #Ez?component in focal
A_pupil=np.zeros(rho_pupil.shape)                                                   
mask_pupil=np.zeros(rho_pupil.shape)                                               
mask_pupil_eff=np.zeros(rho_pupil.shape)                                          
W_pupil=np.zeros(rho_pupil.shape)


#%%
#Beam calculation at the pupil plane 

(n1,n2)=rho_pupil.shape

#%%
#for i in range(0,n1):                                                               #Amplitude profile
 #   for j in range(0,n2):
 #       if rho_pupil[i,j]<=pupil_radius:
   #         A_pupil[i,j]=np.exp(-(X1[i,j]**2+Y1[i,j]**2)/beam_waist**2)
   #         mask_pupil[i,j]=np.angle(phase_mask(rho_pupil[i,j],theta_pupil[i,j],unit_phase_radius*pupil_radius, mode)) #phase mask
   #         W_pupil[i,j]=np.angle(np.exp(1j*zernike(rho_pupil[i,j]/pupil_radius,theta_pupil[i,j],polynomials))) #Wavefront        #np.angle(exp(1j*

#%%
A_pupil[rho_pupil<=pupil_radius] = np.exp(-((np.square(X1[rho_pupil<=pupil_radius])+np.square(Y1[rho_pupil<=pupil_radius]))/beam_waist**2))           #Amplitude profile              
mask_pupil[rho_pupil<=pupil_radius]=np.angle(phase_mask(rho_pupil[rho_pupil<=pupil_radius],theta_pupil[rho_pupil<=pupil_radius],unit_phase_radius*pupil_radius, mode))                           #phase mask
W_pupil[rho_pupil<=pupil_radius]= np.angle(np.exp(1j*zernike(rho_pupil[rho_pupil<=pupil_radius]/pupil_radius,theta_pupil[rho_pupil<=pupil_radius],polynomials)))                             #Wavefront



#%%

#Step of integral
deltatheta=effective_focusing_angle/Ntheta
deltaphi=2*pi/Nphi  

#Debye vector calculation
#for theta in range(0,alpha_eff,deltatheta):
    #for phi in range(0,2*pi,deltatheta):                #convertion function of polarization from object plane to imaging plane

#%%
theta=0
phi=0
for s in range (0,Ntheta+1):
    theta=s*deltatheta
    for q in range(0,Nphi+1):
        phi=q*deltaphi        
        T=[[1+(np.cos(phi)**2)*(np.cos(theta)-1), np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), -np.sin(theta)*np.cos(phi)],
        [np.sin(phi)*np.cos(phi)*(np.cos(theta)-1), 1+np.sin(phi)**2*(np.cos(theta)-1), -np.sin(theta)*np.sin(phi)],
        [np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi), np.cos(theta)]]           # Pola matrix

        #incident beam polarization cases
        p0x=[1,0,1/np.sqrt(2),1j/np.sqrt(2),2/np.sqrt(5),np.cos(phi),-np.sin(phi)]
        p0y=[0,1,1j/np.sqrt(2),1/np.sqrt(2),1j/np.sqrt(5),np.sin(phi),np.cos(phi)]
        p0z=0
        
        #selected incident beam polarization   
        P0=[[p0x[polar-1]],[p0y[polar-1]],[p0z]]    # needs to be a colone vector
        #polarization in focal region
        P=np.multiply(T,P0)

        # Cylindrical coordinates on pupil
        rho_pup=working_distance*np.sin(theta)
        theta_pup=phi
        
        #Incident intensity profile
        Ai=np.exp(-rho_pup**2/(beam_waist**2))
        #Apodization factor
        B=np.sqrt(np.cos(theta))
        #Phase mask
        PM=phase_mask(rho_pup,theta_pup,unit_phase_radius*pupil_radius, mode)
        #Wavefront      
        W=zernike(rho_pup/pupil_radius,theta_pup,polynomials)

        #numerical calculation of field distribution in focal region
        term1=X2*np.cos(phi)+Y2*np.sin(phi)
        term2=np.multiply(np.sin(theta),term1)

        temp=np.exp(1j*wavenumber*(Z2*np.cos(theta)+term2))*deltatheta*deltaphi   # element by element

        Ex2=Ex2+np.sin(theta)*Ai*PM*B*P[0,0]*np.exp(1j*W)*temp
        Ey2=Ey2+np.sin(theta)*Ai*PM*B*P[1,0]*np.exp(1j*W)*temp
        Ez2=Ez2+np.sin(theta)*Ai*PM*B*P[2,0]*np.exp(1j*W)*temp
        


#effective phase mask
for i in range(0,n1):                                                               #Amplitude profile
    for j in range(0,n2):
        if rho_pupil[i,j]<=pupil_radius:
            mask_pupil_eff[i,j]=-1.03*pi
        if rho_pupil[i,j]<=effective_pupil_radius:
            mask_pupil_eff[i,j]=mask_pupil[i,j]

#intensity of different components and total field

Ix2=np.abs(Ex2)**2
Iy2=np.abs(Ey2)**2
Iz2=np.abs(Ez2)**2
I1=Ix2+Iy2+Iz2
Ixy=Ix2+Iy2


#figures
# %%

#figure1 A_pupill
fig1, ax1=plt.subplots()
#c=ax1.imshow(A_pupil)
ax1.pcolor(X1,Y1,A_pupil,shading='auto',cmap='jet', vmin=np.min(A_pupil[A_pupil > 0]), vmax=np.max(A_pupil[A_pupil > 0]))
ax1.set_aspect('equal')
ax1.set_title('A_pupill')


#Figure2 mask_pupill
fig2, ax2=plt.subplots()
ax2.imshow(mask_pupil,interpolation='bicubic',cmap='jet')
#ax2.pcolor(X1,Y1,mask_pupil,shading='auto',cmap='jet')
ax2.set_aspect('equal')
ax2.set_title('mask_pupill')


#Figure 3 Mask_pupill_eff
fig3, ax3=plt.subplots()
ax3.pcolor(X1,Y1,mask_pupil_eff,shading='auto',cmap='jet')
ax3.set_aspect('equal')
ax3.set_title('mask_pupill_eff')

#Figure4 W_pupill
fig4, ax4=plt.subplots()
ax4.pcolor(X1,Y1,W_pupil,shading='auto',cmap='jet')
ax4.set_aspect('equal')
ax4.set_title('W_pupill')

#parametres

x2=np.linspace(-Lfocal,Lfocal,Nx)
y2=np.linspace(-Lfocal,Lfocal,Ny)
z2=np.linspace(-Lfocal,Lfocal,Nz)
[X2,Y2,Z2]=np.meshgrid(x2,y2,z2)


#figure 5 intensity

X=np.squeeze(X2[:,:,Nz//2])
Y=np.squeeze(Y2[:,:,Nz//2])
I=np.squeeze(Ixy[:,:,Nz//2])

fig5, ax5=plt.subplots()
ax5.imshow(I,interpolation='bicubic',cmap='jet')
ax5.set_aspect('equal')
ax5.set_title('Intensity XY')

#figure 6 intensity
Z=np.squeeze(Z2[:,Ny//2,:])
Y=np.squeeze(Y2[:,Ny//2,:])
I=np.squeeze(Ixy[:,Ny//2,:])

fig6, ax6=plt.subplots()
ax6.imshow(I,interpolation='bicubic',cmap='jet')
ax6.set_aspect('equal')
ax6.set_title('Intensity YZ')

#figure 7 intensity
Z=np.squeeze(Z2[Nx//2,:,:])
X=np.squeeze(X2[Nx//2,:,:])
I=np.squeeze(Ixy[Nx//2,:,:])

fig7, ax7=plt.subplots()
ax7.imshow(I,interpolation='bicubic',cmap='jet')
#ax7.pcolor(X*1e6,Z*1e6,I,shading='auto',cmap='jet')
ax7.set_aspect('equal')
ax7.set_title('Intensity XZ')


plt.show()

