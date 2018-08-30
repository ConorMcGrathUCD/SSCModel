
# -*- coding: utf-8 -*-
"""
Synchrotron Self Compton Spectrum Calculator

First working build finished Fri Jun  8 16:40:09 2018

Author: Conor McGrath 
Under supervision of Prof. John Quinn 
High Energry Astrophysics reseach group
University College Dublin
Masters in Space Science and Technology research project.

Example input : main1(0,0.047, 30. ,2.7, 0.04, 1.4E+16, 0.8, 3.5, 13.5, 11.45, 2., 3., 1, 1E+9, 4., 600.,Finke)

Edit:16/07/18 : Using Fiachra Kennedy's EBL calculator improves speed but requires limits on E values. 
Finke : e(TeV) = [0.002, 79], z = [0.00, 4.99]
Franceschini : e(TeV) = [0.03, 166], z = [0.001, 2]
Dominguez : e(TeV) = [0.04, 29], z = [0.01, 2]
"""

import numpy as np
from simpsint import simpsint 
from time import strftime
#from mpmath import mp
import math
from numba import jit
from Combined_Class import EBL 
#mp.dps = 50
"""
Mathematical Constants
"""
LOG10 = 2.302585093
RtD = 57.295779513082325

"""
Physical Constants in Gauss Units 
"""
CGS_c=  2.997925e10        #   cm/s
CGS_me=     9.109e-28      #   g
CGS_me_erg= 8.1760E-7      #   erg
CGS_mp    = 1.6726E-24     #   g
CGS_e  =    4.8066E-10     #   statcoul 
CGS_k  =    1.3807E-16     #   erg/K
CGS_re =    2.818E-13      #   cm 
CGS_pc =    3.0856E+18     #   cm
CGS_h  =    6.6261E-27     #   erg s
CGS_hbar =  1.054589E-27   #   erg s
CGS_tcs  =  6.653E-25      #   Thompson Cross section in cm^2
CGS_SBK  =  5.5670400e-5   #   erg s^-1 cm^-2 K^-4
CGS_MSUN =  1.98892E+33    #   Solar mass in grams
CGS_G    =  6.67300E-8     #   Gravitational Constant in dyne cm^2 gm^-2


"""
Physical Constants in SI Units 
"""
SI_c    =2.997925E+08      #  m/s
SI_h    =6.6261E-34        #  J s
SI_hbar =1.05457657E-34    #  J s
SI_me   =9.109E-31         #  kg 
SI_mp   =1.6726E-27        #  kg 
SI_e    =1.6022E-19        #  C  
SI_k    =1.3807E-23        #  J/K
SI_re   =2.818E-15         #  m 
SI_e0   =8.8542E-12        #  C^2/m^2 N 
SI_pc   =3.0856E+16        #  m 
SI_tcs  =6.653E-29         #  Thompson Cross section in m^2
SI_SBK  =5.5670400e-8      #  W m^-2 K^-4

SI_MSUN =1.98892E+30       #  kg
SI_G	=6.67300E-11       #  m^3 kg^-1 s^-2

"""
Conversion Factors
"""
C_me_eV =511E+3           #   eV 
NU_eV   =4.136E-15
NU_TeV  =4.136E-27
eV_NU   =2.41779E+14
NU_erg  =6.6176E-27
J_eV    =6.242197253e+18
eV_J    =1.602E-19
J_erg   =1E+7             #  Joule in ergs
erg_J   =1E-7
eV_erg  =1.6E-12  
erg_eV  =6.25E+11 

"""
Binning and integration values
These can be altered to increase accuracy
"""
MAX_BINS = 1001
NUMEL =400 
NUMSYNC = 400
NUMIC = 73
NUMSUM = 100
JONES_STEP = 10
KLEIN_NISHINA = 5000
JONES_NISHINA = 50000
DISKBINS = 60
NMODEL = 5

"""
Defining Data array sizes
"""
numbins = 0
data = {
        'f' : [0]*MAX_BINS,
        'fe' : [0]*MAX_BINS,

        'f1' : [0]*MAX_BINS,
        'f2': [0]*MAX_BINS,

        'nu' : [0]*MAX_BINS,

        'idn' : [0]*MAX_BINS
        }

t_x = [ 0.001,0.005,0.01 ,0.025,0.05 ,0.075,0.1  ,0.15 ,0.2  ,0.25 ,
		0.3  ,0.4  ,0.5  ,0.6  ,0.7  ,0.8  ,0.9  ,   1.,1.2  ,1.4  ,
		1.6  ,1.8  ,2.   ,2.5  ,3.   ,3.5  ,4.   ,4.5  ,5.   ,6.   ,
		7.   ,8.   ,9.   ,10.]
t_v = [
		0.213,0.358,0.445,0.583,0.702,0.722,0.818,0.874,0.904,0.917,
		0.919,0.901,0.872,0.832,0.788,0.742,0.694,0.655,0.566,0.486,
		0.414,0.354,0.301,0.200,0.130,0.0845,0.0541,0.0339,0.0214,0.0085,
		0.0033,0.0013,0.0005,0.00019]
	
"""
Initilizing source specific variable dictionary
"""
source = {
          'h0' : 0.0,
          'Omega_M' : 0.0,
          'Omega_Lambda' : 0.0,
          'distance' : 0.0,
          'redshift' : 0.0,
          'radius' : 0.0,
          'volume' : 0.0,
          'B' : 0.0,
          'Gamma' : 0.0,
          'theta' : 0.0,
          'beta' : 0.0,
          'delta' : 0.0,
          'lum' : 0.0,
          
          'ECflag' : 0.0,
          'Mbh' : 0.0,
          'Mdot' : 0.0,
          'blobDist' : 0.0,
          
          'E_min' : 0.0,
          'E_max' : 0.0,
          'E_break' : 0.0,
          'pow1' : 0.0,
          'pow2' : 0.0,
          'gamma_max' : 0.0,
	
          'Nu_max_sy' : 0.0,
          'Nu_max_ic' : 0.0,
          'Po_max_sy' : 0.0,
          'Po_max_ic' : 0.0,
          'od1': 0.0,
          'pp1' : 0.0,
	
          'Nu_KN_lim1' : 0.0,
          'Nu_KN_lim2' : 0.0,
          'chi2' : 0.0,
          'dof' : 0,
          
          'EBLModel' : ""
          }

"""
Main1() is main contorl centre. Primiarily calls other functions and prints initial parameters.
Sets default zalues for undefined variables.
"""   
#def main1(argv):
def main1(z, gamma, theta, B, r, w_p_soll, Emin, Emax, Ebreak, p1, p2, ECFlag, Mbh, Mdot, blobDist, EBLModel):        

    source['h0'] = 0.7
    source['Omega_M'] = 0.3
    source['Omega_Lambda'] = 0.7

    age_Gyr,zage_Gyr,DTT_Gyr,DA_Mpc,kpc_DA,DL_Mpc,DL_Gyr = 0.0,0.0,0.0,0.0,0.0,0.0,0.0

    'Calling NedWright to calculate DL_Mpc (luminosity distance) for further calculations'
    DL_Mpc = NedWright(z, source['h0'], source['Omega_M'], source['Omega_Lambda'], age_Gyr, zage_Gyr, DTT_Gyr, DA_Mpc, kpc_DA, DL_Mpc, DL_Gyr)
    
    'Mpc to m'
    d_l = DL_Mpc * 1e6* SI_pc

    print("Luminosity distance %e Mpc \n" % DL_Mpc)
    
    source['redshift'] = float(z)
    source['distance'] = d_l
    
    """
    Initilising variables for electrons, synchrotron, inverse Compton, disk, external Compton and sum spectra.
    arg 1 = Dictionary
    arg 2 = log10 of lower energy limit
    arg 3 = log10 of upper energy limit
    arg 4 = Number of bins. Greater value = more accuracy 
    """
     
    """
    Electron Spectrum from 1e4 to 1e15.
    Energy in eV = ple['e']
    Electron Density in m^-3 eV^-1 = ple['f']
    """
    ple = init_st(4, 15, NUMEL)
    """
    Synchrotron Spectrum from 1e4 to 1e25
    Frequency in Hz = psync['e']
    Flux in  erg cm^-2 s^-2 = psync['f']
    """
    psync = init_st(4.0, 25.0, NUMSYNC)
    """
    Inverse Compton Spectrum from 1e10 to 1e27.5
    Frequency in Hz = pic['e']
    Flux in  erg cm^-2 s^-2 = pic['f']
    """
    pic = init_st(10.0, 27.5, NUMIC)
    """
    Disk Spectrum from 1e4 to 1e25
    Frequency in Hz = pbb['e']
    Flux in  erg cm^-2 s^-2 = pbb['f']
    """
    pbb = init_st(4.0, 25.0, NUMSYNC)
    """
    External Compton Spectrum from 1e10 to 1e28.5
    Frequency in Hz = pec['e']
    Flux in  erg cm^-2 s^-2 = pec['f']
    """    
    pec = init_st(10.0, 28.5, NUMIC)
    """
    Sum Spectrum from 1e4 to 1e30
    Frequency in Hz = psum['e']
    Flux in  erg cm^-2 s^-2 = psum['f']
    """    
    psum = init_st(4.0, 30.0, NUMSUM)
    
    idn = 1
    
    source['lum'] = 1

    source['Gamma'] = gamma
    source['theta'] = theta/RtD
    source['B'] = B
    source['radius'] = r
    w_p_soll = w_p_soll

    source['beta'] = math.sqrt(1-1/(source['Gamma']**2))
    source['delta'] = 1/(source['Gamma']*(1-source['beta']*math.cos(source['theta'])))
    
    source['volume'] = 4/3*math.pi*(source['radius']**3)
    
    
    source['E_min'] =  Emin
    source['E_max'] =  Emax
    source['E_break'] =  Ebreak
    
    if(source['E_min'] < ple['x1']):
        source['E_min'] =  ple['x1'] + ple['delta']

    if(source['E_max'] > ple['x2']):
        source['E_max'] = ple['x2'] - ple['delta']
        
    if(source['E_break'] < source['E_min']):
        source['E_break'] = source['E_min'] 

    if(source['E_break'] > source['E_max'] ):
        source['E_break']  = source['E_max']

    source['chi2'] = 0.0
   
    source['pow1'] = -p1
    source['pow2'] = -p2

    source['ECflag'] = ECFlag
    source['Mbh'] = Mbh
    source['Mdot'] = Mdot
    source['blobDist'] = blobDist
    source['EBLModel'] = EBLModel

    ple = zero_flux(ple)

    ple = fill_le(ple,source['E_min'],source['E_max'],source['E_break'], source['lum'],source['pow1'],source['pow2'])
   
    w_p = 0
    'Calculating energy density'
    for i in range(0, ple['numbins']):
        w_p+= (ple['f'][i])*(ple['de'][i])*(ple['e'][i])*eV_erg

    w_p /= 1e6 #energy density in erg cm^-3

    source['lum'] *= w_p_soll/w_p
    
    ple = zero_flux(ple)
    
    info(source)
    
    spectrum(idn+1, source, ple, psync, pic, pbb, pec, psum, data)
"""
Prints info/
"""

def info(source):
    x = source['distance']/SI_pc/1e6
    y = source['radius']/SI_c/3600
    z = source['radius']/SI_c/3600/source['delta']
    #print ('%.2f' % var1,'kg =','%.2f' % var2)
    print(type(RtD))
    print("Gamma %.8f \n" % source['Gamma'])
    print("Theta: %.8f \n" % (source['theta']*RtD))
    print("Beta: %.8f \n" % source['beta'])
    print("Redshift: %.8f \n" % source['redshift'])
    print("Distance: %.6e m \n" % source['distance'] )
    print("        : %.6e Mpc \n" % x)
    print("Radius  : %.6e m \n" % source['radius'])
    print("Radius/c: %.6f hrs \n" % y)
    print("Radius/c/delta: %.6f hrs \n" % z)
    print("B       : %.6f T \n" % source['B'] )
    print("lum(1eV) : %.6e es/m^3 eV \n" % source['lum'])
    print("E_min : %.6f log10(E/eV) \n" % source['E_min'])
    print("E_max : %.6f log10(E/eV) \n" % source['E_max'])
    print("E_break : %.6f log10(E/eV)\n" % source['E_break'] )
    print("pow1 %.6f  \n" % source['pow1'])
    print("pow2 %.6f \n" % source['pow2'])
    print("Delta-Jet %.6f \n" % source['delta'])
    
    
"""
Prints further info. 
Calculates some values within.
Args passed are used to calculate printed values 
"""
def info2(source, ple, psync, pic):
    
    w_sy=0
    
    sy_emax = 0.2*1e9*source['B']*((10**source['E_max'])/1e12)**2    #eV
    sy_numax = source['delta'] * sy_emax/NU_eV
    
    print("Expected Sy Peak: %.6e Hz \n" % sy_numax)
    print("Sy Peak: %.6e Hz \n" % source['Nu_max_sy'])
    print("Sy Peak Power: %.6e ergs/cm^2 s \n" % source['Po_max_sy'])
    print("SSA up to: %.6e Hz \n" %source['od1'])
    
    ic_emax = 5*sy_emax*1e3*(((10**source['E_max'])/1e12)**2)*1e9    #eV
    ic_numax = source['delta'] * ic_emax/NU_eV
    
    print("Expected IC Peak: %.6e Hz \n" % ic_numax)
    print("Ic Peak: %.6e Hz \n" % source['Nu_max_ic'])
    print("Ic Peak Power: %.6e ergs/cm^2 s \n" % source['Po_max_ic'])
    
    print("Pair opaque above: %.6e Hz \n" %source['pp1'])
    'Beginning of Klein-Nishina limit'    

    print("Maximal gamma_e: %.6e \n" % source['gamma_max'])
    
    lim1 = C_me_eV/source['gamma_max']
    lim2 = 5*(lim1*1e+3) * ((((10.**source['E_max']))/(1e+12))**2)* 1e+9 #eV
	
    source['Nu_KN_lim1'] = source['delta'] * lim1/NU_eV 
    source['Nu_KN_lim2'] = source['delta'] * lim2/NU_eV 

    print("KN (Sy) at %.6e Hz \n" % source['Nu_KN_lim1']) 
    print("KN (IC) at %.6e Hz \n" % source['Nu_KN_lim2'])  
	
    'Energy density of magnetic field in erg/cm^3'
    w_B = ((source['B']*1E+4)**2)/8/np.pi

    "Energy density of synchrotron radiation in erg/cm^3"
    w_sy = 0
    for i in range(0,psync['numbins']):
        w_sy += (psync['n'][i]*psync['de'][i]*psync['e'][i]*NU_erg)
    w_sy /= 1E+6
	
    "Energy density of particles in erg/cm^3"
    w_p=0.;
    for i in range(0,ple['numbins']):
        w_p += (ple['f'][i]*ple['de'][i]*ple['e'][i]*eV_erg)
    w_p /= 1E+6
	
    print("\n\nw_B : %e erg/cm**3\n" % w_B)
    print("w_sy: %e erg/cm**3\n"% w_sy)
    print("w_p : %e erg/cm**3\n" % w_p)
    print("w_p/w_b: %.6e \n" % (w_p/w_B))
	
    "Synchrotron cooling time"
    t_sync = 3./4.*(C_me_eV/(w_B*erg_eV)) / (SI_tcs *10000. *SI_c *100.)/((10.**source['E_break'])/C_me_eV)
    t_ad   = source['radius'] / SI_c
	
    print("\nt_sync         %.6e hrs at an electron energy of %.6E eV\n" % ((t_sync/3600),(10.**source['E_break'])))
    print("t_sync obs     %.6e hrs at an electron energy of %.6E eV\n" % ((t_sync/3600./source['delta']),(10.**source['E_break'])))
    print("t_ad           %.6e hrs\n" % (t_ad/3600.))
    print("t_ad   obs     %.6e hrs\n" % (t_ad/3600./source['delta']))
	
	
    "Gamma of break energy"
    gamma_b  = 3.*C_me_eV /4. /(1.3*w_B*erg_eV *source['radius']*100. *SI_tcs *10000.)
    b_energy = gamma_b * C_me_eV;
    print("Expected break of electron energy spectrum: Gamma_b = %.6E = %.6E eV\n\n"% (gamma_b,b_energy))
    
    
"""
Calculates synch spectra
"""

def synchrotron(ple, psync, B):
    x = ple['numbins']
    'Range of electron energies'
    for i in range (0, x):
        if (ple['f'][i] > 0):
            energy = ple['e'][i]

            'dN/dE dV * Delta E = Weighting factor'
            weight = ple['f'][i]* ple['de'][i]
            
            'Calculating spectra'
            psync = sync_spec(psync, energy, weight, B)
    return psync
        
"""
Check is and ia. why 13??
Calculates synchrotron spectrum
Result in power/freq, vol, time, solid angle
dp(nu)/dnu dV dt dOmega
dp(eV), nu(Hz), vol(m^3), t(s), solid ang(sr) 
"""
def sync_spec(psync, energy, flux, B):
    dxx, dg, di              = [0.0]*MAX_BINS, [0.0]*MAX_BINS, [0.0]*MAX_BINS 
    n_gdiv = SI_e*B/2/np.pi/SI_me
    
    'electron Gamma factor. Energy in eV'
    gamma = energy/C_me_eV
    
    'All synch frequencies'
    for i in range(0,psync['numbins']):
        nu = psync['e'][i] #Hz
        dalpha = np.pi/24
        alpha = 0.0
        'All pitch angles'
        for j in range(0,13):
            dg[j] = 0.0
            if (alpha>0) and (alpha<np.pi-0.0001):
                'Crit frequency'
                nu_c = 1.5*(gamma**2)* n_gdiv*math.sin(alpha)
                f = F(nu/nu_c) 
                """
                Malcolm Logair, Edt.3, pg 208, eq.8.58
                (Jp/Omega)
                """
                p = math.sqrt(3)/8/(np.pi**2)*(SI_e**3)*B*math.sin(alpha)/SI_e0/SI_me/SI_c*f*0.5*math.sin(alpha)
                'per freq'
                p *= (2*np.pi)*J_eV*2
                dg[j] = float(p)
                
            alpha += dalpha
        'Emissivity j (eV m^-3 s^-1 sr^-1)'
        psync['j'][i] += flux * simpsint(j,dg,di,dxx,0.0,np.pi/2)/4/np.pi
    return psync
    

"""
Calculate self Compton spectrum
"""
def selfCompton(ple, pseed, pic, r, idn):

    weight, energy = 0.0, 0.0
    flag = 0
    'electron energies'
    for i in range(0, ple['numbins']):
        if(ple['f'][i]>0):
            
            energy = ple['e'][i]

            'dN/dE dV * Delta E = Weighting factor'
            weight = ple['f'][i]*ple['de'][i]

            'Calculates spectra'
            pic = inv_spec(pseed, pic, energy, weight, r, i, idn, flag)
            
    return pic

"""
Calculates self Compton spectrum
Result in power/freq, vol, time, solid angle
dp(nu)/dnu dV dt dOmega
dp(eV), nu(Hz), vol(m^3), t(s), solid ang(sr) 
"""

def inv_spec(pseed, pic, energy, flux, r, ie, idn, flag):
    
    dxx, dg, di = [0.0]*MAX_BINS,[0.0]*MAX_BINS,[0.0]*MAX_BINS

    'electron Gamma factor (eV)'
    gamma = energy/C_me_eV
    fac1 = 2*np.pi*(SI_re**2)*C_me_eV *SI_c / gamma # m^3 s^-1
    
    'All IC spec frequecies'
    for i in range(0, pic['numbins']):

        e_1 = NU_eV * pic['e'][i]
        e_2 = NU_eV * fxen(pic, i-0.5)
        e_3 = NU_eV * fxen(pic, i+0.5)
        
        E2 = e_2/energy
        E3 = e_3/energy
        
        dE1 = E3-E2
        
        'Integrate over seed photon frequency'
        for j in range(0, pseed['numbins']):
            dg[j] = 0.0
            if (pseed['j'][j]>0):

                'Energy in eV'
                e = NU_eV*pseed['e'][j]
                G_e = 4* e/C_me_eV * gamma
                
                bl = e/energy
                bu = G_e/(1+G_e)
                
                if(E3<bl):
                    continue
		
                if(E2>bu):
                    continue
                
                if((E2>bl) and (E3<bu)):
                    
                    q2 = E2/G_e/(1-E2)
                    q3 = E3/G_e/(1-E3)
         
                elif((E2>bl) and (E3>bu)):
                    q2 = E2/G_e/(1 - E2)
                    q3 = 1
                 
                elif((E2<bl) and (E3<bu)):
                    
                    q2   = 0.25/(gamma**2)
                    q3   = E3/G_e/(1.-E3)
                
                else:
                    q2   = 0.25/(gamma**2)
                    q3   = 1
                    
                'Blumenthal and Gould, Review of Modern Physics 42, pg 237, eq. 2.48'

                x1 = fac1*pseed['n'][j]/e
                x4 = aver_jones(G_e,q2,q3,1.0,dE1)
                
                dn = x1*x4
                
                dg[j] = float(pseed['e'][j]*dn)

        if((j%2)!=1):
            j -=1
        pseed1 = float(pseed['x1']*LOG10)
        pseed2 = float(pseed['x2']*LOG10)
        help1 = simpsint(j,dg, di, dxx, pseed1, pseed2)
        dn2 = flux * dE1 *help1
       
        'Emissivity (eV m^-3 s^-1 sr^-1)'
        pic['j'][i] += dn2*e_1/pic['de'][i]/4/np.pi #4pi per steradian
    
    return pic

"""
Calculates bracketed part of Blumenthal and Gould, Review of Modern Physics 42, pg 237, eq. 2.48

 2.*q*(np.log(10)+np.log(q/10))+(1.+2.*q)*(1.-q)+0.5*((G_e*q)**2)/(1.+G_e*q)*(1.-q) )

"""
@jit
def aver_jones(G_e,q2,q3,delta,dEs):
    """
    steps should be ~20 times smaller than the distance of the maximum in the scattered photon density in energy space from the maximum achievable energy
    step = minimum(step,(1.-(G_e/(2.+G_e)))/20.);	 
    """
    sums = 0.0
    if(q3<1):
        qmax   = q3
    else:
        qmax = 1.0
    E1     = q2*G_e/(1.+q2*G_e)

    ifirst = 0
    x = (2./(2.+G_e))
    
    y = ((qmax-q2)/JONES_STEP)
    if(x<y):
        z = x
    else:
        z = y
    if(z<0.04):
        schritt = z
    else:
        schritt = 0.04
    #schritt    = min(0.04,min(2./(2.+G_e),(qmax-q2)/JONES_STEP))
    if(schritt < (qmax-q2)/(JONES_NISHINA)):
        schritt = (qmax-q2)/(JONES_NISHINA)
    
    #schritt    = max((qmax-q2)/JONES_STEP/KLEIN_NISHINA,schritt)
	
    if (q2<0.1):
        qstart = q2+min(0.01, schritt/2)

    else:
        qstart     = q2+schritt/2.

    q = qstart
    while (q <= qmax):
        
        E2          = q*G_e/(1.+q*G_e)
        weight      = (E2 - E1)/dEs
        E1          = E2
		
        if (ifirst==1):
            if (G_e<1E+5):
                #sums += weight *( 2.*q*(LOG10+math.log(q/10))+(1.+2.*q)*(1.-q)+0.5*((G_e*q)**2)/(1.+G_e*q)*(1.-q) )
                sums += weight *( 2.*q*(math.log(q))+(1.+2.*q)*(1.-q)+0.5*((G_e*q)**2)/(1.+G_e*q)*(1.-q))
            else:
                sums += weight * 0.5* G_e*q*(1.-q)

        q   += schritt
        ifirst =1

    #res = sums
    return sums
    #if (res!=0.):
     #   return res
    #else:
   #     return 0.0

"""
V.L. Ginzburg & S.I. Syrovatski, The Origin of Cosmic Rays, 1964, p. 402
Calculated exactly as given in literature
"""
def F(x):

#    t_x = [ 0.001,0.005,0.01 ,0.025,0.05 ,0.075,0.1  ,0.15 ,0.2  ,0.25 ,
#		0.3  ,0.4  ,0.5  ,0.6  ,0.7  ,0.8  ,0.9  ,   1.,1.2  ,1.4  ,
#		1.6  ,1.8  ,2.   ,2.5  ,3.   ,3.5  ,4.   ,4.5  ,5.   ,6.   ,
#		7.   ,8.   ,9.   ,10.]
#    t_v = [
#		0.213,0.358,0.445,0.583,0.702,0.722,0.818,0.874,0.904,0.917,
#		0.919,0.901,0.872,0.832,0.788,0.742,0.694,0.655,0.566,0.486,
#		0.414,0.354,0.301,0.200,0.130,0.0845,0.0541,0.0339,0.0214,0.0085,
#		0.0033,0.0013,0.0005,0.00019]
	
    if (x <= 0.001):
        res = (4.*np.pi/math.sqrt(3)/Gamma(1./3.)*((x/2.)**(1./3.))*(1.-Gamma(1./3.)/2.*((x/2.)**(2./3.))+ 3./4.* (x/2.)**2))
        
    elif (x >= 10):
        res = ((np.pi/2.)*math.exp(-x) * math.sqrt(x) * ((1.+55./72./x)-(10151./10368./(x**2))))
    else:
        res = get_table(t_x,t_v,34,x)
    return res	

		
"""
V.L. Ginzburg & S.I. Syrovatski, The Origin of Cosmic Rays, 1964, p. 402
Calculated exactly as given in literature
Never used
"""
#def F_p(x):
# 
#    t_x = [0.001,0.005,0.01 ,0.025,0.05 ,0.075,0.1  ,0.15 ,0.2  ,0.25 ,
#           0.3  ,0.4  ,0.5  ,0.6  ,0.7  ,0.8  ,0.9  ,   1.,1.2  ,1.4  ,
#           1.6  ,1.8  ,2.   ,2.5  ,3.   ,3.5  ,4.   ,4.5  ,5.   ,6.   ,
#           7.   ,8.   ,9.   ,10.]
#    t_v = [0.107,0.184,0.231,0.312,0.388,0.438,0.475,0.527,0.560,0.582,
#           0.596,0.607,0.603,0.590,0.570,0.547,0.521,0.494,0.439,0.386,
#		0.336,0.290,0.250,0.168,0.111,0.0726,0.0470,0.0298,0.0192,0.0077,
#		0.0031,0.0012,0.00047,0.00018]
#	
#    if (x <= 0.001):
#        res = (2.*np.pi/math.sqrt(3)/Gamma(1./3.)*((x/2.)**(1./3.))*(1.-3./2.*Gamma(1./3.)/Gamma(2./3.)*((x/2.)**(4./3.))+ 3.*(x/2.)**2))
#        return res
#    
#    elif (x >= 10):
#        res = ((np.pi/2.)*math.exp(-x) * math.sqrt(x) *(1+7./72./x-455./10368./(x**2)))
#        return res;
#    res = get_table(t_x,t_v,34,x)
#    return res			

"""
Takes exponential of ln(Gamma) to give Gamma
"""
def Gamma(x):
    return math.exp(gammln(x))
    
"""
Lanczos approximation of Gamma function
C. Lanczos, 'A Precision Approximation of the Gamma Function', SIAM, 1964, pg. 86-96
Adapted to numerical recipe
"""
def gammln(xx):
    
    cof=[76.18009172947146,-86.50532032941677,
         24.01409824083091,-1.231739572450155,
         0.1208650973866179e-2,-0.5395239384953e-5]
    y=x=xx;
    tmp=x+5.5
    tmp -= (x+0.5)*math.log(tmp)
    ser=1.000000000190015
    for i in range (0,6):
        ser += cof[i]/(1+y)
    return -tmp+math.log(2.5066282746310005*ser/x)
    
"""
Returens referenced values(x) from inputed tables(t_x and t_v)

"""
def get_table(t_x, t_v,numbins,x):
    count = 0
    if ((x<t_x[0]) or (x>t_x[numbins-1])):
        print("Error in get_table : %f %f %f\n"%(x,t_x[0],t_x[numbins-1]))
        exit (-1)

    for i in range(0,numbins-1):
        if (x<t_x[i]): 
            count = i
            break
	
    delta =  np.log(t_x[count])-np.log(t_x[count-1])
    rest  = (np.log(x     )-np.log(t_x[count-1]))/delta
    res   = (1.-rest)*t_v[count-1] + rest*t_v[count]
    return res
    

"""
from Malcolm Longair, Edt. 2, p. 223, tbl. 8.2
"""
def b_factor(x):

    t_x = [1.   ,1.5  ,2.   ,2.5  ,3.   ,3.5  ,4.   ,4.5  ,5.]
    t_v = [0.397,0.314,0.269,0.244,0.233,0.230,0.236,0.248,0.268]
	
    if ((x < t_x[0]) or (x>t_x[8])):
        print("Bad argument in b_factor: %f\n"%x)
        exit(-1)
    res = get_table(t_x,t_v,9,x)
    return res


"""
Initialises beginning spectrum
"""
def init_st(p1, p2, numbins):
    i = 0
    p = {
         'x1' : p1,
         'x2' : p2,
         'numbins' : numbins,
         'delta' : (p2 - p1)/numbins,
         'f' : [0]*numbins,
         'e' : [0]*numbins,
         'de' : [0]*numbins,
         'nu' : [0]*numbins,
         'j' : [0]*numbins,
         'n' : [0]*numbins,
         'a' : [0]*numbins,
         'tau_i' : [0]*numbins,
         'tau_e' : np.zeros(shape = (NMODEL,MAX_BINS))
         }
    for i in range (0, p['numbins']):
        p['e'][i] = xen(p,i)
        p['de'][i] = fxen(p, float(i)+0.5)-fxen(p,float(i)-0.5)
        p['nu'][i] = xen(p,i) #p['e'][i]

    return p
    

"""
Prints inputted spectrum to relevant time/date stamped .txt files
"""
def print_st(ple, ichoice, source, idn):
    strftime("%Y-%m%d %H:%M:%S")
    x = [0]*4
    
    if(ichoice == 0):
        filename = strftime("%Y_%m_%d_%H_%M_%S") + "_Electron_Spectrum"
    elif (ichoice == 1):
        filename = strftime("%Y_%m_%d_%H_%M_%S") +"_Synchrotron_Spectrum"
    elif (ichoice == 2):
        filename = strftime("%Y_%m_%d_%H_%M_%S") +"_SelfCompton_Spectrum"
    elif (ichoice == 3):
        filename = strftime("%Y_%m_%d_%H_%M_%S") +"_Disk_Spectrum"
    elif (ichoice == 4):
        filename = strftime("%Y_%m_%d_%H_%M_%S") +"_External_Compton_Spectrum"
    elif (ichoice == 5):
        filename = strftime("%Y_%m_%d_%H_%M_%S") +"_Sum_Spectrum"
    
    f = open(filename + '.txt' , 'w')

    if(ichoice == 0):
        source['gamma_max'] = (10**source['E_max'])/C_me_eV
        
    for i in range (0, ple['numbins']):
        
        if (ichoice == 0):
            x[0] = ple['e'][i]
            x[1] = ple['f'][i]
        elif (ichoice > 0):
            x[0] = ple['nu'][i]/(1+source['redshift'])
            x[1] = ple['f'][i]
            x[2] = ple['f'][i]*np.exp(max(-1*ple['tau_i'][i],-20))
            x[3] = (x[2]*np.exp(max(-(ple['tau_e'][0][i]),-20.)))
        if (idn>0):
            f.write("%i %e %e %e %e \n" % (i,x[0],x[1],x[2],x[3]))
    if(idn > 0):
        f.close()
        

"""
Calls all functions to calculate spectra.
Work is done here.
"""
def spectrum(idn, source, ple, psync, pic, pbb, pec, psum, data):
    
    'Sets electron values to zero'
    ple = zero_flux(ple)
    
    'Electron spectrum is filled'
    ple = fill_le(ple, source['E_min'], source['E_max'],source['E_break'],source['lum'],source['pow1'],source['pow2'])

    B = source['B']
    """
    Calculate synchrotron spectrum
    arg 1: Electron spec
    arg 2: Sync spec
    arg 3: B field (Tesla)
    """
    psync = synchrotron(ple, psync, B)
    """
    Calculate synchrotron self absorption coefficient and luminosity (eV/Hz)
    arg 1: Synchrotron spectrum 
    arg 2: Source variables
    """
    psync = ssa(psync, source)
    """
    Calculate inverse Compton spectrum
    arg 1: Electron spec
    arg 2: Seed photon spec
    arg 3: Inverse Compton spec
    arg 4: Emitting region radius (m)
    """
    pic = selfCompton(ple, psync, pic, source['radius'], idn)
    """
    Calculate absorption of inverse Compton emission inside emission volume
    arg 1: Synchrotron spec
    arg 2: Inverse Compton spec
    arg 3: Source varables
    """
    pic = pair_absorption(psync, pic, source)

    'If disk emission is included'
    if (source['ECflag']==1):
        """
        Calculate accretion disk
        arg 1: Source variables
        arg 2: Disk spec
        arg 3: Electron spec
        arg 4: External Compton spec
        arg 5: Inverse Compton spec
        """
        pbb, pec['j'] =calculateDisk(source, pbb, ple, pec, pic)
        """
        Calculate absorption of external Compton 
        arg 1: Synchrotron spec
        arg 2: External Compton spec
        arg 3: Source varables
        """
        pec = pair_absorption(psync, pec, source)
    """
    Calculates flux recieved at Earth accounting for Doppler boosting based in emissivities
    Flux = nu F_nu (erg/cm^2)
    arg 1: Spectrum calculated
    arg 2: Source variables
    arg 3: Choice of methond (1 = synch, 2 = inverse Comp, 3 = external Comp)
    """
    psync, source = comp_flux(psync, source, 1)
    pic, source  = comp_flux(pic, source, 2)
    
    if(source['ECflag']):
        pec, source = comp_flux(pec, source, 3)

    'Calculates IC absorption from Diffuse Extragalactic BG Radiation'
    pic= debra_absorption(pic, source)
    if (source['ECflag']):
        pec = debra_absorption(pec, source)
    'Sums total flux received'
    psum = sum_flux(source, psync,pic,pbb,pec,psum)
    
    'Saves results to .txt files'
    print_st(ple, 0, source, idn)
    print_st(psync, 1, source, idn)
    print_st(pic, 2, source, idn)
    
    if(source['ECflag']):
         print_st(pbb, 3, source, idn)
         print_st(pec, 4, source, idn)   
 
    print_st(psum, 5, source, idn)
    'Prints synchrotron and self Compton info'
    if(idn>0):
         info2(source, ple, psync, pic)

"""
Sets all values of given spectrum to 0
"""         
def zero_flux(ple):
    for i in range(0, ple['numbins']):
        ple['f'][i] = 0
        ple['nu'][i] = 0
        ple['j'][i] = 0
        ple['n'][i] = 0
        ple['a'][i] = 0
        ple['tau_i'][i] = 0
        ple['tau_e'][0][i] = 0
    return ple

"""
Calculates internal pair absorption
"""
def pair_absorption(psync,pic,source):
    """
    /* emissivity -> luminosity in eV/Hz s */
    """
	
    flag          = 1
    source['pp1'] = 1E+99
	
    for i in range(0,pic['numbins']):

        if (pic['j'][i]>0.):

            x   = pic['e'][i] *NU_eV / C_me_eV
            nu  = 1./x *C_me_eV *eV_NU

            n    = get_density(psync,nu)
            corr = (C_me_eV/NU_eV)**2
			
            tau = 0.
            if (n>0.): 
			
                tau = 0.2*SI_tcs * (n/pic['e'][i]*corr) *source['radius']

                #tau = 0.
                pic['tau_i'][i] = tau
                if ((tau>1.) and (flag)): # I feel like it never gets into this statement??
                    print("It got to this tau...", tau)
                    flag = 0
                    source['pp1'] = pic['e'][i] * source['delta']
				
                pic['a'][i] = kappa_nu = tau/2./source['radius']
                
            if (tau<1E-3):
                fac = 4./3.*source['radius']

            else:
                fac = (1.-2./(tau**2)*(1.-math.exp(-tau)*(tau+1.)))/kappa_nu
                #print("FAC is" , fac)	
            pic['f'][i] = np.pi*(source['radius']**2)* 4.*np.pi*pic['j'][i] *fac

    return pic
         
"""
Calcs IC absorption from Diffuse Extragal BG Rad
"""
def debra_absorption(ple,source):	

    for i in range(0, ple['numbins']):

        if ((ple['f'][i]>0.) and (ple['e'][i] > 1E+20) and (ple['e'][i] < 1E+29)):

            energy = NU_TeV*ple['nu'][i]			
#            ple['tau_e'][0][i] = tau1 = absorb(np.log10(energy),source['redshift'],1,source['h0'],source['Omega_M'], source['Omega_Lambda']) 
            if (source['EBLModel'] == "Finke" and energy < 0.002):
                energy = 0.002
            if (source['EBLModel'] == "Franceschini" and energy < 0.03):
                energy = 0.03
            if (source['EBLModel'] == "Dominguez" and energy < 0.04):
                energy = 0.04
            ple['tau_e'][0][i] = EBL(str(source['EBLModel'])).Find_Tau(source['redshift'], energy)
    return ple
         
"""
Calculates actual absorption
arg 3(model) refersd to EBL model. In this case Franceschini model
Reference: Franceschini, Alberto, Rodighiero, Giulia, Vaccari, Mattia, 2008, arXiv0805.1841.

Each function called (dx(), de(), franceschini()) are all steps from the paper to compute final integral
"""
"""
Calculates Disk geometry and emissions
"""
@jit
def calculateDisk(source,pbb, ple, pec, pic):
    NUMPAVER = 18
    NUMABSORB = 150
	
    print("\n Disk Emission\n\n")
	
    Ledd=1.25E+47*source['Mbh']/1E+9
    
    'Mass of black hole (kg and Msun)'
    print("Mass of Black Hole: %e solar masses\n"%source['Mbh'])
    source['Mbh'] *= SI_MSUN
    print("Mass of Black Hole: %e kg\n"%source['Mbh'])
    
    'Schwarzschild Radius'
    RSch = 2.*SI_G*source['Mbh']/(SI_c**2)
    print("Schwarzschild radius %e m\n" % RSch)
    
    'Accretion rate of central blackhole'
    print("Accretion rate: %e solar masses per year\n"%source['Mdot'])
    source['Mdot'] *= SI_MSUN/(365.*86400.)
    print("Accretion rate: %e kg s-1\n"%source['Mdot'])
    print("Accretion rate: %e erg s-1\n"% (source['Mdot']*1000.*((CGS_c)**2)))
    
    'blob distance (Rsch and m)'
    print("blobDist: %e RSch\n"%source['blobDist'])
    source['blobDist'] *= RSch
    print("blobDist: %e m\n"%source['blobDist']);
    print("blobDist: %e times the blob radius (<1 is not a good choice)\n"%(source['blobDist']/source['radius']))
    
    'Loops over segments of the accretion disk'
    ri=3.*RSch
    r1 = ri
    sumFlux=0
    sumFlux2=0
    
    fac = (4.*np.pi*(source['distance']**2))
    ifl=0
    
    for i in range(0,DISKBINS):
        r2 = r1*1.1
        rc = (r1+r2)/2.
    	
        area    = np.pi * ((r2**2)-(r1**2))
        areaCos = area*math.cos(source['theta'])
    		
        Fr = 3.*SI_G*source['Mbh']*source['Mdot']/(8.*np.pi*(rc**3))*(1.-1./math.sqrt(rc/ri));
        T  =((Fr/SI_SBK)**(1./4.))
    		
        sumFlux += areaCos * Fr #W
    		
        'loop over disk frequencies'
        for j in range(0, pbb['numbins']):
			
            ifl = 0
            """
            compute energy spectrum from disk
    
            compute blackbody energy density (in a cavity),not leaking out of a cavity 
            source of missing 2pi I believe (has now been multipplied in)
            """
            uw = 8*np.pi*SI_h*((pbb['e'][j]/SI_c)**3)/(np.exp(SI_h*pbb['e'][j]/(SI_k*T))-1.) # Rybicki, Lightman "Radiative processes in Astrophysics" pg. 19

            'specific intensity	'
            
            Inu  = SI_c*uw
 
            '1/2=2pi/4pi, 2pi from d/dw=>d/dnu and 1/4pi from isotropic uw => to d/dOmega uw'
    
            'luminosity of ring segment '
            Lnu =  areaCos*np.pi*Inu # // F= A * integral dOmega Inu*cos(theta)=A*PI*I 
    			
            'flux received on Earth (frequency redshifted in print_st)'
            pbb['f'][j] += pbb['e'][j]*Lnu/fac
            sumFlux2 += Lnu*pbb['de'][j]
            

            'compute inverse Compton emission'	
            if (ifl): 
                print("Bin %i Radius %e Disk Bin %d Disk photon frequency %e\n"%(i,rc,j,pbb['e'][j]))

            if (ifl): 
                print("uw %e Inu %e Lnu %e\n"%(uw,Inu,Lnu))
    
            'frequency and intensity of pbb photons in jet frame'
            mu = -1./math.sqrt(1.+(rc/source['blobDist'])**2)
            muDash = (mu+source['beta'])/(1.+source['beta']*mu)
            deltaJet = source['Gamma']*(1.+source['beta']*mu)
            nuDash = deltaJet*pbb['e'][j]
            InuDash = (deltaJet**3)*Inu

            if (ifl): 
                print("mu %e muDash %e deltaJet %e deltaJet2 %e nuDash %e InuDash %e\n"%(
        			mu,muDash,deltaJet,1./(source['Gamma']*(1.-source['beta']*muDash)),nuDash,InuDash))
    			
    			
            'solid angle of disk segment (jet frame)'
            mu1 = -1./math.sqrt(1.+(r1/source['blobDist'])**2)
            mu2 = -1./math.sqrt(1.+(r2/source['blobDist'])**2)
            mu1Dash = (mu1+source['beta'])/(1.+source['beta']*mu1)
            mu2Dash = (mu2+source['beta'])/(1.+source['beta']*mu2)
            deltaOmegaDash = 2.*np.pi*(mu2Dash-mu1Dash)
            if (ifl): 
                print("mu1 %e mu2 %e mu1Dash %e mu2Dash %e deltaOmegaDash %f\n"%(mu1,mu2,mu1Dash,mu2Dash,deltaOmegaDash))
    			
            'jet frame photon energy density from ring segment integrated over all solid angles'
            uNuDash = InuDash/SI_c*deltaOmegaDash

            """
            jet frame density of target photons, 
            density of considered frequency bin (n'(nu) * Delta nu)' 
            """
            nDash = uNuDash / (SI_h * nuDash) * (deltaJet*pbb['de'][j]) #'d/dnuDash nDash ->  d/dnuDash nDash x Delta_nuDash'
            if (ifl): 
                print("uNuDash %e nDash %e \n"%(uNuDash,nDash))

        
                 
            
            """	
            jet frame cos of electron pitch angle for 
            electrons producing oberved emission
            """
            muJet = math.cos(source['theta']) 
            muEl = (muJet-source['beta'])/(1.-source['beta']*muJet);
            ez  = muEl;
            ex  = math.sqrt(1.-(muEl**2));
            if (ifl): 
                print("muJet %e muEl %e \n"%(muJet,muEl))
    			
            'electron energy spectrum'
            for k in range (0, ple['numbins']):
                
                'external compton emissivity,(blob frame) '
                gammaEl = ple['e'][k]/C_me_eV
                betaEl=np.sqrt(1.-1./(gammaEl**2))
                
                'average over azimuthal directions '
                
        				
                pz  = muDash;
                px0 = math.sqrt(1.-(mu**2));
    				
                for l in range(0, NUMPAVER):
                    'cos of angle between photon and electron velocities'
                    px = px0*math.cos(l*np.pi/NUMPAVER)
                    muPhEl = ez*pz+ex*px;
        					
                    'doppler boost disk photon frequency into electron rest frame'
                    deltaEl = gammaEl*(1.+betaEl*muPhEl)
                    nuEl = deltaEl*nuDash
        					
                    'photon energy (m_e)'
                    y = nuEl*NU_eV/C_me_eV;
        					
                    'energy transfer (eV)'
                    if (y<1.): 
                        DeltaE = gammaEl*y*C_me_eV # // Thomson
                    else: 
                        DeltaE = gammaEl*C_me_eV # // Klein Nishina
                    
                    'Klein Nishina cross section'
                    if (y<1E-2):
                        sigma = SI_tcs
                    else:
                        sigma = (SI_tcs * 3./4. *((1+y)/(y**3)*(2.*y*(1.+y)/(1.+2.*y)-math.log(1.+2.*y))+1./2./y*math.log(1.+2.*y)-(1.+3.*y)/((1.+2.*y)**2)))
                					
                    'power emitted by electrons (d/dOmega d/dE N(E) * Delta E)'
                    p = (ple['f'][k] * ple['de'][k] * SI_c * (1.+muPhEl)  * sigma * nDash/(NUMPAVER) * DeltaE/(4.*np.pi))
                    'all quantities are computed in the jet frame'
                										
                    xl=np.log10(DeltaE*eV_NU)
                    rest=0.;
                					
                    if ((ifl) and (l==0)): 
                        print("electron bin %d gammaEl %e betaEl %e direction cosines ez %f ex %f pz %f px %f\n"%(
                							   k,gammaEl,betaEl,ez,ex,pz,px))
                        print("deltaEl %e photon energy y %e sigma [tcs] %e log10 DeltaE [ev] %f power %e\n"%(
                							   deltaEl,y,sigma/SI_tcs,xl,p))
                					
                    if ((xl>pec['x1']) and (xl<pec['x2'])):
                        ec, rest = bins(pec, xl, rest)
                        if ((ifl) and (l==0)): 
                            print("EC BIN %d\n\n" % ec)
                
                        'emissivity (eV m^-3 s^-1 sr^-1'
                        pec['j'][ec] += p / pec['de'][ec] # // power per frequency interval 
            
            'absorption of selfCompton and externalCompton photons leaving galaxy'
            
            d1=source['blobDist']
            
            if (source['theta']>10./180.*np.pi): 
                print("ATTENTION: gamma-ray absorption in the disk photon field is computed assuming that the jet angle is small (<10 degree)\n")    
            for k in range(0, NUMABSORB):

                d2=d1*1.2
                dc=(d1+d2)/2.
				
                'solid angle of disk segment in the AGN frame */'
                mu1 = -1./math.sqrt(1.+(r1/dc)**2)
                mu2 = -1./math.sqrt(1.+(r2/dc)**2)
                deltaOmega = 2.*np.pi*(mu2-mu1)
				
                'AGN frame photon energy density from ring segment integrated over all solid angles'
                uNu = Inu/SI_c*deltaOmega
				
                'AGN frame target photon density integrated over width of energy bin' 
                n = uNu / (SI_h * pbb['e'][j])*pbb['de'][j]
                'absorption'
                mu = -1./np.sqrt(1.+(rc/dc)**2);
                """
                Absorption of External Compton and Inverse Compton in AGN frame
                """
                for l in range (0,2):

                    if (l==0): 
                        p=pec
                    else: 
                        p=pic
					
                    for m in range (0,p['numbins']):
                        nuThr = (2. * ((SI_me*SI_c*SI_c)**2)/((SI_h**2)*(source['delta']*p['e'][m])*(1.+mu)))
                        if (nuThr > pbb['e'][j]): 
                            continue
                        betaGG = np.sqrt(1.-nuThr/pbb['e'][j])
                        sigmaGG = (3./16.*SI_tcs*(1.-(betaGG**2))*(2.*betaGG*((betaGG**2)-2.)+(3.-((betaGG**2)**2))*math.log((1.+betaGG)/(1.-betaGG))))
                        deltaTau = sigmaGG * n * (1.+mu) * (d2-d1)/math.cos(source['theta'])
    
                        pec['tau_i'][m] += deltaTau
                d1=d2
        if (i % 10==0):
            print("r1 %e r2 %e T %e K sumFlux %e W \n"%(
				   r1,r2,T,sumFlux))
		
        r1=r2;
    print("Total disk luminosity: %e W, %e erg s-1\n" % (sumFlux,sumFlux*J_erg))
    print("Consistency check: %e W, %e W\n" % (sumFlux,sumFlux2))
    print("Radiative efficiency: %e percent\n" % (100.*sumFlux*J_erg/(source['Mdot']*1000.*(CGS_c**2))))
    print("Eddington luminosity: %e erg s-1\n" % Ledd)
    print("Disk luminosity in units of the Eddington luminosity %e\n\n" % (sumFlux*J_erg/Ledd))
    return pbb, pec['j']



"""
compute fluxes, given the emissivities and the source parameters 
"""
def comp_flux(ple,source,ichoice):

    if (ichoice==1):
        source['Nu_max_sy'] = source['Po_max_sy'] = 0.
    elif (ichoice==2):
        source['Nu_max_ic'] = source['Po_max_ic'] = 0.
	
    for i in range (0, ple['numbins']):
        if (ple['f'][i]>0):
            #energy   = NU_eV*ple['e'][i]

            'get relativistic doppler factor'
            delta    = source['delta']

            'compute luminosity distance'
            distance = 4.*np.pi*(source['distance']**2)

            'power -> received flux'
            fac      = (delta**3)/distance

            'doppler boosting'
            if (ichoice!=3): 
                ple['nu'][i] = ple['e'][i] * delta
            
            'power per logarthmic energy interval  (erg / cm**s sec)'
            ple['f'][i]  = eV_erg/1E+4 * ple['nu'][i] *ple['f'][i] *fac
            
            'max SEDs'
            if ((ichoice==1) and (ple['f'][i]>source['Po_max_sy'])):
                source['Po_max_sy']  = ple['f'][i]
                source['Nu_max_sy']  = ple['nu'][i]
            if ((ichoice==2) and (ple['f'][i]>source['Po_max_ic'])):
                source['Po_max_ic']  = ple['f'][i]
                source['Nu_max_ic']  = ple['nu'][i]   
    return ple, source
    
"""
Sums all fluxes and stores them in psum dict.
"""
def sum_flux(source, psync, pic, pbb, pec, psum):

    for i in range(0, 3):
        if(i == 0):
            p = psync
        elif(i == 1):
            p = pic
        elif(i == 2):
            p = pbb
        elif(i == 3):
            p = pec
        else:
            print("Error in sum_flux\n")
            exit (-1)
		
        for j in range(0,psum['numbins']):
            flux = get_flux2(p,psum['nu'][j],i)
            psum['f'][j] += flux
    return psum

"""
Calculates Synchrotron self absorption

C++ has 5.67e8
Book uses 5.68e9
"""  
def ssa(psync,source):

    flag          = 1;
    source['od1'] = 1E+99;
    p        =  -source['pow1']
    kappa    = source['lum'] * (1E+9**source['pow1'])*1E+9;
	
    for i in range (0, psync['numbins']):
        if (psync['j'][i]>0):

            """
            Calcs synchrotron self absorption coefficient, Malcolm Longair, Edt. 3, pg. 224 */
            """	
            psync['a'][i] = kappa_nu = (20.9 * kappa * (source['B']**((p+2.)/2.)) * (5.67E+9**p) * b_factor(p) *(psync['e'][i]**(-(p+4.)/2.)))
			
            tau = 2. * kappa_nu * source['radius']
			
            psync['tau_i'][i] = tau
            """
            Calcs mean density of photons,
            Gould A&A, 76, 306*/
            """
            if (tau<1E-3):
                fac = source['radius']
            else:
                fac = (1.-np.exp(-tau/2.))/kappa_nu
			
            psync['n'][i] = 1./SI_c/(psync['e'][i]*NU_eV)*0.79*4.*np.pi*psync['j'][i]*fac
            """
            Emission power (eV Hz^-1 s^-1) */
            """
            if ((tau<1.) and (flag)):
                flag = 0
                source['od1'] = psync['e'][i] * source['delta']

            if (tau<1e-3):
                fac = 4./3.*source['radius']

            else:
                fac = (1.-2./(tau**2)*(1.-np.exp(-tau)*(tau+1.)))/kappa_nu
			
            psync['f'][i] = np.pi*(source['radius']**2)* 4.*np.pi*psync['j'][i] *fac
    return psync

"""
Calculate index of histogramm, giving adjacent bin-numbers num1, num2 and rest (value between 0 and 1 that gives the distance between num1 and num2)
"""    
def xnummer(xlow, xhigh, nbins, xvalue, num1, num2):

    xvalue = min(xvalue,xhigh)
    xvalue = max(xvalue,xlow)
    if    ( (xvalue>=xlow ) and (xvalue<xhigh) ):
        diff  = (xvalue-xlow)
        delta = (xhigh -xlow)/(nbins)
        num1 = (min(nbins-1,max(0,(diff/delta))))
    else: 
        if (xvalue==xhigh):
            num1 = nbins
        else:
            print("ERROR in nummer: %f %f %f\n",xlow,xhigh,xvalue);
	
    rest = xvalue - (xlow + ((num1)-.5)*delta)
    rest = rest/delta
    if (rest>0.):
        num2 = num1+1
    elif (rest<0.):
        num2 = num1-1
	
    if(num2<0):
        num2 = 0
    elif (num2>=nbins):
        num2 = nbins-1
    return abs(rest)

"""
Never used
"""
def least_squares(source, psync, pic, data):
    chi2 = 0
    dof = 0
    for i in range(0, data['numbins']):
        fsync = get_flux(psync, data['nu'][i]/source['delta'])
        fic = get_flux(pic, data['nu'][i]/source['delta'])
        f = max(fsync, fic)
        
        d     = data['f'][id]
        de    = data['fe'][id]
		
        if (de>0.):
            chi2 += ((f-d)/de)**2
            dof+=1
        else:

            print("ERROR in least_squares: %d %e %e\n"% i,d,de);
            exit(-1);
    source['chi2'] = chi2
    source['dof']  = dof 
    
"""
Gets flux with same method as get_density()
"""
def get_flux(ple, frequency):

    rest = 0.0
    xl   = np.log10(frequency)
	
    i1, rest   = bins(ple,xl,rest)
	
    if (i1>0): 
        i2   = i1+1
		
        w1   = ple['f'][i1]  
        w2   = ple['f'][i2]
		
        flux = w1*(1.-rest)+w2*rest 
		
        
    else:
        flux = 0.
    return flux

         
def get_flux2(ple, nu, ichoice):
    
    if (nu < ple['nu'][0] or (nu > ple['nu'][ple['numbins']-1])):
        return 0 

    for i1 in range(0, ple['numbins']-1):
        count1 = i1

        if(ple['nu'][i1+1] > nu):
            break

    xl = math.log(nu)
    count2 = count1+1
     
    if((ple['f'][count1] < 1E-30) and (ple['f'][count2] < 1E-30)):
        flux = 0

    x1 = np.log(ple['nu'][count1])
    x2 = np.log(ple['nu'][count2])
    rest = (xl-x1)/(x2-x1)
     
    flux = np.log(ple['f'][count1])*(1-rest)+np.log(ple['f'][count2])*rest
    flux = math.exp(flux)
    
    if((ichoice == 1) or (ichoice == 3)):
        tau = (ple['tau_i'][count1]+ple['tau_e'][0][count1])*(1.-rest)+(ple['tau_i'][count2]+ple['tau_e'][0][count2])*rest
        flux*=math.exp(-tau)

    return flux            
    
"""
Finds flux density
"""
def get_density(ple, frequency):
    
    i1 = 0.0
    i2 = 0.0
    rest = 0.0
    w1, w2, flux = 0.0,0.0,0.0

    xl = np.log10(frequency)
    i1, rest = bins(ple, xl, rest)

    if (i1>0) and (i1<ple['numbins']-1):
        
        i2 = i1+1
        w1 = ple['n'][i1]
        w2 = ple['n'][int(i2)]

        flux = w1*(1-rest)+w2*rest

    else:
        flux = 0
    return flux

"""
Ned Wright's cosmology calculator. arXiv 0609593 - "A Cosmology Calculator for the World Wide Web"
Can calculate various cosmological values. Implimented here to calculate Luminosity Distance.
"""
def NedWright(z, h0, Omega_M, Omega_Lambda, age_Gyr, zage_Gyr, DTT_Gyr, DA_Mpc, kpc_DA, DL_Mpc, DL_Gyr):

    n=1000	#number of integral points 
    c = 299792.458 #c in km/sec
    Tyr = 977.8 #1/H to Gyr

    H0 = h0*100	#Hubble constant
    
    'Densities'
    WM = Omega_M  # Omega(matter)
    WV = Omega_Lambda # Omega(lambda)
    h = H0/100    
    WR = 4.165E-5/(h*h)    #  Omega(radiation), includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV  #  Omega curvaturve = 1-Omega(total)
	
    a = 1.0 #	scale factor 
    az = 1.0/(1+1.0*z)
	
    age = 0;
    for i in range(0, n):
        a = az*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 1/adot
        
    zage = az*age/n
    """
    Correction for annihilations of particles not present now like e+/e-
    """
    lpz = np.log((1+1.0*z))/np.log(10.0)
    dzage = 0
    if (lpz >  7.500): 
        dzage = 0.002 * (lpz -  7.500)
    if (lpz >  8.000): 
        dzage = 0.014 * (lpz -  8.000) +  0.001 
    if (lpz >  8.500): 
        dzage = 0.040 * (lpz -  8.500) +  0.008
    if (lpz >  9.000): 
        dzage = 0.020 * (lpz -  9.000) +  0.028
    if (lpz >  9.500): 
        dzage = 0.019 * (lpz -  9.500) +  0.039
    if (lpz > 10.000): 
        dzage = 0.048
    if (lpz > 10.775): 
        dzage = 0.035 * (lpz - 10.775) +  0.048
    if (lpz > 11.851): 
        dzage = 0.069 * (lpz - 11.851) +  0.086
    if (lpz > 12.258): 
        dzage = 0.461 * (lpz - 12.258) +  0.114
    if (lpz > 12.382): 
        dzage = 0.024 * (lpz - 12.382) +  0.171
    if (lpz > 13.055): 
        dzage = 0.013 * (lpz - 13.055) +  0.188
    if (lpz > 14.081): 
        dzage = 0.013 * (lpz - 14.081) +  0.201
    if (lpz > 15.107): 
        dzage = 0.214
    zage = zage*pow(10.0,dzage);
    zage_Gyr = (Tyr/H0)*zage;
	
    DTT = 0.0;
    DCMR = 0.0;

    'integral over a=1/(1+z) from az to 1 in n steps using midpoint rule'
    for i in range(0,n):
        a = az+(1.-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)

	
    DTT = (1-az)*DTT/n
    DCMR = (1-az)*DCMR/n
	
    age = DTT+zage
    age_Gyr = age*(Tyr/H0)
	
	
    DTT_Gyr = (Tyr/H0)*DTT
	
	
    
	
    'tangential comoving distance'
    ratio = 1.00
    
    x = np.sqrt(abs(WK))*DCMR
    if (x > 0.1): 
        if (WK>0): 
            ratio = 0.5*(np.exp(x)-np.exp(-x))/x
        else: 
            ratio = np.sin(x)/x
        y = ratio*DCMR
    
    else:
        y = x*x;
        if (WK < 0):
            y = -y

        ratio = 1 + y/6 + y*y/120
        y= ratio*DCMR

    DCMT = y
	
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
	
    print("Distance parameters: age_Gyr %e zage_Gyr %e DTT_Gyr %e DA_Mpc %e kpc_DA %e DL_Mpc %e \n" % (age_Gyr,zage_Gyr,DTT_Gyr,DA_Mpc,kpc_DA,DL_Mpc))
	
    return DL_Mpc

"""
No longer use this version.
From NedWright's website. But causes overflow errors from different outputted DL_Mpc value
"""

def NedWrightOLD(z, h0, Omega_M, Omega_Lambda):#, age_Gyr, zage_Gyr, DTT_Gyr, DA_Mpc, kpc_DA, DL_Mpc, DL_Gyr):
    ' // Reference: Wright, E. L., 2006, PASP, 118, 1711.'
    try:
      z=float(z)    # redshift
      H0 = float(h0) # Hubble constant
      WM = float(Omega_M) # Omega(matter)
      WV = float(Omega_Lambda) # Omega(vacuum) or lambda
      WR = 0.        # Omega(radiation)
      WK = 0.        # Omega curvaturve = 1-Omega(total)
      c = 299792.458 # velocity of light in km/sec
      Tyr = 977.8    # coefficent for converting 1/H into Gyr
      DTT = 0.5      # time from z to now in units of 1/H0
      DTT_Gyr = 0.0  # value of DTT in Gyr
      age = 0.5      # age of Universe in units of 1/H0
      age_Gyr = 0.0  # value of age in Gyr
      zage = 0.1     # age of Universe at redshift z in units of 1/H0
      zage_Gyr = 0.0 # value of zage in Gyr
      DCMR = 0.0     # comoving radial distance in units of c/H0
      DCMR_Mpc = 0.0 
      DCMR_Gyr = 0.0
      DA = 0.0       # angular size distance
      DA_Mpc = 0.0
      DA_Gyr = 0.0
      kpc_DA = 0.0
      DL = 0.0       # luminosity distance
      DL_Mpc = 0.0
      DL_Gyr = 0.0   # DL in units of billions of light years
      V_Gpc = 0.0
      a = 1.0        # 1/(1+z), the scale factor of the Universe
      az = 0.5       # 1/(1+z(object))
    
      h = H0/100.
      WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
      WK = 1-WM-WR-WV
      az = 1.0/(1+1.0*z)
      age = 0.
      n=1000         # number of points in integrals
      for i in range(n):
        a = az*(i+0.5)/n
        adot = math.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 1./adot
    
      zage = az*age/n
      zage_Gyr = (Tyr/H0)*zage
      DTT = 0.0
      DCMR = 0.0
    
    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
      for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = math.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)
    
      DTT = (1.-az)*DTT/n
      DCMR = (1.-az)*DCMR/n
      age = DTT+zage
      age_Gyr = age*(Tyr/H0)
      DTT_Gyr = (Tyr/H0)*DTT
      DCMR_Gyr = (Tyr/H0)*DCMR
      DCMR_Mpc = (c/H0)*DCMR
    
    # tangential comoving distance
    
      ratio = 1.00
      x = math.sqrt(abs(WK))*DCMR
      if x > 0.1:
        if WK > 0:
          ratio =  0.5*(math.exp(x)-math.exp(-x))/x 
        else:
          ratio = math.sin(x)/x
      else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.
      DCMT = ratio*DCMR
      DA = az*DCMT
      DA_Mpc = (c/H0)*DA
      kpc_DA = DA_Mpc/206.264806
      DA_Gyr = (Tyr/H0)*DA
      DL = DA/(az*az)
      DL_Mpc = (c/H0)*DL
      DL_Gyr = (Tyr/H0)*DL
    
    # comoving volume computation
    
      ratio = 1.00
      x = math.sqrt(abs(WK))*DCMR
      if x > 0.1:
        if WK > 0:
          ratio = (0.125*(math.exp(2.*x)-math.exp(-2.*x))-x/2.)/(x*x*x/3.)
        else:
          ratio = (x/2. - math.sin(2.*x)/4.)/(x*x*x/3.)
      else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/5. + (2./105.)*y*y
      VCM = ratio*DCMR*DCMR*DCMR/3.
      V_Gpc = 4.*math.pi*((0.001*c/H0)**3)*VCM
    
      #if verbose == 1:
      print ('For H_o = ' + '%1.1f' % H0 + ', Omega_M = ' + '%1.2f' % WM + ', Omega_vac = ',)
      print ('%1.2f' % WV + ', z = ' + '%1.3f' % z)
      print ('It is now ' + '%1.1f' % age_Gyr + ' Gyr since the Big Bang.')
      print ('The age at redshift z was ' + '%1.1f' % zage_Gyr + ' Gyr.')
      print ('The light travel time was ' + '%1.1f' % DTT_Gyr + ' Gyr.')
      print ('The comoving radial distance, which goes into Hubbles law, is',)
      print ('%1.1f' % DCMR_Mpc + ' Mpc or ' + '%1.1f' % DCMR_Gyr + ' Gly.')
      print ('The comoving volume within redshift z is ' + '%1.1f' % V_Gpc + ' Gpc^3.')
      print ('The angular size distance D_A is ' + '%1.1f' % DA_Mpc + ' Mpc or',)
      print ('%1.1f' % DA_Gyr + ' Gly.')
      print ('This gives a scale of ' + '%.2f' % kpc_DA + ' kpc/".')
      print ('The luminosity distance D_L is ' + '%1.1f' % DL_Mpc + ' Mpc or ' + '%1.1f' % DL_Gyr + ' Gly.')
      print ('The distance modulus, m-M, is '+'%1.2f' % (5*np.log10(DL_Mpc*1e6)-5))
      #else:
      print ('%1.2f' % zage_Gyr,)
      print ('%1.2f' % DCMR_Mpc,)
      print ('%1.2f' % kpc_DA,)
      print ('%1.2f' % (5*np.log10(DL_Mpc*1e6)-5))
    
      return (age_Gyr, zage_Gyr, DTT_Gyr, DA_Mpc, kpc_DA, DL_Mpc, DL_Gyr)
    
    except IndexError:
      print ('need some values or too many values')
    except ValueError:
      print ('nonsense value or option')
    #return (age_Gyr, zage_Gyr, DTT_Gyr, DA_Mpc, kpc_DA, DL_Mpc, DL_Gyr)
"""
Returns the bin index of an inputted value.

Not sure about how this handles rest variable. Look into it 
"""
def bins(p, xl, rest):
    ibin = 0
    ibin = int((xl-p['x1'])/p['delta'])
            
    if (ibin > (p['numbins']-1)):
        ibin = -1

    elif(ibin<0):
        ibin = -1

    rest = 0.5+(xl - xlog(p, ibin))/p['delta']
    return ibin, rest

"""
Returns the center value of a given float bin index
"""
def fxen(ple, i):
    val = 0.0
    val = float(ple['x1'] + (i + 0.5) * ple['delta'])
    return 10**val
    
"""
Returns the center value of a given float bin index

Make sure that when it calls x1 it is only x1 from the function where it is called
"""
def xen(ple,i):
    val = 0.0 
    val = ple['x1'] + (float(i) + 0.5) * ple['delta']
    val = float(val)    
    return float(10**val)

"""
Returns the center value of a given integer bin index
ln
"""
def xl(ple, i):
    val = 0.0 
    val = ple['x1'] + (i + 0.5) * ple['delta']
    return val*LOG10
"""
Returns the center value of a given integer bin index
log10
"""
def xlog(ple, i):
    val = 0.0 
    val = ple['x1'] + (i + 0.5) * ple['delta']
    return val
   
"""
Fills spectrum with broken power law.
arg 1: Spectrum to be filled 
arg 2: Min electron energy 
arg 3: Max electron energy 
arg 4: Break of power law 
arg 5: Luminosity
arg 6: Spectral index of electron spec (Emin -> Ebreak) 
arg 7: Spectral index of electron spec (Ebreak -> Emax)
"""
def fill_le(ple,x1,x2,xb,x0,xindex1,xindex2):
    
    rest = 0.0
    i1, rest= bins(ple,x1,rest)
    i2, rest= bins(ple,xb,rest)
    if (rest>0.5): 
        i2+=1
    i3, rest = bins(ple,x2,rest)

    """
    Broken power law
    """
    
    for i in range (int(i1), int(i2)):
        
        x = ple['e'][i]
        ple['f'][i] = x0* (x**int(xindex1))
    
    helpv = (10.**xb)
    norm = x0* (helpv**xindex1)/(helpv**xindex2)
    
    for i in range(int(i2), int(i3)+1):
        x = ple['e'][i]
        ple['f'][i] = norm* (x**xindex2)
    return ple

"""
Never used
"""
def fill_le2(ple, x1,x2,xb,x0,xindex1):	
    rest = 0
    i1, rest = bins(ple,x1,rest)
    i3, rest = bins(ple,x2,rest)
	
    '/* POWER LAW WITH EXPONENTIAL CUT-OFF */'
    for i in range(i1,i3):

        x         = ple['e'][i];
        ple['f'][i] = x0* (x**xindex1) *math.exp(-x/(10.**xb))
    print("************FILL LE2**********")
    print(ple['f'])
    return ple
import time
import argparse

parser = argparse.ArgumentParser(description='Synchrotron Self-Compton model \n \n \n \n Usage:\n\"spectrum [z=0.034] [Gamma=50] [theta=3] [B=0.025E-4] [radius=1E+12] [w_p_soll=10.] [log Emin=10.6] [log Emax=13.5] [log Ebreak=12.5] [p1=2] [p2=3] [ECflag=1] [Mbh=1E+9] [Mdot=1.] [blobDist=100] [EBLModel = Franceschini] \n\n\n')
parser.add_argument('-z',  default= 0.048, help="Source's redshift",type=float)
parser.add_argument('-g',  default= 10.23 ,help='Bulk Lorentz factor',type=float)
parser.add_argument('-the',  default= 0.8 ,help='Jet angle to observer (Degrees)',type=float)
parser.add_argument('-B',  default= 0.04e-4,help='Magnetic field strenght',type=float)
parser.add_argument('-r',  default= 5.8e13 ,help='radius of the emission volume in meters',type=float)
parser.add_argument('-wp',  default= 0.22 ,help=' jet-energy density of the electrons in erg/cm^3',type=float)
parser.add_argument('-min',  default= 3.5 ,help='Minimum electron energy ',type=float)
parser.add_argument('-max', default= 12.3 ,help='Maximum electron energy ',type=float)
parser.add_argument('-ebreak',  default=11.8 ,help='Energy power law break point',type=float)
parser.add_argument('-p1',  default=2 ,help=' differential spectral index of the electron spectrum Emin to Ebreak',type=float)
parser.add_argument('-p2', default=3 ,help=' differential spectral index of the electron spectrum Ebreak to Emax',type=float)
parser.add_argument('-ec', default=0 ,help='Include external compton? 1=Yes, 0=No')
parser.add_argument('-mbh', default=3.8 ,help='Mass of black hole',type=float)
parser.add_argument('-mdot', default=4 ,help='Mass accretion rate of blackhole',type=float)
parser.add_argument('-dis', default=600 ,help='height of the emitting volume above the accretion disk in Schwarzchild radii',type=float)
parser.add_argument('-ebl', default='Franceschini' ,help='Choice of EBL model, Franceschini, Finke or Dominguez',type=str)
args = parser.parse_args()

start_time = time.time()

main1(float(args.z), float(args.g), float(args.the), float(args.B), float(args.r), float(args.wp), float(args.min), float(args.max), float(args.ebreak), float(args.p1), float(args.p2), float(args.ec), float(args.mbh), float(args.mdot), float(args.dis), str(args.ebl))

print(time.time() - start_time)
 
