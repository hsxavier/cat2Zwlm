#!/usr/bin/env python

"""
This script takes a catalogue <CAT> with columns ra, dec (given in degrees) and r 
(given in h^-1Mpc); a radial selection function <NR> (table with columns r and n(r) 
given in # of galaxies per arcmin^2 per h^-1Mpc); an angular selection 
function <ANG> (Healpix map in fits format); the radial mode coverage limits in 
h^-1Mpc <RMIN> and <RMAX>; the maximum radial mode index <QMAX>; and the maximum 
angular mode <LMAX>; and outputs a table <OUT> containing the columns q, l, m which 
identify a logarithmic spherical wave mode Zwlm and the coefficient for the 
corresponding mode.

USAGE:   cat2Zwlm.py <CAT> <NR> <ANG> <RMIN> <RMAX> <QMAX> <LMAX> <OUT>
EXAMPLE: cat2Zwlm.py cmass-cat.dat nz.dat completeness.fits 100 5000 10 40 cmass-Zwlm.dat

Written by Henrique S. Xavier, hsxavier@if.usp.br, on 16/mar/2017. 
"""

import sys
import numpy as np
import healpy as hp
#from scipy.special import sph_harm as Y  # Y(m,l,phi,theta)
import ctypes as ct



############################
### Function definitions ###
############################

# Load C function:
lib = ct.cdll.LoadLibrary('/home/skems/cicSDSS/prog/cat2Zwlm/multi-Ylm.so')
CalcYlm = lib.multiYlm

# Definition of u:
def r2u(r):
    return np.log(r)

# w-independent factor of Zeta_w(r), the radial part of the logarithmic spherical waves:
def ZetaFactor(r):
    return 1.0/np.sqrt(2*np.pi)/r**1.5

# Zeta_w*(r), the conjugate of the radial part of the logarithmic spherical waves:
def ZetaConj(Zfac, w, u):
    return Zfac*np.exp(1.0j*w*u)

# The logarithmic spherical wave modes:
def Zeta(w, r):
    return np.conj( ZetaConj(ZetaFactor(r), w, r2u(r)) ) 



#########################
### Beginning of code ###
#########################

# Docstring output:
if len(sys.argv) != 1 + 8: 
    print(__doc__)
    sys.exit(1)

# Load input:
catfile = sys.argv[1]
nrfile  = sys.argv[2]
angfile = sys.argv[3]
rmin    = float(sys.argv[4])
rmax    = float(sys.argv[5])
qmax    = int(sys.argv[6])
lmax    = int(sys.argv[7])
outfile = sys.argv[8]
# Derive other quantities:
Nmultipoles = (lmax+1)*(lmax+2)/2

# Load and prepare catalogue:
print "Loading catalog..."
ra, dec, r = np.loadtxt(catfile,usecols=(0,1,2),unpack=True)
print "Preparing catalog data..."
Ngals = len(r)
theta = (90.0-dec)/180.0*np.pi
phi   = ra/180.0*np.pi
u     = r2u(r)
zfac  = ZetaFactor(r)

# Load selection function data:
print "Loading radial selection function..."
RadSelR, RadSelN = np.loadtxt(nrfile, unpack=True)
print "Getting radial selection function for each galaxy..."
RadSelByGal = np.interp(r, RadSelR,RadSelN)
print "Loading angular selection function..."
completeness = hp.read_map(angfile, verbose=False)
Nside = int(np.sqrt(len(completeness)/12))
print "Getting completeness for each galaxy..."
ComplByGal = completeness[hp.ang2pix(Nside,theta,phi)] 

# Compute part of the modes that does not depend on l,m:
print "Computing (l,m)-independent part of the modes..."
NoLMpart = []
wList    = 2*np.pi*np.arange(0,qmax+1)/(r2u(rmax)-r2u(rmin))
rFactor  = 1.0/(RadSelByGal*ComplByGal)
rFactor[ComplByGal==0] = 0 # Ignores galaxies in masked regions (this is an angular coord. truncation problem).
for w in wList:    
    NoLMpart.append(ZetaConj(zfac, w, u)*rFactor)  # Checked by summing over galaxies and comparing with Mathematica.


# Compute spherical harmonics Ylm part:
print "Computing spherical harmonic part of the modes..."
ReYlm = np.zeros(Ngals*Nmultipoles,dtype=np.double)
ImYlm = np.zeros(Ngals*Nmultipoles,dtype=np.double)
CalcYlm(ct.c_int(lmax), ct.c_void_p(theta.ctypes.data), ct.c_void_p(phi.ctypes.data), \
        ct.c_int(Ngals), ct.c_void_p(ReYlm.ctypes.data), ct.c_void_p(ImYlm.ctypes.data))
# Rearrange Ylm's by multipole and galaxy:
Ylm = np.reshape(ReYlm + 1.0j*ImYlm, (Nmultipoles, Ngals))

# Compute modes:
print "Computing full modes..."
Nqlm = []
# Compute q<0 modes:
for q in range (-qmax,0):
    fZq = np.conj(NoLMpart[-q])
    for l in range(0,lmax+1):
        for m in range (-l,0):
            Nqlm1   = (-1)**-m * np.sum(fZq*np.conj(Ylm[l*(l+1)/2+m]))
            Nqlm.append( [q,l,m, Nqlm1.real, Nqlm1.imag] )
        for m in range(0,l+1):
            Nqlm1   = np.sum(fZq*Ylm[l*(l+1)/2+m])
            Nqlm.append( [q,l,m, Nqlm1.real, Nqlm1.imag] )
# Compute q>=0 modes:
for q in range (0,qmax+1):
    fZq = NoLMpart[q]
    for l in range(0,lmax+1):
        for m in range (-l,0):
            Nqlm1   = (-1)**-m * np.sum(fZq*np.conj(Ylm[l*(l+1)/2+m]))
            Nqlm.append( [q,l,m, Nqlm1.real, Nqlm1.imag] )
        for m in range(0,l+1):
            Nqlm1   = np.sum(fZq*Ylm[l*(l+1)/2+m])
            Nqlm.append( [q,l,m, Nqlm1.real, Nqlm1.imag] )
            

# Save result:
print "Writing mode coefficients to file..."
np.savetxt(outfile, Nqlm, fmt=['%4d', '%4d', '%4d', '%14.6e', '%14.6e'], header=' q    l    m     Nqlm[Re]       Nqlm[Im]')


