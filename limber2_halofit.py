#!/usr/bin/env python3
#
# Python code to compute the Limber integral to convert
# P(k) -> C_l.
#

import numpy as np
import astropy.cosmology
from   scipy.interpolate import InterpolatedUnivariateSpline as Spline
import glob
import sys
import re
import time
import os
from multiprocessing import Pool

# Set up CLEFT object


class PowerSpectrum():
	"""A class which returns P(k) from CAMB linear P(k)"""
	# The main purpose of this class is to smoothly handle the
	# interpolation in z.  If we have a small-ish number of points
	# Lagrange interpolating polynomials are ideal for vectorized ops.
	# Currently P(k) is computed from linear P(k) from CAMB.
	def lagrange_spam(self,z):
		"""Returns the weights to apply to each z-slice to interpolate to z."""
		dz = self.zlist[:,None] - self.zlist[None,:]
		singular = (dz == 0)
		dz[singular] = 1.0
		fac = (z - self.zlist) / dz
		fac[singular] = 1.0
		return(fac.prod(axis=-1))
	def __init__(self,k,Pk_interpolator,halofit,zlist):
		"""Makes the P(k) tables for some k points,
		a Pk_interpolator object that calls the linear power spectrum,
		a list of redshifts. one_loop and shear tell you whether
		to include 1-loop and shear terms in the LPT expansion.
		Also adds a halofit object for the magnification term;
		halofit is a P(k) interpolator object with nonlinear=True."""
		self.halofit = halofit


	def __call__(self,kval,zz,b1_HF,b2_HF,one_loop=True,shear=True):
		"""Returns power spectra at k=kval and z=zz.
		pars is a dict telling you which method to use (halozeldovich, zeft, halofit)
		and the bias/1-halo parameters for 2 tracers.  
		ex: pars = {'method': 'halofit', 'tracer1': {'b': lambda z: 1.5},
		'tracer2': {'b': lambda z: 2.0}}.
		Note that biases must be functions of redshift.
		This function then returns all 4 spectra needed for CMB lensing:
		tracer1 x tracer2 , tracer1 x matter, tracer2 x matter, and matter x matter.
		If you want auto-spectra, just use the same bias coefficients for tracer1 and tracer2."""
		# Get the interpolating coefficients then just weigh each P(k,z).
		t0 = time.time()
		
		p_mm = np.zeros((1,len(kval)))
		p_gm_HF = np.zeros((1,len(kval)))
		p_gg_HF = np.zeros((1,len(kval)))
		#print('coef',coef)

		#print('pmm',p_mm)
		#print('pgm1_HF',p_gm1_HF)
		p_mm[0,:] = self.halofit(zz, kval)
		p_gm_HF[0,:] = self.halofit(zz, kval)
		p_gg_HF[0,:] = self.halofit(zz, kval)
				
		#print('kval',np.shape(kval))
		#print('kk',np.shape(kk))

		return(p_gg_HF, p_gm_HF, p_mm)

def mag_bias_kernel(cosmo, dndz, s, zatchi, chival, chivalp, zvalp, Nchi_mag=1000):
	"""Returns magnification bias kernel as a function of chival/zval. Arguments
	'cosmo' is an astropy Cosmology instance,
	'dndz' is the redshift distribution,
	's' is the slope of the number counts dlog10N/dm,
	'chival' is the array of comoving distances for which the magnification
	bias kernel is defined (and 'zval' is the corresponding array of redshifts),
	'chistar' is the comoving distance to the surface of last scattering,
	'zatchi' is a function taking comoving distance as input and returning redshift,
	and 'Nchi_mag' gives the number of sampling points in the integral over chiprime.
	"""
	dndz_norm = np.interp(zvalp,\
						dndz[:,0],dndz[:,1],left=0,right=0)
	norm = np.trapz( dndz[:,1], x = dndz[:,0])
	dndz_norm  = dndz_norm/norm
	Om0 = cosmo.Om0
	g = chival * np.trapz( 1. / (chivalp) * (chivalp - chival[np.newaxis,:]) * dndz_norm * (1.0/2997.925) * np.sqrt(Om0 * (1+zvalp)**3. + 1-Om0), x =  chivalp, axis=0)
	mag_kern = 1.5 * (cosmo.Om0) * (1.0/2997.925)**2 * (1+zatchi(chival)) * g * (5. * s - 2.)
	return mag_kern

def setup_chi(cosmo, dndz1, dndz2, Nchi, Nchi_mag):
	hub      = cosmo.H0.value / 100.
	zmin     = np.min(np.append(dndz1[:,0],dndz2[:,0]))
	zmax     = np.max(np.append(dndz1[:,0],dndz2[:,0]))
	zval     = np.linspace(zmin,zmax,1000) # Accuracy doesn't depend much on the number of zbins
	chival   = cosmo.comoving_distance(zval).value*hub	# In Mpc/h.
	zatchi   = Spline(chival,zval)
	# Spline comoving distance as well to make it faster
	chiatz   = Spline(zval, chival)
	# Work out W(chi) for the objects whose dNdz is supplied.
	chimin   = np.min(chival) + 1e-5
	chimax   = np.max(chival)
	chival   = np.linspace(chimin,chimax,Nchi)
	zval     = zatchi(chival)
	chistar  = cosmo.comoving_distance(1098.).value * hub
	chivalp = np.array(list(map(lambda x: np.linspace(x,chistar,Nchi_mag),chival))).transpose()
	zvalp = zatchi(chivalp)
	#print('chival',chival)
	return zatchi, chiatz, chival, zval, chivalp, zvalp
	
def do_limber(ell, cosmo, dndz1, dndz2, s1, s2, pk, b1_HF, b2_HF, alpha_auto, alpha_cross, use_zeff=True, autoCMB=False, Nchi=50, dndz1_mag=None, dndz2_mag=None, normed=False, setup_chi_flag=False, setup_chi_out=None):
	"""Does the Limber integral, returning Cell.
	Returns two lists, [auto, cross], each with a number of different terms that you might want.
	The contents of auto are: [C_{ell}^{gg}, C_{ell}^{gg; alpha}, C_{ell}^{gmu}, C_{ell}^{mu mu}]
	defined as C_{ell}^{gg} = \int d\chi W1_g W2_g / chi^2 P_mm(k \chi = \ell + 0.5)
	where P_mm is from Halofit.
	The k^2-dependent bias term:
	C_{ell}^{gg; alpha} = \int d\chi W1_g W2_g / chi^2 -0.5 * \alpha_auto(z) k^2 P_mm(k \chi = \ell + 0.5)
	where \alpha_auto(z) is specified by the argument alpha_auto.
	Note that the galaxy window functions W1 and W2 can generically be *different* between 
	C_{ell}^{gg} and C_{ell}^{gg; alpha}, since the relevant window for C_{ell}^{gg} is b(z) * dN/dz
	whereas it is dN/dz for C_{ell}^{gg; alpha}.
	C_{ell}^{gmu} and C_{ell}^{mu mu} are magnification bias terms.  Again the code supports
	different dN/dz for the magnification bias kernel and for the galaxy window (which has b(z) also included).
	The power spectrum in the magnification bias kernel is a fixed linear bias * Halofit.
	While there may also be higher-order terms for this kernel (i.e. b2, shear bias, k^2 dependent bias),
	these are not included in this code as they are assumed to be a second-order small quantity,
	since the magnification bias term is already small compared to the clustering term.
	
	The contents of cross are: [C_{ell}^{\kappa g}, C_{ell}^{\kappa g; alpha}, C_{ell}^{\kappa mu}
	C_{ell}^{\kappa g} = \int d\chi W1_g WCMB_g / chi^2 P_mm(k \chi = \ell + 0.5)
	C_{ell}^{\kappa g; alpha} = \int d\chi W1_g WCMB_g / chi^2 -0.5 * \alpha_cross(z) k^2 P_mm(k \chi = \ell + 0.5)
	and C_{ell}^{\kappa mu} the cross-correlation between magnification bias and CMB lensing.

	'ell' is an array of ells for which you want Cell.
	'cosmo' is an astropy Cosmology instance,
	'dndz1' contains two columns (z and dN/dz for sample 1),
	'dndz2' contains two columns (z and dN/dz for sample 2),
	's1' and 's2' are the slope of number counts dlog10N/dm for the 2 samples
	(for magnification bias),
	'pk' is a power spectrum instance (k and P, in Mpc/h units).
	'b1_HF' and 'b2_HF' are the fixed linear biases for the magnification terms.
	'alpha_auto' and 'alpha_cross' give the coefficient for the k^2-dependent bias term as a function of redshift.
		
	If use_zeff==True then P(k,z) is assumed to be P(k,zeff).
	if autoCMB=True, returns CMB auto power spectrum.
	Nchi is the number of integration points to use.

	'dndz1_mag' and 'dndz2_mag' are optional additional dn/dz to use in the magnification
	bias term (if you want to use a different dn/dz here and in the clustering term).

	If normed=True, the code assumes that your redshift distribution is already normalized.
	setup_chi_flag and setup_chi_out allow you to pass a pre-defined chi-grid, saving the evaluation time of this step."""
	t0 = time.time()
	hub      = cosmo.H0.value / 100.
	Nchi_mag = 25 # Number of sampling points in magnification bias integral
	if setup_chi_flag == False:	
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi(cosmo, dndz1, dndz2, Nchi, Nchi_mag)
	else:
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi_out

	# Galaxy kernels, assumed to be b(z) * dN/dz
	fchi1    = np.interp(zatchi(chival),\
		 dndz1[:,0],dndz1[:,1]*cosmo.H(dndz1[:,0]).value,left=0,right=0)
	if not normed:
		fchi1   /= np.trapz(fchi1,x=chival)
	fchi2    = np.interp(zatchi(chival),\
		 dndz2[:,0],dndz2[:,1]*cosmo.H(dndz2[:,0]).value,left=0,right=0)
	if not normed:
		fchi2   /= np.trapz(fchi2,x=chival)
	
	# Galaxy kernels for k^2 dependent bias term, assumed to be dN/dz
	fchi1_xmatch    = np.interp(zatchi(chival),\
		 dndz1_mag[:,0],dndz1_mag[:,1]*cosmo.H(dndz1_mag[:,0]).value,left=0,right=0)
	if not normed:
		fchi1_xmatch   /= np.trapz(fchi1_xmatch,x=chival)
	fchi2_xmatch    = np.interp(zatchi(chival),\
		 dndz2_mag[:,0],dndz2_mag[:,1]*cosmo.H(dndz2_mag[:,0]).value,left=0,right=0)
	if not normed:
		fchi2_xmatch   /= np.trapz(fchi2_xmatch,x=chival)
		
	# Magnification kernels
	if dndz1_mag is None:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1_mag, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	if dndz2_mag is None:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2_mag, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	
	chistar  = cosmo.comoving_distance(1098.).value * hub

	fchiCMB    = 1.5* (cosmo.Om0) *(1.0/2997.925)**2*(1+zatchi(chival))
	fchiCMB   *= chival*(chistar-chival)/chistar
	if autoCMB:
		fchi2    = 1.5* (cosmo.Om0) *(1.0/2997.925)**2*(1+zatchi(chival))
		fchi2   *= chival*(chistar-chival)/chistar
		fchi1 = fchi2

	# Get effective redshift     
	if use_zeff:
		kern = fchi1*fchi2/chival**2
		zeff = np.trapz(kern*zval,x=chival)/np.trapz(kern,x=chival)
	else:
		zeff = -1.0
		

	Nell = len(ell)
	cell_f1f2 = np.zeros( (Nell, 1, Nchi) )
	cell_f1f2_alpha_auto = np.zeros( (Nell, 1, Nchi) )
	cell_f1fCMB_alpha_cross = np.zeros( (Nell, 1, Nchi) )
	cell_f1m2 = np.zeros( (Nell, 1, Nchi) )
	cell_m1m2 = np.zeros( (Nell, 1, Nchi) )
	
	cell_f1fCMB = np.zeros( (Nell, 1, Nchi) )
	cell_m1fCMB = np.zeros( (Nell, 1, Nchi) )
	
		
	#print('before integral',time.time()-t0)
	for i,chi in enumerate(chival):
		if (fchi2[i] != 0) | (mag_kern2[i] != 0):
			kval = (ell+0.5)/chi
						
			if use_zeff:
				# Assume a fixed P(k,zeff).
				# This should return both pofk for the galaxy [with the bias]
				# and pofk for matter [no bias, but would include shotnoise] 
				
				p_gg, p_gm, p_mm = pk(kval, zeff, b1_HF, b2_HF)
			else:
				# Here we interpolate in z.
				zv   = zatchi(chi)
				p_gg, p_gm, p_mm = pk(kval, zv, b1_HF, b2_HF)
				

			f1f2 = fchi1[i]*fchi2[i]/chi**2 * p_gg
			f1f2_alpha_auto =-0.5 * alpha_auto(zv) * fchi1_xmatch[i]*fchi2_xmatch[i]/chi**2 * kval**2 * p_mm
			f1fCMB_alpha_cross =-0.5 * alpha_cross(zv) * fchi1_xmatch[i]*fchiCMB[i]/chi**2 * kval**2 * p_mm
			f1m2 = 	fchi1[i]*mag_kern2[i]/chi**2 * p_gm
			m1m2 = mag_kern1[i]*mag_kern2[i]/chi**2 * p_mm

			cell_f1f2[:,:,i] = f1f2.transpose()
			cell_f1f2_alpha_auto[:,:,i] = f1f2_alpha_auto.transpose()
			cell_f1fCMB_alpha_cross[:,:,i] = f1fCMB_alpha_cross.transpose()
			cell_f1m2[:,:,i] = f1m2.transpose()
			cell_m1m2[:,:,i] = m1m2.transpose()

			f1fCMB = fchi1[i]*fchiCMB[i]/chi**2 * p_gm
			m1fCMB = mag_kern1[i]*fchiCMB[i]/chi**2 * p_mm
						
			cell_f1fCMB[:,:,i] = f1fCMB.transpose()
			cell_m1fCMB[:,:,i] = m1fCMB.transpose()
			

	cell_f1f2 = np.trapz(cell_f1f2,x=chival,axis=-1)
	cell_f1f2_alpha_auto = np.trapz(cell_f1f2_alpha_auto,x=chival,axis=-1)
	cell_f1fCMB_alpha_cross = np.trapz(cell_f1fCMB_alpha_cross,x=chival,axis=-1)
	cell_f1m2 = np.trapz(cell_f1m2,x=chival,axis=-1)
	cell_m1m2 = np.trapz(cell_m1m2,x=chival,axis=-1)
	cell_m1fCMB = np.trapz(cell_m1fCMB,x=chival,axis=-1)
	cell_f1fCMB = np.trapz(cell_f1fCMB,x=chival,axis=-1)

	return( [cell_f1f2, cell_f1f2_alpha_auto, cell_f1m2, cell_m1m2] , [cell_f1fCMB, cell_f1fCMB_alpha_cross, cell_m1fCMB])
	#






if __name__=="__main__":
    if len(sys.argv)!=2:
        raise RuntimeError("Usage: {:s} ".format(sys.argv[0])+\
                           "<dndz-fname>")
    else:
        # Assume these are ascii text files with two columns each:
        # z,dNdz for the first and k,P(k) for the second.  The best
        # interface to be determined later.
        dndz1= np.loadtxt(sys.argv[1])
        dndz2= np.loadtxt(sys.argv[1])
        pk   = PowerSpectrum()
        l,Cl = do_limber(astropy.cosmology.Planck15,dndz1,dndz2,pk,\
                         auto=False,use_zeff=True)
        print(l,Cl)
        l,Cl = do_limber(astropy.cosmology.Planck15,dndz1,dndz2,pk,\
                         auto=True,use_zeff=False)
        print(l,Cl)
    #
