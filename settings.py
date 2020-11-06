import time
from cachetools import cached
from cachetools.keys import hashkey
import numpy as np
#from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import copy
import functools
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import yaml

# This works with the version of Cobaya from github on March 20, 2020
# the development version fixes a couple bugs w/r/t 2.0.3 and is preferred!

from astropy.cosmology import FlatLambdaCDM
from cobaya.model import get_model
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

import limber2_halofit as L2

# velocileptors dependency. cleft = False turns it off
cleft = False
if cleft:
	import limber2_cleft as L2_cleft
	from LPT.cleft_fftw import CLEFT
	
# Turn off dn/dz uncertainty
dndz_uncertainty = False

# Turn off k^2 dependent bias terms (multiplying Halofit)
alphas = False

# Define your cobaya modules path
modules_path = '/global/cfs/cdirs/m3058/krolewski/cosmo_modules/'

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

################################ SETTINGS ##############################################

color = 'green'
# Specify data, magnification bias, sample: for WISE, indicated by color = (red, green, blue)
# ell limits
low_ell_gg = 100
low_ell_kg = 20
high_ell = 600

# binning
bin_width = 50
low_ind_gg = int(low_ell_gg/bin_width)
low_ind_kg = int(low_ell_kg/bin_width)
high_ind = int(high_ell/bin_width)

# Cross and auto data
data_cross = np.loadtxt('input/clkg.txt')
data_auto = np.loadtxt('input/clgg.txt')

# Redshift distributions. For WISE we consider both "xcorr" dn/dz == b1(z) * dN/dz, and "xmatch" dn/dz == dn/dz
dndz_xcorr = np.loadtxt('input/dndz.txt')
dndz_xmatch = np.loadtxt('input/dndz.txt')

# Magnification bias
s_wise = 0.4
# Euleian bias for Halofit for the magnification-bias term
b1_HF = 2.172

# Camb linear P(k) parameters
zmax = 4.0
Nz = 41
kmax = 10.0

# Limber parameters
Nchi = 200
Nchi_mag = 25
zlist = np.linspace(0.0,4.0,41)


# Cleft parameters
one_loop = True
shear = True
N = 2000
cutoff = 20
jn = 5

# Use the no-neutrino power spectrum
var = 'delta_nonu'

# Output name
output_name = 'chains/analytic_test'

# fsky
fsky_cov = 0.589848876565425

# b1-b2-bs priors (digitized from Abidi et al. 2018)

if cleft:
	abidi = np.loadtxt('Abidi_all_bias.csv',delimiter=',')

	# Eulerian b1(z) prior. Used to define the higher-order bias terms.
	redshift = [0, 0.05, 0.25, 0.4, 0.55, 0.75, 1.00, 1.25, 1.4, 1.50, 1.75, 2.00, 2.15]
	bsml = [0.91, 1.20, 1.25, 1.34, 1.41, 1.43, 1.87, 2.40, 2.79, 2.98, 3.95, 5.36, 5.85]

if alphas:
	# k^2 dependent bias evolution. Defined as -alpha/2 k^2 P_mm, with separate
	# alphas for the galaxy autocorrelation and galaxy-matter cross-correlation
	redshift_for_alpha = np.array([0.00, 0.55, 1.00, 1.25, 1.50, 2.00, 2.50])
	alpha_auto_for_alpha = np.array([0.00, 0.00, -22.22, -38.67, -78.88, 8.88, -33.33])
	alpha_cross_for_alpha = np.array([0.00, 0.00,-9.55,-10.44,-18.77,-43.33,-11.11])


################################ PRIORS ##############################################

# Uniform priors on logA, Om, b1, b2
logA_min = 1.0
logA_max = 4.0

Omegam_min = 0.1
Omegam_max = 0.9

b1_min = -10.0
b1_max = 10.0

b2_min = -10.0
b2_max = 10.0

bs_min = -10.0
bs_max = 10.0

# Gaussian priors on logSN
logSN_prior_sigma = 0.2

################################ PRELIMINARIES ##############################################

# most of this stuff shouldn't be touched--it will only affect the performance of the minimizer
# and thus doesn't really matter

# Starting point for the minimizer

Omegam =0.302496
As = 1e-10*np.exp(3.1515)
b1 = 1.9378
b2 = -5.806960239812743e-06
bs = 0.0
SN = (1./(1844.*(180./np.pi)**2.)) # for runpb, I put in the SN by hand, matching Paper I and *not* 1/nbar (which is 1868 deg^-2 for the green sample)
alpha_cross = 0.9977
alpha_auto = 0.9981
alpha_matter = 0.0
shift = 0.03356
width = 1.0940

# Width of the reference distributions
# These only matter for the minimizer (so basically they don't matter),
# so don't touch them

logA_ref_sigma = 0.01
Omegam_ref_sigma = 0.03

b1_ref_sigma = 0.05
b2_ref_sigma = 0.3
bs_ref_sigma = 0.3
logSN_ref_sigma = 0.2

alpha_cross_ref_sigma = 100.0
alpha_auto_ref_sigma = 100.0
alpha_matter_ref_sigma = 100.0

shift_ref_sigma = 0.1
width_ref_sigma = 0.1

# Fixed cosmology (the other cosmological parameters of the simulation)
ombh2 = 0.02247
Tcmb0 = 2.7255
H0 = 67.7
tau = 0.0925
mnu = 0.0
nnu = 3.046
num_massive_neutrinos = 0
ns = 0.96824

# Fiducial cosmology for higher bias terms
Omegam_for_higher_bias = 0.3092
As_fo_higher_bias = 2.102e-9

# Specify ell range
all_ell = np.linspace(0,high_ell,int(high_ell+1))	

# Planck noise on Clkk (for covariance)
planck_noise = np.loadtxt('input/nlkk.dat')[:,1]

# Get data
bp_cross = data_cross[1,low_ind_kg:high_ind]
bp_cross_err = data_cross[2,low_ind_kg:high_ind]

ell_cross = data_auto[0,low_ind_kg:high_ind]

bp_auto = data_auto[1,low_ind_gg:high_ind]
bp_auto_err = data_auto[2,low_ind_gg:high_ind]

ell_auto = data_auto[0,low_ind_gg:high_ind]

lmin = np.array([19.5, 51.5, 101.5,  151.5, 201.5, 251.5, 301.5, 351.5, 401.5, 451.5, 501.5, 551.5])
lmax = np.array([51.5, 101.5,  151.5, 201.5, 251.5, 301.5, 351.5, 401.5, 451.5, 501.5, 551.5, 601.5])

Nmodes = lmax ** 2. - lmin ** 2.

auto_cross_cov = 2 * data_cross[1,low_ind_gg:high_ind] * data_auto[1,low_ind_gg:high_ind]/(fsky_cov * Nmodes[low_ind_gg:high_ind])

upper = np.concatenate((np.zeros((len(data_auto[1,low_ind_gg:high_ind]),low_ind_gg-low_ind_kg)), (np.diag(auto_cross_cov))
	), axis=1)
lower = np.concatenate((np.zeros((low_ind_gg-low_ind_kg,len(data_auto[1,low_ind_gg:high_ind]))), (np.diag(auto_cross_cov))
	), axis=0)
	
data_cov = np.concatenate(
	(np.concatenate((np.diag(bp_cross_err**2.), upper),axis=0),
	np.concatenate((lower, np.diag(bp_auto_err**2.)),axis=0)), axis=1)

inv_cov = np.linalg.inv(data_cov)


# Set up CLEFT object
if cleft:
	powerspec = np.loadtxt('test_PS.txt') # dummy power spectrum
	k = powerspec[:,0]
	pk = powerspec[:,1]
	zelda =  CLEFT(k, pk, one_loop=one_loop, third_order=False, shear=shear, threads=1, N=N, cutoff=cutoff, jn=jn, import_wisdom=True)

if cleft:
	# Load b1-b2-bs priors (digitized from Abidi et al. 2018)
	b1_abidi = abidi[:,1]
	b2_abidi = abidi[:,2]
	bs_abidi = abidi[:,3]

	b2_spl = InterpolatedUnivariateSpline(b1_abidi,b2_abidi)
	b2_spl_const = InterpolatedUnivariateSpline(b1_abidi,b2_abidi,ext='const')

	def b2_spl_correct(b1_in):
		b2_out = np.zeros_like(b1_in)
		b2_out[b1_in < b1_abidi[0]] = b2_spl_const(b1_in[b1_in < b1_abidi[0]])
		b2_out[b1_in >= b1_abidi[0]] = b2_spl(b1_in[b1_in >= b1_abidi[0]])
		return b2_out

	bs_spl = InterpolatedUnivariateSpline(b1_abidi,bs_abidi)
	bs_spl_const = InterpolatedUnivariateSpline(b1_abidi,bs_abidi,ext='const')


	def bs_spl_correct(b1_in):
		bs_out = np.zeros_like(b1_in)
		bs_out[b1_in < b1_abidi[0]] = bs_spl_const(b1_in[b1_in < b1_abidi[0]])
		bs_out[b1_in >= b1_abidi[0]] = bs_spl(b1_in[b1_in >= b1_abidi[0]])
		return bs_out
	
	def bsml_fn(z):
		bsml_spl = InterpolatedUnivariateSpline(redshift,bsml)
		return bsml_spl(z)


################################ FUNCTIONS ##############################################

# Binning function
def bin(ell,theory_cl,lmin,lmax):
	binned_theory_cl = np.zeros_like(lmin)
	for i in range(len(lmin)):
		binned_theory_cl[i] = np.mean(theory_cl[(ell >= lmin[i]) & (ell < lmax[i])])
	return binned_theory_cl

def wise_bias(z,color):
	'''A rough fit to the wise b(z). To see that this works,
	run implied_bias_rmin2.52_propagate_dndz_error.py and overplot 
	the function from here'''
	if color == 'blue':
		return 0.8 + 1.2 * z
	elif color == 'green':
		return np.max([1.6 * z ** 2.0, 1])	
	elif color == 'red' or color == 'red_16.6' or color == 'red_16.5' or color == 'red_16.2':
		return np.max([2.0 * z ** 1.5, 1])
		
def normalize_dndz(bdndz, color):
	z = bdndz[:,0]
	dz = bdndz[:,1]
	data = bdndz[:,2]
	
	bias = np.array(list(map(lambda x: wise_bias(x, color), z)))
	
	dndz = data/bias
	
	norm = np.sum(dndz * dz)
	return (dndz/norm) * bias

def apply_shift(bdndz, color, deltaz):
	z = bdndz[:,0]
	dz = bdndz[:,1]
	data = bdndz[:,2]
	
	dndz = normalize_dndz(bdndz, color)
		
	shifted_dndz = np.interp(z+deltaz,z,dndz,left=0,right=0)
	
	new_bdndz = np.array([z, dz, shifted_dndz]).T
	
	out = normalize_dndz(new_bdndz, color)
	#out[0] = 0
	#out[-1] = 0
	
	return out
	
def apply_width(bdndz, color, width):
	z = bdndz[:,0]
	dz = bdndz[:,1]
	data = bdndz[:,2]
	
	dndz = normalize_dndz(bdndz, color)
	
	zbar = np.sum(z * dndz * dz)/np.sum(dndz * dz)
		
	shifted_dndz = np.interp(width * (z-zbar) + zbar,z,dndz,left=0,right=0)
	
	new_bdndz = np.array([z, dz, shifted_dndz]).T
	
	out = normalize_dndz(new_bdndz, color)
	#out[0] = 0
	#out[-1] = 0
	
	return out

@cached(cache={}, key=lambda theory, H0, ombh2, omch2, nnu, mnu, num_massive_neutrinos, As, ns: 
	hashkey(H0, ombh2, omch2, nnu, mnu, num_massive_neutrinos, As, ns))
def setup_halofit(theory, H0, ombh2, omch2, nnu, mnu, num_massive_neutrinos, As, ns):
	pars = theory.camb.CAMBparams()

	pars.set_cosmology(H0=H0,
		ombh2=ombh2,
		omch2=omch2,
		nnu=nnu,
		mnu=mnu,
		num_massive_neutrinos=num_massive_neutrinos)
	pars.InitPower.set_params(As=As,
		ns=ns)
	from camb import model as CAMBModel
	pars.NonLinear = CAMBModel.NonLinear_both
	
	halofit = theory.camb.get_matter_power_interpolator(pars,
		nonlinear=True,hubble_units=True,k_hunit=True,kmax=kmax,
		var1=var,var2=var).P
	return halofit

@cached(cache={}, key=lambda Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, 
	nnu, ns, tau, num_massive_neutrinos: hashkey(As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos))
def wrap_halofit(Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos):
	# This is a wrapper for the limber call of the Halofit portion of the power spectrum
	# The idea here is to ensure that at fixed cosmology, the limber integral is cached
	# so that varying the bias and shot noise is as simple as multiplying by a constant
	# and adding another constant.
	# This requires the use of the "@cached" decorator from cachetools
	# see https://stackoverflow.com/questions/30730983/make-lru-cache-ignore-some-of-the-function-arguments
	# The idea is that you pass all of the cosmological parameters to this function and the power spectrum interpolator
	# and it caches the output with a key given by all of the parameters
	# If you add more parameters into your cosmology, don't forgot to add them to this block of code!
	t0 = time.time()
	h = H0/100.
	#omch2 = h**2 * (Om0 - (mnu/93.14)/h**2.) - ombh2
	Om0 = (omch2 + ombh2)/(h**2.) + (mnu/93.14)/h**2.
	Ob0 = (ombh2)/h**2.
	Neff = nnu

	# This cosmology is needed for distance-redshift conversions
	# since we are at low redshift, I include neutrinos in the matter density (assume they are non-relativistic)
	cosmo = FlatLambdaCDM(H0=H0,Om0=Om0,Ob0=Ob0,Tcmb0=2.7255,Neff=Neff)	


	setup_chi_out = L2.setup_chi(cosmo, dndz_xcorr, dndz_xcorr, Nchi, Nchi_mag)

	
	# Only the *internal* number of k points affects the runtime
	# These points are just interpolated between, so the runtime doesn't care about them
	# these should just be something reasonable, where I get reasonable answers out
	k = np.logspace(-4,np.log10(10),1000)
	
	halofit_pk = L2.PowerSpectrum(k, Pk_interpolator, halofit, zlist) #, zelda)
	
	return halofit_pk
	
@cached(cache={}, key=lambda Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, 
	nnu, ns, tau, num_massive_neutrinos: hashkey(As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos))
def wrap_cleft(Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos):
	# This is a wrapper for the limber call of the cleft portion of the power spectrum
	# The idea here is to ensure that at fixed cosmology, the limber integral is cached
	# so that varying the bias and shot noise is as simple as multiplying by a constant
	# and adding another constant.
	# This requires the use of the "@cached" decorator from cachetools
	# see https://stackoverflow.com/questions/30730983/make-lru-cache-ignore-some-of-the-function-arguments
	# The idea is that you pass all of the cosmological parameters to this function and the power spectrum interpolator
	# and it caches the output with a key given by all of the parameters
	# If you add more parameters into your cosmology, don't forgot to add them to this block of code!
	t0 = time.time()
	h = H0/100.
	Om0 = (omch2 + ombh2)/(h**2.) + (mnu/93.14)/h**2.
	Ob0 = (ombh2)/h**2.
	Neff = nnu

	# This cosmology is needed for distance-redshift conversions
	# since we are at low redshift, I include neutrinos in the matter density (assume they are non-relativistic)
	cosmo = FlatLambdaCDM(H0=H0,Om0=Om0,Ob0=Ob0,Tcmb0=2.7255,Neff=Neff)	


	setup_chi_out = L2.setup_chi(cosmo, dndz_xcorr, dndz_xcorr, Nchi, Nchi_mag)

	
	# Only the *internal* number of k points affects the runtime
	# These points are just interpolated between, so the runtime doesn't care about them
	# these should just be something reasonable, where I get reasonable answers out
	k = np.logspace(-4,np.log10(10),1000)
	#print('before power spectrum',time.time()-t0)
	
	cleft_pk = L2_cleft.PowerSpectrum(k, Pk_interpolator, halofit, zlist) #, zelda)
	
	return cleft_pk

@cached(cache={}, key=lambda Pk_interpolator, halofit, s_wise, As, H0, omch2, ombh2, mnu, 
	nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB: hashkey(s_wise, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB))
def wrap_limber(Pk_interpolator, halofit, s_wise, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB):
	t0 = time.time()
	halofit_pk = wrap_halofit(Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos)
	#print('cleft pk',time.time()-t0)
	h = H0/100.
	#omch2 = h**2 * (Om0 - (mnu/93.14)/h**2.) - ombh2
	Om0 = (omch2 + ombh2)/(h**2.) + (mnu/93.14)/h**2.
	Ob0 = (ombh2)/h**2.
	Neff = nnu

	cosmo = FlatLambdaCDM(H0=H0,Om0=Om0,Ob0=Ob0,Tcmb0=2.7255,Neff=Neff)	

	# Modify dn/dz by a shift and a width
	out = apply_shift(np.array([dndz_xcorr[:,0],0.0035*np.ones_like(dndz_xcorr[:,0]),dndz_xcorr[:,1]]).T, color, shift)
	out = apply_width(np.array([dndz_xcorr[:,0], 0.0035*np.ones_like(dndz_xcorr[:,0]), out]).T, color, width)	
	dndz_xcorr_modified = np.array([dndz_xcorr[:,0], out]).T		

	out = apply_shift(np.array([dndz_xmatch[:,0],0.0035*np.ones_like(dndz_xmatch[:,0]),dndz_xmatch[:,1]]).T, color, shift)
	out = apply_width(np.array([dndz_xmatch[:,0], 0.0035*np.ones_like(dndz_xmatch[:,0]), out]).T, color, width)	
	dndz_xmatch_modified = np.array([dndz_xmatch[:,0], out]).T		

	# Pre-define chi- grid to save evaluation time
	setup_chi_out = L2.setup_chi(cosmo, dndz_xcorr_modified, dndz_xmatch_modified, Nchi, Nchi_mag)
	
	# Define alpha_cross(z) and alpha_auto(z) for k^2-dependent bias terms.
	if alphas:
		alpha_cross  = lambda z: np.interp(z, redshift_for_alpha, alpha_cross_for_alpha)
		alpha_auto = lambda z: np.interp(z, redshift_for_alpha, alpha_auto_for_alpha)
	else:
		alpha_cross = lambda z: 1.0
		alpha_auto = lambda z: 1.0
	
	out = L2.do_limber(all_ell, cosmo, dndz_xcorr_modified, dndz_xcorr_modified, s_wise, s_wise, halofit_pk, b1_HF, b1_HF, alpha_auto, alpha_cross, Nchi=Nchi, autoCMB=autoCMB, use_zeff=False, dndz1_mag=dndz_xmatch, dndz2_mag=dndz_xmatch,setup_chi_flag=True,setup_chi_out=setup_chi_out)	
		
	return out
	
@cached(cache={}, key=lambda Pk_interpolator, halofit, s_wise, As, H0, omch2, ombh2, mnu, 
	nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB: hashkey(s_wise, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB))
def wrap_limber_cleft(Pk_interpolator, halofit, s_wise, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos, shift, width, autoCMB):
	t0 = time.time()
	cleft_pk = wrap_cleft(Pk_interpolator, halofit, As, H0, omch2, ombh2, mnu, nnu, ns, tau, num_massive_neutrinos)
	#print('cleft pk',time.time()-t0)
	#print('As',As)
	#print('H0',H0)
	h = H0/100.
	#omch2 = h**2 * (Om0 - (mnu/93.14)/h**2.) - ombh2
	Om0 = (omch2 + ombh2)/(h**2.) + (mnu/93.14)/h**2.
	Ob0 = (ombh2)/h**2.
	Neff = nnu

	cosmo = FlatLambdaCDM(H0=H0,Om0=Om0,Ob0=Ob0,Tcmb0=2.7255,Neff=Neff)	

	# Define the fiducial evolution of b2, shear bias, and k^2 dependent bias
	pars = {'option': 'auto plus CMB cross','b1': 1.0, 'b2': lambda z: b2_spl_correct(bsml_fn(z)-1.), 'bs': lambda z: bs_spl_correct(bsml_fn(z)-1.), 'alpha_cross': lambda z: np.interp(z, redshift_for_alpha, alpha_cross_for_alpha), 'alpha_auto': lambda z: np.interp(z, redshift_for_alpha, alpha_auto_for_alpha), 'alpha_matter': 1.0}

	# Apply shift and width to dN/dz
	out = apply_shift(np.array([dndz_xcorr[:,0],0.0035*np.ones_like(dndz_xcorr[:,0]),dndz_xcorr[:,1]]).T, color, shift)
	out = apply_width(np.array([dndz_xcorr[:,0], 0.0035*np.ones_like(dndz_xcorr[:,0]), out]).T, color, width)	
	dndz_xcorr_modified = np.array([dndz_xcorr[:,0], out]).T		

	out = apply_shift(np.array([dndz_xmatch[:,0],0.0035*np.ones_like(dndz_xmatch[:,0]),dndz_xmatch[:,1]]).T, color, shift)
	out = apply_width(np.array([dndz_xmatch[:,0], 0.0035*np.ones_like(dndz_xmatch[:,0]), out]).T, color, width)	
	dndz_xmatch_modified = np.array([dndz_xmatch[:,0], out]).T		

	setup_chi_out = L2_cleft.setup_chi(cosmo, dndz_xcorr_modified, dndz_xmatch_modified, Nchi, Nchi_mag)
	out = L2_cleft.do_limber(all_ell, cosmo, dndz_xcorr_modified, dndz_xcorr_modified, dndz_xmatch_modified, s_wise, s_wise, cleft_pk, pars, b1_HF, b1_HF, Nchi=Nchi, autoCMB=autoCMB, use_zeff=False, dndz1_mag=dndz_xmatch, dndz2_mag=dndz_xmatch,setup_chi_flag=True,setup_chi_out=setup_chi_out)	
	
		
	return out


def get_angular_power_spectra(halofit,_theory,
	s_wise=0.4,
	Omegam=0.3,
	b1=1.0,
	b2=0.0,
	bs=0.0,
	alpha_auto=0.0,
	alpha_cross=0.0,
	alpha_matter=0.0,
	SN=1e-7,
	shift=0.0,
	width=1.0,autoCMB=False):
	'''Returns angular power spectra'''
	t0 = time.time()
	Pk_interpolator = _theory.get_Pk_interpolator()['delta_nonu_delta_nonu'].P
	#print('pk interpolator',time.time()-t0)

	auto, cross = wrap_limber(Pk_interpolator, halofit, s_wise, _theory.get_param('As'), _theory.get_param('H0'),
		_theory.get_param('omch2'), _theory.get_param('ombh2'), _theory.get_param('mnu'),
		_theory.get_param('nnu'), _theory.get_param('ns'), _theory.get_param('tau'),
		_theory.get_param('num_massive_neutrinos'), shift, width, autoCMB)
		

	# Multiply by bias	
	lim_clkg, alpha_cross_term, lim_clkmu = cross
	if alphas:
		lim_cross  = b1* lim_clkg + lim_clkmu + alpha_cross * alpha_cross_term
	else:
		lim_cross = b1* lim_clkg + lim_clkmu

	# Multiply by bias
	lim_clgg, alpha_auto_term, lim_clgmu, lim_clmumu = auto
	if alphas:
		lim_auto = b1**2 * lim_clgg + 2*b1*lim_clgmu + lim_clmumu + alpha_auto * alpha_auto_term
	else:
		lim_auto = b1**2 * lim_clgg + 2*b1*lim_clgmu + lim_clmumu
	
	if cleft:	
		# Fix the cosmology for the b2 and shear bias terms
		auto_xcorr, cross_xcorr, auto_xmatch, cross_xmatch, auto_mixed = wrap_limber_cleft(Pk_interpolator, halofit, 0.4, As_for_higher_bias, H0,
			Omegam_for_higher_bias*(H0/100.)**2-ombh2, ombh2, mnu,
			nnu, ns, tau,
			num_massive_neutrinos, 0.0, 1.0, autoCMB)

		lim_clkg_higher_order = b2 * cross_xmatch[0][:,2] + bs * cross_xmatch[0][:,3]	

		# Multiply by bias.
		# The assumption here is that b2 and bs terms use dN/dz as the galaxy kernel, multiplied by the fiducial b2 and bs evolution, 
		# and b1 uses b(z) * dN/dz in the galaxy kernel.
		# This leads to the auto_xmatch, auto_mixed, etc. terms.
		lim_clgg_higher_order = np.squeeze((b2 * auto_xmatch[0][:,3]
			 + (b1 * auto_mixed[0][:,4] - auto_xmatch[0][:,4]) * b2
			 + b2**2 * auto_xmatch[0][:,5] + bs * auto_xmatch[0][:,6] 
			 + (b1 * auto_mixed[0][:,7] - auto_xmatch[0][:,7]) * bs
			 + b2 * bs * auto_xmatch[0][:,8] + bs**2 * auto_xmatch[0][:,9]))
		 
		return np.squeeze(lim_cross)+lim_clkg_higher_order, np.squeeze(lim_auto)+lim_clgg_higher_order
	else:
		return np.squeeze(lim_cross), np.squeeze(lim_auto)
	
def covariance(model, like_dict):
	#As, Omegam, b1, b2, bs, SN, alpha_cross, alpha_auto, alpha_matter):
	'''Returns diagonal covariance of Clkg as a function of the cosmological parameters'''
	# Set up halofit object
	pars =  model.likelihood.theory.camb.CAMBparams()
	pars.set_cosmology(H0=model.likelihood.theory.get_param('H0'),
		ombh2=model.likelihood.theory.get_param('ombh2'),
		omch2=model.likelihood.theory.get_param('omch2'),
		nnu=model.likelihood.theory.get_param('nnu'),
		mnu=model.likelihood.theory.get_param('mnu'),
		num_massive_neutrinos=model.likelihood.theory.get_param('num_massive_neutrinos'))
	pars.InitPower.set_params(As=model.likelihood.theory.get_param('As'),
		ns=model.likelihood.theory.get_param('ns'))
	from camb import model as CAMBModel
	pars.NonLinear = CAMBModel.NonLinear_both
	
	halofit = model.likelihood.theory.camb.get_matter_power_interpolator(pars,
		nonlinear=True,hubble_units=True,k_hunit=True,kmax=kmax,
		var1=var,var2=var).P
		
	cross, auto = get_angular_power_spectra(halofit,model.likelihood.theory,
		s_wise,
		like_dict['Omegam'],
		like_dict['b1'],
		1.0,
		1.0,
		0,
		0,
		0,
		info['params']['SN'],
		0,
		1)

	lim_clkk, _ = get_angular_power_spectra(halofit,model.likelihood.theory,
		s_wise,
		like_dict['Omegam'],
		like_dict['b1'],
		1.0,
		1.0,
		0,
		0,
		0,
		info['params']['SN'],
		0,
		1,
		autoCMB=True)


	clkg_cov_diag = (cross**2. + (auto + info['params']['SN']) * (lim_clkk + planck_noise[:high_ell+1]))/(fsky_cov * (2*all_ell + 1.))
	
	clgg_cov_diag = (2 * (auto + info['params']['SN']) ** 2.)/(fsky_cov * (2*all_ell + 1.))
	
	clkg_clgg_cross_cov_diag = (2 * (auto + info['params']['SN']) * cross)/(fsky_cov * (2*all_ell + 1.))
		
	upper = np.concatenate((np.zeros((high_ell - low_ell_gg, low_ell_gg-low_ell_kg)),
		np.diag(clkg_clgg_cross_cov_diag[low_ell_gg:high_ell])), axis=1)
	lower = upper.T
	
	return np.concatenate( 
	(np.concatenate((np.diag(clkg_cov_diag[low_ell_kg:high_ell]), upper),axis=0), 
	np.concatenate((
	lower,
	np.diag(clgg_cov_diag[low_ell_gg:high_ell])),axis=0)),axis=1) 


def clkg_likelihood(s_wise=0.4,
	b1=1.0,
	b2=0.0,
	bs=0.0,
	alpha_auto=0.0,
	alpha_cross=0.0,
	alpha_matter=0.0,
	SN=1e-7,
	shift=0.0,
	width=1.0,
	_theory={'Pk_interpolator': {'z': np.linspace(0,zmax,Nz), 
	'k_max': kmax, 'nonlinear': False,
	'hubble_units': True,'k_hunit': True, 
	'vars_pairs': [[var,var]]}}):
	'''Function for the likelihood.'''
	t0 = time.time()
	Omegam = (_theory.get_param('ombh2') + _theory.get_param('omch2'))/((_theory.get_param('H0')/100.)**2.)
	
	Pk_interpolator = _theory.get_Pk_interpolator()['delta_nonu_delta_nonu'].P

	# Set up halofit object
	
	halofit = setup_halofit(_theory, _theory.get_param('H0'), _theory.get_param('ombh2'),
		_theory.get_param('omch2'), _theory.get_param('nnu'),
		_theory.get_param('mnu'),_theory.get_param('num_massive_neutrinos'),
		_theory.get_param('As'),_theory.get_param('ns'))	
	
	SN_auto = SN
	SN_cross = 0.	

	lim_cross, lim_auto = get_angular_power_spectra(halofit, _theory, s_wise, Omegam, b1, b2, bs, alpha_auto, alpha_cross, alpha_matter, SN, shift, width)

	#print('lim_auto immediately before bin',lim_auto)

	# Bin
	lim_bin = bin(all_ell,lim_cross,lmin[low_ind_kg:high_ind],lmax[low_ind_kg:high_ind]) + SN_cross
	# Correct for namaster binning
	lim_bin_cross = lim_bin #* clkg_corr[low_ind:high_ind]
	# Make log-likelihood
	loglike_clkg = -0.5 * np.sum((bp_cross-lim_bin_cross)**2./bp_cross_err**2.)
	
	
	# Bin and add SN
	lim_bin = bin(all_ell,lim_auto,lmin[low_ind_gg:high_ind],lmax[low_ind_gg:high_ind]) + SN_auto
	# Correct for namaster binning
	lim_bin_auto = lim_bin #* clgg_corr[low_ind:high_ind]
	# Make log-likelihood
	loglike_clgg = -0.5 * np.sum((bp_auto-lim_bin_auto)**2./bp_auto_err**2.)
			
			
	delta = np.concatenate((bp_cross - lim_bin_cross,bp_auto - lim_bin_auto),axis=0)
	
	loglike = -0.5 * np.dot(np.dot(delta,inv_cov),delta)
	
	print('chisq',_theory.get_param('As'),Omegam,b1,b2,s_wise,alpha_cross,SN,shift,width,-2*loglike,-2*loglike_clkg,-2*loglike_clgg)

	return loglike

################################ INFO ##############################################

likelihood_block = {'clkg_likelihood': {'external': "import_module('settings').clkg_likelihood", 'speed': 125.}}
theory_block = {'camb': {'stop_at_error': True, 'speed': 4.5, 
       'extra_args':
        {'redshifts': [0.],
        'halofit_version': 'mead',
         'WantTransfer': True,
         'WantCls': False,
         'Want_CMB': False,
         'Want_cl_2D_array': False,
         'Want_CMB_lensing': False,
         'AccuracyBoost': 1.0,
         'lAccuracyBoost': 1.0,
         'lSampleBoost': 1.0,
        'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
        'lens_potential_accuracy': 1}}}
minimize_block =  {'method': 'scipy',
        'ignore_prior': False,
        'override_scipy': {'method': 'Nelder-Mead'},
        'max_evals': 1e6,
        'confidence_for_unbounded': 0.9999995}
        	
info = {
    'params': {
        # Fixed
        'ombh2': ombh2, 'H0': H0, 'tau': tau,
        'mnu': mnu, 'nnu': nnu, 'num_massive_neutrinos': num_massive_neutrinos,
        'ns': ns, 
        #'SN': SN,
        #'As': As,
        #'Omegam': Om,
         #Sampled
        'logA': {
           'prior': {'min': logA_min, 'max': logA_max},
           'ref': 3.05,
           'proposal': logA_ref_sigma,
               'latex': '\log(10^{10} A_\mathrm{s})',
               'drop': True},
        'As': {'value': 'lambda logA: 1e-10*np.exp(logA)',
               'latex': 'A_\mathrm{s}'},
        'Omegam': {
            'prior': {'min': Omegam_min, 'max': Omegam_max},
            'ref': Omegam,
            'latex': '\Omega_m',
            'proposal': Omegam_ref_sigma,
            'drop': True},
        'logSN': {
           'prior': {'dist': 'norm',
              'loc': -7.,
              'scale': logSN_prior_sigma},
              'ref': -7.,
              'proposal': logSN_ref_sigma,
              'drop': True},
        'SN': {'value': 'lambda logSN: 10**logSN',
               'latex': 'SN'},
        #'As': {
        #    'prior': {'min': 0.1e-9, 'max': 5.0e-9},
        #    'ref': As,
        #    'latex': 'A_s',
        #    'proposal': As_err},
        #'As': As,
        #'Omegam':Om,
        #'SN': {
        #    'prior': {'min': 0.0e-7, 'max': 100.0e-7},
        #    'ref': SN,
        #    'latex': 'SN',
        #    'proposal': SN_err},
        'b1': {
            'prior': {'min': b1_min, 'max': b1_max},
            'ref': b1,
            'latex': r'b1',
            'proposal': b1_ref_sigma},
        'b2': 1.0,
        'bs': 1.0,
        's_wise': {
            'prior': {'dist': 'norm',
            'loc': s_wise,
            'scale': 0.05},
            'ref': s_wise,
            'latex': r's',
            'proposal': 0.05},
        #'alpha_cross': {
        #    'prior': {'min': -10000.0, 'max': 10000.0},
        #    'ref': alpha_cross,
        #    'latex': r'\alpha_{cross}',
        #    'proposal': alpha_cross_err},
        #'alpha_auto': {
        #    'prior': {'min': -10000.0, 'max': 10000.0},
        #    'ref': alpha_auto,
        #    'latex': r'\alpha_{auto}',
        #    'proposal': alpha_auto_err},
        #'alpha_matter': {
        #    'prior': {'min': -10000.0, 'max': 10000.0},
        #    'ref': alpha_matter,
        #    'latex': r'\alpha_{matter}',
        #    'proposal': alpha_matter_err},
        'alpha_cross': {
            'prior': {'min': 0.0, 'max': 2.0},
            'ref': 1.0,
            'latex': r'\alpha_{cross}',
            'proposal': 0.05},
        #'alpha_auto': {
        #    'prior': {'dist': 'norm',
        #    'loc': alpha_auto,
        #    'scale': alpha_auto_prior_sigma},
        #    'ref': alpha_auto,
        #    'latex': r'\alpha_{auto}',
        #    'proposal': alpha_auto_ref_sigma},
        'alpha_matter': 0.0,
        #'alpha_cross': 0.0,
        'alpha_auto': {
            'prior': {'min': 0.0, 'max': 2.0},
            'ref': 1.0,
            'latex': r'\alpha_{auto}',
            'proposal': 0.05},
        'shift': {'ref': shift,
            'prior': {'min': -1.0, 'max': 1.0},
            'latex': 'shift',
            'proposal': shift_ref_sigma},
        'width': {'ref': width,
            'prior': {'min': 0.0, 'max': 10.0},
            'latex': 'width',
            'proposal': width_ref_sigma},
        #'alpha_matter': {
        #    'prior': {'dist': 'norm',
        #    'loc': alpha_matter,
        #    'scale': alpha_matter_prior_sigma},
        #    'ref': alpha_matter,
        #    'latex': r'\alpha_{matter}',
        #    'proposal': alpha_matter_ref_sigma},
        #'bs': "lambda ombh2,H0,tau,mnu,nnu,num_massive_neutrinos,ns,As,Omegam,SN,b1,alpha_cross,alpha_auto,alpha_matter: -2./7. * b1",
        #'b2': "lambda ombh2,H0,tau,mnu,nnu,num_massive_neutrinos,ns,As,Omegam,SN,b1,alpha_cross: ((b1 * 1.686 + 1)**2. - 3 * (b1 * 1.686 + 1))/1.686**2.",
        },
    'likelihood': likelihood_block,
    'theory': theory_block,
    'sampler': {'minimize': 
        minimize_block},
    'modules': modules_path,
    'output': output_name + '_minimize',
    'timing': True,
    'resume': True}
    

if not os.path.exists('input/likelihood_info.yaml'):
	yaml.dump(info, open('input/likelihood_info.yaml','w'))

info = yaml.load(open('input/likelihood_info.yaml','r'))

# Activate timing (we will use it later)
#info['timing'] = True
#info['resume'] = True
#info['debug'] = True

################################ PRIOR ##############################################

#info['prior'] = {'b2_abidi': lambda b1, b2: -(b2 - b2_spl_correct(b1))**2./b2_prior_sigma**2.}
#			'bs_abidi': lambda b1, bs: -(bs - bs_spl_correct(b1))**2./sigma_bs**2.}
if dndz_uncertainty:
	shift_width_covariance = np.array([[0.00120118, 0.00210338],
		   [0.00210338, 0.00513655]])
	   
	def chisq(shift,width):
		arr = np.array([shift,width-1.0])
		chisq = np.dot(arr.T, np.dot(np.linalg.inv(shift_width_covariance),arr))
		return chisq

	info['prior'] = {'shift_width': lambda shift, width: -0.5 * chisq(shift,width)}