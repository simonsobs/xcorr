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
from LPT.cleft_fftw import CLEFT
from multiprocessing import Pool

# Set up CLEFT object
powerspec = np.loadtxt('test_PS.txt') # dummy power spectrum
k = powerspec[:,0]
pk = powerspec[:,1]
one_loop = True
shear = True
kmin = 1e-3
kmax = 0.5
Nk = 100
try:
	zelda =  CLEFT(k, pk, one_loop=one_loop, third_order=False, shear=shear, threads=1, N=2000, cutoff=20, jn=5, import_wisdom=True)
except FloatingPointError:
	np.seterr(over='ignore',under='ignore')
	zelda =  CLEFT(k, pk, one_loop=one_loop, third_order=False, shear=shear, threads=1, N=2000, cutoff=100, jn=20, import_wisdom=True)
	np.seterr(over='raise',under='raise')

def load_cleft(z, k, pk, pk_HF, nk, kmin, kmax):
	zelda =  CLEFT(k, pk, one_loop=one_loop, third_order=False, shear=shear, threads=1, N=2000, cutoff=50, jn=5, import_wisdom=True)
	#zelda.update_power_spectrum(k,pk)
	zelda.make_ptable(nk = nk, kmin=kmin, kmax=kmax)
	table = zelda.pktable
	#np.savetxt('table_%.2f.txt' % z,table)
	#np.savetxt('pklin_%.2f.txt' % z,np.array([k,pk]).T)
	#np.savetxt('pkHF_%.2f.txt' % z,np.array([k,pk_HF]).T)

	# sometimes floating underflow errors give NaNs
	#print('k',k)
	#print('pk',pk)
	# replace these with zeros
	table[:1,1:] = 0
	table[np.isnan(table)] = 0
	table[:,1][np.abs(table[:,1]) >= 1e6] = np.interp(table[:,0][np.abs(table[:,1]) >= 1e6],k,pk_HF)
	table[:,2][np.abs(table[:,2]) >= 1e6] = np.interp(table[:,0][np.abs(table[:,2]) >= 1e6],k,2*pk_HF)
	table[:,3][np.abs(table[:,3]) >= 1e6] = np.interp(table[:,0][np.abs(table[:,3]) >= 1e6],k,pk_HF)
	
	table[:,4][np.abs(table[:,4]) >= 1e6] = 0
	table[:,5][np.abs(table[:,5]) >= 1e6] = 0
	table[:,6][np.abs(table[:,6]) >= 1e6] = 0
	
	if shear:
		table[:,7][np.abs(table[:,7]) >= 1e6] = 0
		table[:,8][np.abs(table[:,8]) >= 1e6] = 0
		table[:,9][np.abs(table[:,9]) >= 1e6] = 0
		table[:,10][np.abs(table[:,10]) >= 1e6] = 0
	
		table[:,11][np.abs(table[:,11]) >= 1e6] = np.interp(table[:,0][np.abs(table[:,11]) >= 1e6],k,pk_HF)
	else:
		table[:,7][np.abs(table[:,7]) >= 1e6] = np.interp(table[:,0][np.abs(table[:,7]) >= 1e6],k,pk_HF)

	
	#np.savetxt('As_2109_Om_3092/table_%.2f.txt' % z,table)
	#np.savetxt('table.txt',table)
	#np.savetxt('pklin.txt',np.array([k,pk]).T)
	#np.savetxt('pkHF.txt',np.array([k,pk_HF]).T)
	return table


#import zeldovich as Z

#def znorm(dndz):
#	'''return dn/dz normalized in redshift'''
#	return dndz[:,1]/np.trapz(dndz[:,1],x=dndz[:,0])

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
		t0  = time.time()
		#num_of_k_parts = 2
		
		pool = Pool(processes=len(zlist))
		
		
		pk_array = np.zeros((len(k),len(zlist)))# *num_of_k_parts))
		for i,z in enumerate(zlist):
			pk_array[:,i] = Pk_interpolator(z,k)
			
		pk_HF_array = np.zeros((len(k),len(zlist)))# *num_of_k_parts))
		for i,z in enumerate(zlist):
			pk_HF_array[:,i] = halofit(z,k)
			#pk_array[:,len(zlist)+i] = Pk_interpolator(z,k)
			
		#print('after pk interpolator',time.time()-t0)
		
		#kmed = np.logspace(np.log10(kmin),np.log10(kmax),Nk)[Nk//num_of_k_parts]
		
		
			
		zzip = zip(zlist, [k]*len(zlist), pk_array.transpose(), pk_HF_array.transpose(),
			#[one_loop]*len(zlist)*num_of_k_parts, [shear]*len(zlist)*num_of_k_parts,
			#[zelda]*len(zlist)*num_of_k_parts,
			[Nk]*len(zlist), [kmin]*len(zlist), [kmax]*len(zlist))
			
		test = load_cleft(zlist[0], k, pk_array.transpose()[0],pk_HF_array.transpose()[0], Nk, kmin, kmax)
		#print('Nk',Nk)
		#print('kmin',kmin)
		#print('kmax',kmax)
		#print('zlist[0]',zlist[0])
		#np.savetxt('k.txt',k)
		#np.savetxt('pk_array.txt',pk_array.transpose()[0])
		#print('test',test)
		#print(5/0)
			
		#print('zzip',time.time()-t0)
		#print('zzip',list(zzip)[4])
		#print('kmin',np.concatenate(([kmin]*len(zlist),[kmed]*len(zlist))))
		
		#print('k',k)
		#print('Nk',Nk)
		#print('kmin',kmin)
		#print('kmax',kmax)
		#np.savetxt('k.txt',k)
		
		pktable_list_from_pool = np.array(pool.starmap(load_cleft,zzip))
		
		#self.pktable_list = np.zeros((np.shape(pktable_list_from_pool)[0]//2, np.shape(pktable_list_from_pool)[1]*2,np.shape(pktable_list_from_pool)[2]))
		#for i in range(len(zlist)):
		#	self.pktable_list[i,:Nk//num_of_k_parts,:] = pktable_list_from_pool[i,:,:]
		#	self.pktable_list[i,Nk//num_of_k_parts:,:] = pktable_list_from_pool[len(zlist)+i,:,:]
		
		self.pktable_list = pktable_list_from_pool
		
		#print('shape pktable list',np.shape(pktable_list_from_pool))
		#np.savetxt('pktable_list_from_pool.txt',pktable_list_from_pool[0])
		#np.savetxt('pk_array.txt',pk_array.T)
			
		#print('pktable is nan',np.where(np.isnan(self.pktable_list)))
		
		#pktest = Pk_interpolator(1.34,k)
		#cleft_test = load_cleft(1.34, k, pktest, 100, 0.01, 10.0)
		#np.savetxt('cleft_test.txt',cleft_test)
		
		pool.close()
				
		
		#print('pktable_list',self.pktable_list[0][:,0])
		#print('pktable_list_from_pool',np.shape(pktable_list_from_pool))
		#print(5/0)
		#self.pktable_list = []
		# There is an "update_power_spectrum" function that I could use instead of re-generating
		# CLEFT objects at all cosmologies (so make one master cleft object, then keep
		# updating its power spectrum for all cosmologies)
		# But it is barely any faster (at least on my laptop. would be good to do the same
		# test on a nersc compute node)
		
		self.zlist = np.array(zlist)
		#print('done making PS',time.time()-t0)
		
		#np.savetxt('cutoff20.txt',self.pktable_list.flatten())
		#np.savetxt('k.txt',k)
		#np.savetxt('pk_array.txt',pk_array)

	def __call__(self,pars,kval,zz,b1_HF,b2_HF,one_loop=True,shear=True):
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
		coef = self.lagrange_spam(zz)
		
		p_gg = np.zeros((11,len(kval)))
		p_gm1 = np.zeros((5,len(kval)))
		p_gm2 = np.zeros((5,len(kval)))
		p_mm = np.zeros((1,len(kval)))
		p_gm1_HF = np.zeros((1,len(kval)))
		p_gm2_HF = np.zeros((1,len(kval)))
		#print('coef',coef)
		#print('pktable_list',self.pktable_list)
		j = 0
		for c,z in zip(coef,self.pktable_list):
			# We support 2 options for the output power spectrum.
			# In normal operations, pars['option'] = 'auto plus CMB cross',
			# and the code outputs auto and cross spectra for a single tracer.
			# If pars['option'] = 'galaxy cross', the code outputs
			# the cross spectra between two tracers!
			# We want the option to exist to get the galaxy cross-spectra but it is not clear
			# if we will ever fit parameters to the cross spectra.
			if pars['option'] == 'galaxy cross':
				b1_A = pars['tracer1']['b1']
				b1_B = pars['tracer2']['b1']
				
				b2_A = pars['tracer1']['b2']
				b2_B = pars['tracer2']['b2']
				
				bs_A = pars['tracer1']['bs']
				bs_B = pars['tracer2']['bs']
				
				alpha_auto = pars['alpha_auto'](zz)
				alpha_cross = pars['alpha_cross'](zz)
				alpha_matter = pars['alpha_matter']
				
				kk = z[0,:]
				
				if shear == True:
					lenpar = 12
				elif shear == False:
					lenpar = 8
				
				for i in range(np.shape(p_gg)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[3] = 0.5 * (b1_A + b1_B)
					elif i == 2:
						par = np.zeros(lenpar)
						par[4] = b1_A * b1_B
					elif i == 3:
						par = np.zeros(lenpar)
						par[5] = 0.5 * (b2_A + b2_B)
					elif i == 4:
						par = np.zeros(lenpar)
						par[6] = 0.5 * (b1_A * b2_B + b1_B * b2_A)
					elif i == 5:
						par = np.zeros(lenpar)
						par[7] = b2_A * b2_B
					elif i == 6:
						par = np.zeros(lenpar)
						if shear:
							par[8] = 0.5 * (bs_A + bs_B)
					elif i == 7:
						par = np.zeros(lenpar)
						if shear:
							par[9] = 0.5 * (b1_A * bs_B + b1_B * bs_A)
					elif i == 8:
						par = np.zeros(lenpar)
						if shear:
							par[10] = 0.5 * (b2_A * bs_B + b2_B * bs_A)
					elif i == 9:
						par = np.zeros(lenpar)
						if shear:
							par[11] = bs_A * bs_B
					elif i == 10:
						par = np.zeros(lenpar)
						par[0] = -0.5 * alpha_auto 
					if i != 10:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
					#print('before interp',time.time()-t0)
					p_gg[i,:] += np.interp(kval,kk,pk,left=0,right=0)*c
					#print('after interp',time.time()-t0)
					

				for i in range(np.shape(p_gm1)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[3] = b1_A/2.
					elif i == 2:
						par = np.zeros(lenpar)
						par[5] = b2_A/2.
					elif i == 3:
						par = np.zeros(lenpar)
						if shear:
							par[8] = bs_A/2.
					elif i == 4:
						par = np.zeros(lenpar)
						par[0] = -0.5 * alpha_cross
					if i != 4:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
					p_gm1[i,:] += np.interp(kval,kk,pk,left=0,right=0)*c

				for i in range(np.shape(p_gm2)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[3] = b1_B/2.
					elif i == 2:
						par = np.zeros(lenpar)
						par[5] = b2_B/2.
					elif i == 3:
						par = np.zeros(lenpar)
						if shear:
							par[8] = bs_B/2.
					elif i == 4:
						par = np.zeros(lenpar)
						par[0] = -0.5 * alpha_cross
					if i != 4:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
					p_gm2[i,:] += np.interp(kval,kk,pk,left=0,right=0)*c


				for i in range(np.shape(p_mm)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[0] = -0.5 * alpha_matter
					if i != 1:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
					p_mm[i,:] += np.interp(kval,kk,pk,left=0,right=0)*c
					#print('z1',z[:,1])

				
				# see limber2_split.py for the correct way to do this!
			elif pars['option'] == 'auto plus CMB cross':
				# let's check to see how the pktable looks if I turn one-loop
				# and shear on/off
				
				# the relevant equations are B2 in https://arxiv.org/pdf/1609.02908.pdf (galaxy auto)
				# and 2.7 in https://arxiv.org/pdf/1706.03173.pdf (CMB cross)
				# Note that the shot noise terms are added in the external module and so we don't deal with them here.
				
				'''if one_loop == True:
					if shear == True:
						par = np.array([1, 1, 1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
					elif shear == False:
						par = np.array([1, 1, 1, b1, b1**2, b2, b1*b2, b2**2])
				elif one_loop == False:
					if shear == True:
						par = np.array([1, 0, 0, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
					elif shear == False:
						par = np.array([1, 0, 0, b1, b1**2, b2, b1*b2, b2**2])'''
				b1 = pars['b1']
				b2 = pars['b2'](zz)
				bs = pars['bs'](zz)
				alpha_auto = pars['alpha_auto'](zz)
				alpha_cross = pars['alpha_cross'](zz)
				alpha_matter = pars['alpha_matter']
				
				kk = z[:,0]
				
				if shear == True:
					lenpar = 11
				elif shear == False:
					lenpar = 7
				
				for i in range(np.shape(p_gg)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[1] = b1
					elif i == 2:
						par = np.zeros(lenpar)
						par[2] = b1**2
					elif i == 3:
						par = np.zeros(lenpar)
						par[3] = b2
					elif i == 4:
						par = np.zeros(lenpar)
						par[4] = b1*b2
					elif i == 5:
						par = np.zeros(lenpar)
						par[5] = b2**2
					elif i == 6:
						par = np.zeros(lenpar)
						if shear:
							par[6] = bs
					elif i == 7:
						par = np.zeros(lenpar)
						if shear:
							par[7] = b1*bs
					elif i == 8:
						par = np.zeros(lenpar)
						if shear:
							par[8] = b2*bs
					elif i == 9:
						par = np.zeros(lenpar)
						if shear:
							par[9] = bs**2
					elif i == 10:
						par = np.zeros(lenpar)
						par[10] = -0.5 * alpha_auto 
					if i != 10:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
					#print(time.time()-t0,'before interp')
					interpval = np.interp(kval,kk,pk,left=0,right=0)*c
					#print(time.time()-t0,'after interp')
					#print('interpval',interpval)
					#print('interp',np.interp(kval,kk,pk,left=0,right=0))
					#print('par',par)
					#print('z',z)
					#print('z1',z[:,1])


					#print(time.time()-t0,'right after interp')
					p_gg[i,:] += interpval

					if i == 0:
						p_gm1[0,:] += interpval
						p_gm2[0,:] += interpval
					elif i == 1:
						p_gm1[1,:] += 0.5 * interpval
						p_gm2[1,:] += 0.5 * interpval
					elif i == 3:
						p_gm1[2,:] += 0.5 * interpval
						p_gm2[2,:] += 0.5 * interpval
					elif i == 6:
						p_gm1[3,:] += 0.5 * interpval
						p_gm2[3,:] += 0.5 * interpval					
					#elif i == 10:
					#	p_gm1[i,:] += 0.5 * interpval
					#	p_gm2[i,:] += 0.5 * interpval
					#print(time.time()-t0,'after interp')

				for i in range(np.shape(p_gm1)[0]):
					'''if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[3] = b1/2.
					elif i == 2:
						par = np.zeros(lenpar)
						par[5] = b2/2.
					elif i == 3:
						par = np.zeros(lenpar)
						if shear:
							par[8] = bs/2.'''
					if i == 4:
						par = np.zeros(lenpar)
						par[10] = -0.5 * alpha_cross
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
						interpval = np.interp(kval,kk,pk,left=0,right=0)*c
						p_gm1[i,:] += interpval
						p_gm2[i,:] += interpval

					'''if i == 4:
						print('kval',kval)
						print('kk',kk)
						print('pk',pk)
						print('par',par)
						print('z[:,10]',z[:,10])
						print('interpval',interpval)
						print('zlist[j]',self.zlist[j])
						print('p_gm1',p_gm1[i,:])'''
					
				'''print(p_gm1[0,:],'p_gm0')
				print(p_gm1[1,:],'p_gm1')
				print(p_gm1[2,:],'p_gm2')
				print(p_gm1[3,:],'p_gm3')
				p_gm1[0,:] = 0.
				p_gm1[1,:] = 0.
				p_gm1[2,:] = 0.
				p_gm1[3,:] = 0.
				p_gm1[4,:] = 0.

				p_gm2[0,:] = 0.
				p_gm2[1,:] = 0.
				p_gm2[2,:] = 0.
				p_gm2[3,:] = 0.
				p_gm2[4,:] = 0.


				
				
				p_gm1[0,:] += p_gg[0,:]
				p_gm1[1,:] += 0.5 * p_gg[1,:]
				p_gm1[2,:] += 0.5 * p_gg[3,:]
				p_gm1[3,:] += 0.5 * p_gg[6,:]
				par = np.zeros(lenpar)
				par[0] = -0.5 * alpha_cross
				pk = np.sum(par*z[:,1:],axis=1)*kk**2.
				p_gm1[4,:] += np.interp(kval,kk,pk,left=0,right=0)*c
				
				p_gm2[0,:] += p_gm1[0,:]
				p_gm2[1,:] += p_gm1[1,:]
				p_gm2[2,:] += p_gm1[2,:]
				p_gm2[3,:] += p_gm1[3,:]
				p_gm2[4,:] += p_gm1[4,:]'''

				'''for i in range(np.shape(p_mm)[0]):
					if i == 0:
						par = np.zeros(lenpar)
						par[0] = 1
						if one_loop:
							par[1] = 1
							par[2] = 1
					elif i == 1:
						par = np.zeros(lenpar)
						par[0] = -0.5 * alpha_matter
					if i != 1:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)
					else:
						prod = par*z[:,1:]
						prod[np.isnan(prod)] = 0
						prod[np.isinf(prod)] = 0
						pk = np.sum(prod,axis=1)*kk**2.
			
					p_mm[i,:] += np.interp(kval,kk,pk,left=0,right=0)*c'''
				
				#print('zz', zz)
				#print('kk', kk)
				#print('halofit',self.halofit(zz,kk))	
				#p_mm[0,:] = np.interp(kval,kk,np.squeeze(self.halofit(zz, kk)),left=0,right=0)*c
				
				#p_gm1_HF[0,:] = b1_HF * np.interp(kval,kk,np.squeeze(self.halofit(zz, kk)),left=0,right=0)*c
				
				#print('halofit',self.halofit(zz,kk))
				#print('kk',kk)
				#print('kval',kval)
				#print('interped halofit',np.interp(kval,kk,np.squeeze(self.halofit(zz, kk)),left=0,right=0))
				#print('pmm',p_mm[0,:])
				#print('max kval', np.max(kval))
				
				#p_gm2_HF[0,:] = b2_HF * np.interp(kval,kk,np.squeeze(self.halofit(zz, kk)),left=0,right=0)*c
				
				#print('len pktable list',np.shape(self.pktable_list))
				#print('c',c)
				#print('z',time.time()-t0)
			
				# Also need to work out handling of magnification bias!
				# I guess since I found kmax=3 is fine from earlier
				# then I should continue to use kmax=3 here.
				# (thus I can use LPT for mag bias term, don't need to use Halofit...)
			j += 1
		#print('pmm',p_mm)
		#print('pgm1_HF',p_gm1_HF)
		p_mm[0,:] = self.halofit(zz, kval)
		p_gm1_HF[0,:] = b1_HF * self.halofit(zz, kval)
		p_gm2_HF[0,:] = b2_HF * self.halofit(zz, kval)
				
		#print('kval',np.shape(kval))
		#print('kk',np.shape(kk))
		p_mm[0,:][(kval > np.max(kk))] = 0
		p_mm[0,:][(kval < np.min(kk))] = 0
		p_gm1_HF[0,:][(kval > np.max(kk))] = 0
		p_gm1_HF[0,:][(kval < np.min(kk))] = 0
		p_gm2_HF[0,:][(kval > np.max(kk))] = 0
		p_gm2_HF[0,:][(kval < np.min(kk))] = 0
		
		p_gg[10,:] = self.halofit(zz, kval) * (-0.5 * alpha_auto) * kval**2.
		p_gm1[4,:] = self.halofit(zz, kval) * (-0.5 * alpha_cross) * kval**2.
		p_gm2[4,:] = self.halofit(zz, kval) * (-0.5 * alpha_cross) * kval**2.

		return(p_gg, p_gm1, p_gm1_HF, p_gm2, p_gm2_HF, p_mm)

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
	#g = chival * np.trapz( 1. / (chivalp) * (chivalp - chival[np.newaxis,:]) * dndz_norm * (1.0/2997.925) * cosmo.H(zvalp).value/cosmo.H(0).value, x =  chivalp, axis=0)
	#H0 = cosmo.H(0).value
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
	zval = np.loadtxt('zval.txt')
	chival = np.loadtxt('chival.txt')
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
	
def do_limber(ell, cosmo, dndz1, dndz2, dndz_xmatch, s1, s2, pk, pars, b1_HF, b2_HF, use_zeff=True, autoCMB=False, Nchi=50, dndz1_mag=None, dndz2_mag=None, normed=False, setup_chi_flag=False, setup_chi_out=None):
	"""Does the Limber integral, returning Cell.
	If use_zeff==True then P(k,z) is assumed to be P(k,zeff).
	On input:
	'ell' is an array of ells for which you want Cell.
	'cosmo' is an astropy Cosmology instance,
	'dndz1' contains two columns (z and dN/dz for sample 1),
	'dndz2' contains two columns (z and dN/dz for sample 2),
	's1' and 's2' are the slope of number counts dlog10N/dm for the 2 samples
	(for magnification bias),
	'pk' is a power spectrum instance (k and P, in Mpc/h units),
	'pars' is a dict with elements 'method' (which can be 'halozeldovich',
	'zeft', or 'halofit'), 'tracer1' giving the bias terms for tracer1
	(Lagrangian b1 and b2 for halozeldovich/zeft or Eulerian b for halofit),
	optionally tracer2, and sn/alpha for halozeldovich/zeft.  Biases, sn and alpha
	must be given as functions of redshift.
	If crossCMB is false returns the object-object auto-correlation while
	if crossCMB is true returns object-CMB(kappa) cross-correlation
	neglecting dndz2.
	'Nchi' gives the number of integration points,
	and 'mag_bias' can be 'all' (all terms), 'only' (mu-mu or mu-kappa only) or 'cross': (2 x mu-galaxy).
	(if you only want the clustering term, set s1=s2=0.4).
	'dndz1_mag' and 'dndz2_mag' are optional additional dn/dz to use in the magnification
	bias term (if you want to use a different dn/dz here and in the clustering term).

	Per test_limber.py, Nchi=1000 is accurate to 0.08% for ell < 1000 and 0.3% for ell < 2000.
	[this depends on how 'spiky' dn/dz is....]

	This is the ``simplified'' Limber code which takes as input P_{NN}, the power spectrum
	of the non-neutrino density fluctuations.  In CAMB-speak, this is P(k) with var1=delta_nonu
	and var2=delta_nonu.  See limber2_complex.py for an explanation of this approximation.
	Empirically this works to ~0.1% at ell > 100; see the limber2 and limber2_complex tests
	(and note how similar they are!)."""
	# Set up cosmological parameters.
	#if 'tracer2' not in pars:
	#	pars['tracer2'] = pars['tracer1']
	# Set up the basic distance-redshift conversions.
	t0 = time.time()
	hub      = cosmo.H0.value / 100.
	Nchi_mag = 25 # Number of sampling points in magnification bias integral
	if setup_chi_flag == False:	
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi(cosmo, dndz1, dndz2, Nchi, Nchi_mag)
	else:
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi_out
		#print 5/0
	#print(time.time()-t0,'zatchi')
	
	fchi1    = np.interp(zatchi(chival),\
		 dndz1[:,0],dndz1[:,1]*cosmo.H(dndz1[:,0]).value,left=0,right=0)
	if not normed:
		fchi1   /= np.trapz(fchi1,x=chival)
	fchi2    = np.interp(zatchi(chival),\
		 dndz2[:,0],dndz2[:,1]*cosmo.H(dndz2[:,0]).value,left=0,right=0)
	if not normed:
		fchi2   /= np.trapz(fchi2,x=chival)
		
	fchi_dndz    = np.interp(zatchi(chival),\
		 dndz_xmatch[:,0],dndz_xmatch[:,1]*cosmo.H(dndz_xmatch[:,0]).value,left=0,right=0)
	if not normed:
		fchi_dndz /= np.trapz(fchi_dndz,x=chival)


	if dndz1_mag is None:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1_mag, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	if dndz2_mag is None:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2_mag, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	# If we're not doing galaxy-galaxy cross-corelations make
	# fchi2 be W(chi) for the CMB.
	
	chistar  = cosmo.comoving_distance(1098.).value * hub

	#if crossCMB:
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
		#print("zeff=",zeff)
	else:
		zeff = -1.0
		
	#print(time.time()-t0,'setup done')

	#print fchi1
	#print fchi2
	# and finally do the Limber integral.
	Nell = len(ell)
	cell_f1f2 = np.zeros( (Nell, 11, Nchi) )
	cell_f1m2 = np.zeros( (Nell, 5, Nchi) )
	cell_m1f2 = np.zeros( (Nell, 5, Nchi) )
	cell_m1m2 = np.zeros( (Nell, 2, Nchi) )
	
	cell_f1fCMB = np.zeros( (Nell, 5, Nchi) )
	cell_m1fCMB = np.zeros( (Nell, 2, Nchi) )
	
	cell_fdndz_fdndz = np.zeros( (Nell, 11, Nchi) )
	cell_fdndz_fCMB  = np.zeros( (Nell, 5, Nchi) )
	cell_f1_fdndz  = np.zeros( (Nell, 11, Nchi) )
	
	'''for i,z in enumerate(pk.zlist):
		chi = chiatz(z)
		kval = (ell+0.5)/chi
		p_gg, p_gm1, p_gm1_HF, p_gm2, p_gm2_HF, p_mm= pk(pars, kval, z, b1_HF, b2_HF)
		np.savetxt('p_gg_%i.txt' % i,p_gg)
		np.savetxt('p_gm_%i.txt' % i,p_gm1)'''
		
	#print('before integral',time.time()-t0)
	
	for i,chi in enumerate(chival):
		if (fchi2[i] != 0) | (mag_kern2[i] != 0):
			kval = (ell+0.5)/chi
			
			if i == 0:
				p_gg, p_gm1, p_gm1_HF, p_gm2, p_gm2_HF, p_mm= pk(pars, kval, 1.75, b1_HF, b2_HF)
				np.savetxt('kval.txt',kval)
				np.savetxt('pgg.txt',p_gg)
				np.savetxt('pgm.txt',p_gm1)
				t0 = time.time()
			
			if use_zeff:
				# Assume a fixed P(k,zeff).
				# This should return both pofk for the galaxy [with the bias]
				# and pofk for matter [no bias, but would include shotnoise] 
				
				p_gg, p_gm1, p_gm1_HF, p_gm2, p_gm2_HF, p_mm = pk(pars, kval, zeff, b1_HF, b2_HF)
			else:
				# Here we interpolate in z.
				zv   = zatchi(chi)
				p_gg, p_gm1, p_gm1_HF, p_gm2, p_gm2_HF, p_mm= pk(pars, kval, zv, b1_HF, b2_HF)
				np.savetxt('p_gm1_%.2f.txt' % zv, p_gm1)
				np.savetxt('kval_%.2f.txt' % zv, kval)
				#print('pgg',time.time()-t0)
				#print('zv',zv)
				#print('pmm',p_mm)
				
				#_, _, _, _, _, p_mm_test = pk(pars, np.logspace(-4,1,100), 0.0, b1_HF, b2_HF)
				#np.savetxt('p_mm_test_z0.txt',p_mm_test)

			#print(time.time()-t0,'pk call')
			#print('chi',chi)


			#print('k',kval)
			#print('p_gg',p_gg)
			

				
			#if not crossCMB:
			f1f2 = fchi1[i]*fchi2[i]/chi**2 * p_gg
			#print('f1f2',f1f2)
			f1m2 = 	fchi1[i]*mag_kern2[i]/chi**2 * p_gm1_HF
			m1f2 = mag_kern1[i]*fchi2[i]/chi**2 *  p_gm2_HF
			m1m2 = mag_kern1[i]*mag_kern2[i]/chi**2 * p_mm
			#print('f1f2',f1f2)
			#print('f1m2',f1m2)
			#print('m1f2',m1f2)
			#print('m1m2',m1m2)
			#print f1f2, f1m2, m1f2, m1m2
			#print p_auto, p_cross1, p_cross2, p_mat
			cell_f1f2[:,:,i] = f1f2.transpose()
			cell_f1m2[:,:,i] = f1m2.transpose()
			cell_m1f2[:,:,i] = m1f2.transpose()
			cell_m1m2[:,:,i] = m1m2.transpose()

			#else:
			f1fCMB = fchi1[i]*fchiCMB[i]/chi**2 * p_gm1
			m1fCMB = mag_kern1[i]*fchiCMB[i]/chi**2 * p_mm
						
			cell_f1fCMB[:,:,i] = f1fCMB.transpose()
			cell_m1fCMB[:,:,i] = m1fCMB.transpose()
			
			fdndz_fdndz = fchi_dndz[i]*fchi_dndz[i]/chi**2 * p_gg			
			#print('fdndz_fdndz',fdndz_fdndz)
			#print('diff',fdndz_fdndz-f1f2)
			
			fdndz_fCMB = fchi_dndz[i]*fchiCMB[i]/chi**2 * p_gm1			
			f1_fdndz = fchi1[i] * fchi_dndz[i]/chi**2 * p_gg
			
			cell_fdndz_fdndz[:,:,i] = fdndz_fdndz.transpose()
			cell_fdndz_fCMB[:,:,i] = fdndz_fCMB.transpose()
			cell_f1_fdndz[:,:,i] = f1_fdndz.transpose()
			#print('kernels',time.time()-t0)

	#cell = np.trapz(cell,x=chival,axis=-1)
	#np.savetxt('diff.txt',(cell_f1f2-cell_fdndz_fdndz).flatten())
	#if not crossCMB:
	cell_f1f2 = np.trapz(cell_f1f2,x=chival,axis=-1)
	cell_f1m2 = np.trapz(cell_f1m2,x=chival,axis=-1)
	cell_m1f2 = np.trapz(cell_m1f2,x=chival,axis=-1)
	cell_m1m2 = np.trapz(cell_m1m2,x=chival,axis=-1)
	cell_m1fCMB = np.trapz(cell_m1fCMB,x=chival,axis=-1)
	cell_f1fCMB = np.trapz(cell_f1fCMB,x=chival,axis=-1)
	cell_fdndz_fdndz = np.trapz(cell_fdndz_fdndz,x=chival,axis=-1)
	cell_fdndz_fCMB = np.trapz(cell_fdndz_fCMB,x=chival,axis=-1)
	cell_f1_fdndz = np.trapz(cell_f1_fdndz,x=chival,axis=-1)
	#print(time.time()-t0,'all done')
	#print('cell_f1f2',cell_f1f2)
	#print('cell_fdndz_fdndz',cell_fdndz_fdndz)

	return( [cell_f1f2, cell_f1m2, cell_m1f2, cell_m1m2] , [cell_f1fCMB,cell_m1fCMB], [cell_fdndz_fdndz], [cell_fdndz_fCMB], [cell_f1_fdndz])
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
