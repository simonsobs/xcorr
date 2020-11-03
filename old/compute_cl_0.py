#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
from astropy.io import fits
from time import time
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
# Import the NaMaster python wrapper
import pymaster as nmt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Compute cls using NaMaster.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

parser.add_argument('--pathOutputDir', dest='pathOutputDir', type=str, required=False)
parser.add_argument('--mode', dest='mode', type=str, required=False)
parser.add_argument('--lMinCl', dest='lMinCl', type=int, required=False)
parser.add_argument('--lMaxCl', dest='lMaxCl', type=int, required=False)
parser.add_argument('--nBinsCl', dest='nBinsCl', type=int, required=False)
parser.add_argument('--binSpacing', dest='binSpacing', type=str, required=False)
parser.add_argument('--binWeighting', dest='binWeighting', type=str, required=False)
parser.add_argument('--nameOutputClFile', dest='nameOutputClFile', type=str, required=False)
parser.add_argument('--spins', dest='spins', type=list, required=False)
parser.add_argument('--pixWindow', dest='pixWindow', type=list, required=False)
parser.add_argument('--pathMap1', dest='pathMap1', type=str, required=False)
parser.add_argument('--pathMask1', dest='pathMask1', type=str, required=False)
parser.add_argument('--pathMap2', dest='pathMap2', type=str, required=False)
parser.add_argument('--pathMask2', dest='pathMask2', type=str, required=False)
parser.add_argument('--pathCl12', dest='pathCl12', type=str, required=False)
parser.add_argument('--pathCl11', dest='pathCl11', type=str, required=False)
parser.add_argument('--pathCl22', dest='pathCl22', type=str, required=False)
parser.add_argument('--pathNl12', dest='pathNl12', type=str, required=False)
parser.add_argument('--pathNl11', dest='pathNl11', type=str, required=False)
parser.add_argument('--pathNl22', dest='pathNl22', type=str, required=False)


### END OF PARSER ###




def read_cl(path):
   logger.info('Reading cl from '+path)
   data = np.genfromtxt(path)
   # check that the lmax requested is in the input file
   if data[-1,0]<config['lMaxCl']:
      raise ValueError('The lmax required is higher than in the cl input file')
   # interpolate the input cl, in case it was not given at each ell, 
   # or did not start at ell=0.
   fCl = interp1d(data[:,0], data[:, 1], kind='linear', bounds_error=False, fill_value=0.)
   return fCl




def read_all_cl(config):
   #!!! Only implemented for scalar fields   
   
   # Read signal power spectra
   if config['pathCl12'] is not None:
      fCl12 = read_cl(config['pathCl12'])
   else:
      fCl11 = lambda l: 0.
   #
   if config['pathCl11'] is not None:
      fCl11 = read_cl(config['pathCl11'])
   else:
      fCl11 = lambda l: 0.
   #
   if config['pathCl22'] is not None:
      fCl22 = read_cl(config['pathCl22'])
   else:
      fCl22 = lambda l: 0.

   # Read noise power spectra
   if config['pathNl12'] is not None:
      fNl12 = read_cl(config['pathNl12'])
   else:
      fNl12 = lambda l: 0.
   #
   if config['pathNl11'] is not None:
      fNl11 = read_cl(config['pathNl11'])
   else:
      fNl11 = lambda l: 0.
   #
   if config['pathNl22'] is not None:
      fNl22 = read_cl(config['pathNl22'])
   else:
      fnl22 = lambda l: 0.

   # add signal and noise
   f12 = lambda l: fCl12(l) + fNl12(l)
   f11 = lambda l: fCl11(l) + fNl11(l)
   f22 = lambda l: fCl22(l) + fNl22(l)

   return f12, f11, f22 

















def get_wsp(f1, f2, mask1, mask2, b):
   '''Computes the workspace with caching,
   ie it saves the workspace and does not recompute it
   if asked with the same masks.
   Warning: it does not check that the bins are identical.
   '''
   # if first call ever, set up the cache dictionary
   if not hasattr(get_wsp, "cache"):
      get_wsp.cache = []

   # check to see if this workspace was already computed   
   for i in range(len(get_wsp.cache)):
      if np.all(np.array(get_wsp.cache[i][:-1])==np.array([mask1, mask2])):
         return get_wsp.cache[i][-1]

   # if this calculation was not done before, do it
   logger.info('Computing workspace element.')
   tStart = time()
   wsp = nmt.NmtWorkspace()
   wsp.compute_coupling_matrix(f1, f2, b)
   get_wsp.cache.append([mask1, mask2, wsp])
   tStop = time()
   logger.info('took '+str((tStop-tStart)/60.)+' min')
   return wsp







if __name__ == '__main__':

   args = parser.parse_args()

   # read parameters from yaml file
   config = yaml.load(open(args.path2config))
   logger.info('Read config from {}.'.format(args.path2config))
   
   # Overwrite options in yaml file with optional input,
   # if needed
   argDict = args.__dict__
   for key in argDict:
      if config.has_key(key) and argDict[key] is not None:
         config[key] = argDict[key] 

   # Check that output directory exists, create it if not
   pathOutputDir = config['pathOutputDir']
   if not os.path.isdir(pathOutputDir):
      try:
         os.makedirs(pathOutputDir)
         logger.info('Created directory {}.'.format(pathOutputDir))
      except:
         logger.info('Directory {} already exists.'.format(pathOutputDir))
         pass


   # Only curved (healpix) maps for now
   if config['mode'] <> 'curved':
      raise NotImplementedError()

   elif config['mode'] == 'curved':
      import healpy as hp

      spin1, spin2 = config['spins']
      logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

      logger.info('Read map1 from {}.'.format(config['pathMap1']))
      map1 = hp.read_map(config['pathMap1'], field=None)
      nSide1 = hp.pixelfunc.get_nside(map1)

      logger.info('Read map2 from {}.'.format(config['pathMap2']))
      map2 = hp.read_map(config['pathMap2'], field=None)
      nSide2 = hp.pixelfunc.get_nside(map2)

      if nSide1<>nSide2:
         raise NotImplementedError()
      else:
         nSide = nSide1

      if config['pathMask1'] is None:
         mask1 = np.ones_like(map1)
      else:
         mask1 = hp.read_map(config['pathMask1'])
         mask1 = hp.ud_grade(mask1, nSide)
      logger.info('Read mask1 from {}.'.format(config['pathMask1']))
      
      
      if config['pathMask2'] is None:
         mask2 = np.ones_like(map2)
      else:
         mask2 = hp.read_map(config['pathMask2'])
         mask2 = hp.ud_grade(mask2, nSide)
      logger.info('Read mask2 from {}.'.format(config['pathMask2']))


      pixWin1, pixWin2 = config['pixWindow']
      if pixWin1==1:
         pixWin1 = hp.sphtfunc.pixwin(nSide1, pol=False)
      else:
         pixWin1 = None
      if pixWin2==1:
         pixWin2 = hp.sphtfunc.pixwin(nSide2, pol=False)
      else:
         pixWin2 = None


      if config['binSpacing']=='lin':
         lEdges = np.linspace(config['lMinCl'], config['lMaxCl'], config['nBinsCl']+1)
      if config['binSpacing']=='log':
         lEdges = np.logspace(np.log10(config['lMinCl']), np.log10(config['lMaxCl']), config['nBinsCl']+1)
      lEdges = lEdges.astype('int')
      

      # choose the maximum multipole in map to compute coupling matrix
      lMaxMap = min(3.*nSide-1, config['lMaxMap'])
      # Generate the ell bins
      #ells = np.arange(config['lMaxCl']+1)
      ells = np.arange(lMaxMap+1)
      weights = np.zeros_like(ells, dtype='float64')
      bpws = -1*np.ones_like(ells)
      # Careful: bins start at zero
      for i in range(config['nBinsCl']):
         bpws[lEdges[i]:lEdges[i+1]] = i
         if config['binWeighting']=='uniform':
            weights[lEdges[i]:lEdges[i+1]] = 1.
         elif config['binWeighting']=='numModes':
            weights[lEdges[i]:lEdges[i+1]] = 2. * ells[lEdges[i]:lEdges[i+1]] + 1.

      b = nmt.NmtBin(nSide, bpws=bpws, ells=ells, weights=weights, lmax=lMaxMap)
      # The effective sampling rate for these bandpowers can be obtained calling:
      ells_decoupled = b.get_effective_ells()

      # Only spins 0 for now
      if spin1<>0 or spin2<>0:
         raise RaiseNotImplementedError()

#!!! this assumes that gamma1 and gamma2 are part of one healpix map,
# where the T field is empty. Hence indices 1 and 2 rather than 0 and 1
      if spin1==2 and spin2==0:
         raise NotImplementedError()
         emaps1 = [map1[1], map1[2]]
         emaps2 = [map2]
      elif spin1==0 and spin2==2:
         raise NotImplementedError()
         emaps1 = [map1]
         emaps2 = [map2[1], map2[2]]
      elif spin1==2 and spin2==2:
         raise NotImplementedError()
         emaps1 = [map1[1], map1[2]]
         emaps2 = [map2[1], map2[2]]
      else:
         emaps1 = [map1]
         emaps2 = [map2]


      # Define fields
      f1 = nmt.NmtField(mask1, emaps1, purify_b=False, beam=pixWin1)
      f2 = nmt.NmtField(mask2, emaps2, purify_b=False, beam=pixWin2)

      # initialize the workspace
      wsp = get_wsp(f1, f2, mask1, mask2, b)

      # Compute pseudo-Cls
      logger.info('Computing (coupled) pseudo-cls')
      cl_coupled = nmt.compute_coupled_cell(f1, f2)
      # Uncoupling pseudo-Cls
      # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
      # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
      logger.info('Decoupling cls.')
      cl_decoupled = wsp.decouple_cell(cl_coupled)
      # array of measured cl to be saved to file
      cl_out = np.vstack((ells_decoupled, cl_decoupled))
      # Save measured cl to file
      path = pathOutputDir + "/" + config['nameOutputClFile'] + "_measuredcl.txt"
      np.savetxt(path, cl_out.T)
      logger.info('Written measured cls to {}.'.format(path))

      
      # read the theory cl if requested
      if config['doTheory']:
         # read the theory Cl
         fClTheory, fCl11, fCl22 = read_all_cl(config)

         
         # couple, bin and decouple the theory
         logger.info('Theory cl: couple, bin, decouple')
         clTheory = [fClTheory(ells)]
         # couple with the mask, bin, then decouple
         clTheory_decoupled = wsp.decouple_cell(wsp.couple_cell(clTheory))
         # array of measured cl to be saved to file
         clTheory_out = np.vstack((ells_decoupled, clTheory_decoupled))
         # save it to file
         path = pathOutputDir + "/" + config['nameOutputClFile'] + "_theorycl.txt"
         np.savetxt(path, clTheory_out.T)
         logger.info('Written theory cls to {}.'.format(path))

        
         # Theory Gaussian covariance
         if config['doCov']:
            logger.info('Theory cov')
            tStart = time()
            cw = nmt.NmtCovarianceWorkspace()
            logger.info('Computing covariance workspace')
            tStart = time()
            cw.compute_coupling_coefficients(f1, f2, flb1=f1, flb2=f2)
            tStop = time()
            logger.info('cov took '+str((tStop-tStart)/60.)+' min')
            cl12 = fClTheory(np.arange(cw.wsp.lmax+1))
            cl11 = fCl11(np.arange(cw.wsp.lmax+1))
            cl22 = fCl22(np.arange(cw.wsp.lmax+1))
            # !!! Syntax valid for spin 0 fields only
            cov = nmt.gaussian_covariance(cw, 
                                          spin1, spin2, spin1, spin2,
                                          [cl11], [cl12], [cl12], [cl22],
                                          wa=wsp, wb=wsp)
            # save it to file
            path = pathOutputDir + "/" + config['nameOutputClFile'] + "_theorycov.txt"
            np.savetxt(path, cov)
            logger.info('Written theory cov to {}.'.format(path))
            


   # plot to check if requested
   if config['plot']:# and config['doTheory'] and config['doCov']:


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #

      ax.errorbar(ells_decoupled, cl_decoupled[0], yerr=np.sqrt(np.diag(cov)), c='b', label=r'measured, decoupled')
      ax.errorbar(ells_decoupled, -cl_decoupled[0], yerr=np.sqrt(np.diag(cov)), c='r')
      ax.plot(ells_decoupled, clTheory_decoupled[0], '.', label=r'theory, binned & decoupled')
      ax.plot(ells, fClTheory(ells), label=r'theory')
      #
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=1)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$c_\ell$')


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.errorbar(ells_decoupled, cl_decoupled[0]/clTheory_decoupled[0] -1., yerr=np.sqrt(np.diag(cov))/clTheory_decoupled[0], c='b')
      ax.axhline(0., color='k')
      #
      ax.set_xlabel(r'$\ell$')
      #ax.set_ylabel(r'$c_\ell^\text{measured} / c_\ell^\text{th binned} - 1$')



      plt.show()
