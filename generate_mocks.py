#! /usr/bin/env python
# Code to generate GRF mocks, in healpix format

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
import healpy as hp
import copy
import pymaster as nmt
from scipy.interpolate import interp1d
import sharedmem
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')
parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)
parser.add_argument('--nproc', dest='nproc', type=int, help='Optional number of processes to run.', required=False)

### END OF PARSER ###



def enrich_params(params):
   """
   Infers the unspecified parameters from the parameters provided and
   updates the parameter dictionary accordingly.
   :param :
   :return :
   """
   
   params = copy.deepcopy(params)
   params['spins'] = np.array(params['spins'])
   # The ell array starts at 0, so the number of ells is lmax + 1
   if 'lmax' in params:
     params['nell'] = params['lmax']+1
   # coutning probes, fields, and their power spectra
   params['nprobes'] = len(params['probes'])
   params['ncls_probes'] = params['nprobes'] * (params['nprobes'] + 1) // 2
   params['nScalProbes'] = np.sum(params['spins']==0).astype('int')
   params['nTensProbes'] = np.sum(params['spins']==2).astype('int')
   params['nFields'] = params['nScalProbes'] + params['nTensProbes'] * 2
   params['ncls_fields'] = params['nFields'] * (params['nFields'] + 1) // 2
   return params



#def read_cl(params):
#   """
#   Reads in all the auto and cross power spectra needed to construct the set of
#   correlated maps.
#   It also multiplies the theoretical power spectra by the HEALPix pixel window
#   functions and if the flag tempbeam is set it also multiplies them by the CMB
#   beam window function.
#   This is in order to test the pixel and beam window deconvolutions.
#   :param :
#   :return cls: 3D array with 0. and 1. axis denoting the number of the power spectrum and the
#   3. axis is the power spectrum belonging to this index
#   """
#
#   logger.info('Setting up cl array.')
#   cls = np.zeros((params['ncls_fields'], params['nell']))
#   print(cls.shape)
#   logger.info('Cl array shape = {}.'.format(cls.shape))
#
#   # Convert the cls for the probes into the cls for the fields:
#   # for tensor fields, the input power spectra are for only for the E-modes, and assume the B-modes aare zero.
#   # Here, we add the B-modes power, set to zero or to the noise power spectrum
#   k = 0 # index over probe power spectra
#   j = 0 # index over field power spectra
#   # loop over probes, in row-major order
#   for i, probe1 in enumerate(params['probes']):
#      for ii, probe2 in enumerate(params['probes'][i:], start=i):
#
#         logger.info('Reading cls for probe1 = {} and probe2 = {}.'.format(probe1, probe2))
#         path2cls = params['path2cls'][k]
#         data = np.genfromtxt(path2cls)
#         logger.info('Read {}.'.format(path2cls))
#         
#         # check that the lmax requested is in the input file
#         if data[-1,0]<params['lmax']:
#            raise ValueError('The lmax required is higher than in the cl input file')
#         # interpolate the input cl, in case it was not given at each ell, 
#         # or did not start at ell=0.
#         fcl = interp1d(data[:params['nell'],0], data[:params['nell'], 1], kind='linear', bounds_error=False, fill_value=0.)
#         cls_temp = fcl(np.arange(params['nell']))
#         #cls_temp = data[:params['nell'], 1]
#
#         cls[j, :] = cls_temp
#         # fFor spin2-spin2, add the spectra: E1B2, E2B1, B1B2
#         #!!! Warning: are we missing some power spectra here?
#         if params['spins'][i] == 2 and params['spins'][ii] == 2:
#            cls[j+1, :] = np.zeros_like(cls_temp)
#            cls[j+2, :] = np.zeros_like(cls_temp)
#            j += 3
#         # For spin0-spin2, add the TB power spectrum
#         elif params['spins'][i] == 2 and params['spins'][ii] == 0 or params['spins'][i] == 0 and params['spins'][ii] == 2:
#            cls[j+1, :] = np.zeros_like(cls_temp)
#            j += 2
#         # for spin0-spin0, nothing to add
#         else:
#            j += 1
#
#         k += 1
#
#   return cls
#






















def read_cl(params):
   """
   Reads in all the auto and cross power spectra needed to construct the set of
   correlated maps.
   It also multiplies the theoretical power spectra by the HEALPix pixel window
   functions and if the flag tempbeam is set it also multiplies them by the CMB
   beam window function.
   This is in order to test the pixel and beam window deconvolutions.
   :param :
   :return cls: 3D array with 0. and 1. axis denoting the number of the power spectrum and the
   3. axis is the power spectrum belonging to this index
   """

   # 2d matrix of cls for the fields
   clFieldsMat = np.zeros((params['nFields'], params['nFields'], params['nell']))

   # Convert the cls for the probes into the cls for the fields:
   # for tensor fields, the input power spectra are for only for the E-modes, and assume the B-modes aare zero.
   # Here, we add the B-modes power, set to zero or to the noise power spectrum
   iProbePair = 0
   iField = 0
   jField = 0
   # loop over probes, in row-major order
   for iProbe1, probe1 in enumerate(params['probes']):

      if params['path2noisecls'] is not None:
         path = params['path2noisecls'][iProbe1]
         logger.info('Read noise power from {}.'.format(path))
         data = np.genfromtxt(path)
         # check that the lmax requested is in the input file
         if data[-1,0]<params['lmax']:
            raise ValueError('The lmax required is higher than in the cl input file')
         # interpolate the input cl, in case it was not given at each ell, 
         # or did not start at ell=0.
         fcl = interp1d(data[:params['nell'],0], data[:params['nell'], 1], kind='linear', bounds_error=False, fill_value=0.)
         noisecls_temp = fcl(np.arange(params['nell']))
      else:
         noisecls_temp = np.zeros(params['nell'])

      for iProbe2, probe2 in enumerate(params['probes'][iProbe1:], start=iProbe1):
         logger.info('Reading cls for probe1 = {} and probe2 = {}.'.format(probe1, probe2))

         path = params['path2cls'][iProbePair]
         logger.info('Read signal power from {}.'.format(path))
         data = np.genfromtxt(path)
         # check that the lmax requested is in the input file
         if data[-1,0]<params['lmax']:
            raise ValueError('The lmax required is higher than in the cl input file')
         # interpolate the input cl, in case it was not given at each ell, 
         # or did not start at ell=0.
         fcl = interp1d(data[:params['nell'],0], data[:params['nell'], 1], kind='linear', bounds_error=False, fill_value=0.)
         cls_temp = fcl(np.arange(params['nell']))

         # Move to the next pair of probes in the input cls
         iProbePair += 1

         # If spin0 - spin0, just copy the power spectrum
         if params['spins'][iProbe1]==0 and params['spins'][iProbe2]==0:
            clFieldsMat[iField, jField, :] = cls_temp + (iProbe1==iProbe2) * noisecls_temp
            jField += 1
         # If spin0 - spin2, add the TB power spectrum
         if params['spins'][iProbe1]==0 and params['spins'][iProbe2]==2:
            clFieldsMat[iField, jField, :] = cls_temp
            #clFieldsMat[iField, jField+1, :] = np.zeros_like(cls_temp)
            jField += 2
         # If spin2 - spin2, add the E1B2 and B1B2 power spectra
         if params['spins'][iProbe1]==2 and params['spins'][iProbe2]==2:
            clFieldsMat[iField, jField, :] = cls_temp + (iProbe1==iProbe2) * noisecls_temp
            #clFieldsMat[iField, jField+1, :] = np.zeros_like(cls_temp)
            clFieldsMat[iField+1, jField+1, :] = (iProbe1==iProbe2) * noisecls_temp
            jField += 2

      # set the row index to the new line, for the new probe1
      if params['spins'][iProbe1]==0:
         iField += 1
      if params['spins'][iProbe1]==2:
         iField += 2
      # set the column index to the row index,
      # since we start on the diagonal
      jField = iField


   # Convert the 2d matrix into a 1d array,
   # in row-major order
   cls = np.zeros((params['ncls_fields'], params['nell']))
   iFieldPair = 0
   for iField in range(params['nFields']):
      for jField in range(iField, params['nFields']):
         cls[iFieldPair,:] = clFieldsMat[iField, jField, :]
         iFieldPair += 1

   #print(cls)


   return cls
































def generate_map(params, cls, iRea):
   """
   Convenience method for calculating the signal and noise cls for
   a given mock realization. This is a function that can be pickled and can be thus
   used when running the mock generation in parallel using multiprocessing pool.
   :param realis: number of the realisation to run
   :param noise: boolean flag indicating if noise is added to the mocks
   noise=True: add noise to the mocks
   noise=False: do not add noise to the mocks
   :param probes: list of desired probes to run the mock for
   :param maskmat: matrix with the relevant masks for the probes
   :param clparams: list of dictionaries with the parameters for calculating the
   power spectra for each probe
   :return cls: 3D array of signal and noise cls for the given realisation,
   0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
   to this configuration
   :return noisecls: 3D array of noise cls for the given realisation,
   0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
   to this configuration
   :return tempells: array of the ell range of the power spectra
   """

   logger.info('Generate GRF realization: {}'.format(iRea))

   # Generate the GRF maps
   maps = nmt.synfast_spherical(params['nside'], cls, spin_arr=params['spins'], seed=iRea, \
         beam=None)


#   if params['nspin2'] > 0:
#      raise NotImplementedError()
#      logger.info('Spin 2 fields present. Reordering maps.')
#      reordered_maps = reorder_maps(maps)
#
#      if params['nspin2'] == 1:
#         assert np.sum([np.all(maps[i] == reordered_maps[i]) for i in range(len(maps))]) == len(maps), \
#               'Something went wrong with map reordering.'
#   else:
#      logger.info('No spin 2 fields. Keeping map ordering.')
#   reordered_maps = copy.deepcopy(maps)



   # Saving the scalar maps
   for iScalProbe in range(params['nScalProbes']):
      iMap = iScalProbe
      path = params['path2outputdir'] + '/' + params['probes'][iScalProbe] + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, maps[iMap], overwrite=True)


   # Saving the tensor maps
   iMap = params['nScalProbes']
   for iTensProbe in range(params['nScalProbes'], params['nprobes']):

      # save Q map as gamma 1
      path = params['path2outputdir'] + '/' + params['probes'][iTensProbe] + '_gamma1' + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, maps[iMap], overwrite=True)

      # save -U map as gamma 2
      path = params['path2outputdir'] + '/' + params['probes'][iTensProbe] + '_gamma2' + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, -maps[iMap+1], overwrite=True)
      
      # jump 2 by 2 for tensor maps
      iMap += 2

   return maps




#def reorder_maps( maps):
#   '''This seems to take a list of maps of the form
#   (T, T, T, Q, U, Q, U, Q, U)
#   and turn it into
#   (T, T, T, Q, Q, Q, U, U, U).
#   This assumes that the spin 0 fields are first,
#   followed by the spin 2 fields.
#   '''
#   raise NotImplementedError()
#
#   logger.info('Reordering maps.')
#
#   tempmaps = copy.deepcopy(maps)
#
#   spins = np.array(params['spins'])
#   nspin2 = np.sum(spins == 2)
#   ind = np.where(spins == 2)[0]
#   min_ind = np.amin(ind)
#   tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
#   tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]
#
#   return tempmaps
#



if __name__ == '__main__':

   args = parser.parse_args()

   # read parameters from yaml file
   logger.info('Read config from {}.'.format(args.path2config))
   config = yaml.load(open(args.path2config))
   # add inferred parameters from yaml file
   params = enrich_params(config)

   # create output directory if needed
   if not os.path.isdir(config['path2outputdir']):
      os.makedirs(config['path2outputdir'])

   # Only curved (healpix) maps for now
   if config['mode'] <> 'curved':
      raise NotImplementedError()

   # Only implemented noise free maps for now
   if config['path2noisecls'] is not None:
      logger.info('Generating noisy mocks.')
   else:
      logger.info('Generating noise-free mocks.')
   

   # read the input theory cl
   # modify to add the noise power spectrum to the autos
   clArray = read_cl(params)

   # Number of processes to run on
   if args.nproc is None:
      nProc = 1
   else:
      nProc = args.nproc
   logger.info('Running on '+str(nProc)+' processes')

   # generate and save all the mock maps
   #maps = generate_map(params, clArray, 0)
   nRea = config['nrealiz']
   with sharedmem.MapReduce(np=nProc) as pool:
      f = lambda iRea:  generate_map(params, clArray, iRea)
      pool.map(f, range(nRea))


def run(pathConfig):

   # read parameters from yaml file
   logger.info('Read config from {}.'.format(pathConfig))
   config = yaml.load(open(pathConfig))
   # add inferred parameters from yaml file
   params = enrich_params(config)

   # create output directory if needed
   if not os.path.isdir(config['path2outputdir']):
      os.makedirs(config['path2outputdir'])

   # Only curved (healpix) maps for now
   if config['mode'] <> 'curved':
      raise NotImplementedError()

   # Only implemented noise free maps for now
   if config['path2noisecls'] is not None:
      logger.info('Generating noisy mocks.')
   else:
      logger.info('Generating noise-free mocks.')


   # read the input theory cl
   # modify to add the noise power spectrum to the autos
   clArray = read_cl(params)

   # Number of processes to run on
   if config['nProc'] is None:
      nProc = 1
   else:
      nProc = config['nProc']
   logger.info('Running on '+str(nProc)+' processes')

   # generate and save all the mock maps
   #maps = generate_map(params, clArray, 0)
   nRea = config['nrealiz']
   with sharedmem.MapReduce(np=nProc) as pool:
      f = lambda iRea:  generate_map(params, clArray, iRea)
      pool.map(f, range(nRea))


