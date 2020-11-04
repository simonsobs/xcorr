from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from settings import *
import yaml

info['params']['omch2'] = "lambda H0,ombh2,tau,mnu,nnu,num_massive_neutrinos,ns,SN,As,Omegam,b1: Omegam * (H0/100.)**2. - ombh2 - (mnu/93.14)"
info['params']['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Set up log file

logfile = output_name + '_evaluate.log'

#if rank == 0:
#	time.sleep(50)
#	print(os.path.exists(logfile))
#	#print(5/0)
#	if os.path.exists(logfile):
#		raise Exception('Logfile already exists! Please delete it before running your chains')

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.terminal.flush()
        self.log.flush()  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

np.random.seed(123+rank)
#products = run(info)

info['sampler'] = 'evaluate'
params_yaml = yaml.load(open('input/params.yaml','r'))
#info['params'] = params_yaml
print(info['params'])

import time

# Need to get the likelihood first
model = get_model(info)

logA_for_test = 3.05
Omegam_for_test = 0.3
b1_for_test = 1.0

t0 = time.time()
like = model.loglike({'logA': logA_for_test, 'Omegam': Omegam_for_test, 'b1': b1_for_test})
print(time.time()-t0)

#print(5/0)
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

lim_cross, lim_auto = get_angular_power_spectra(halofit,
	model.likelihood.theory,
	s_wise,
	Omegam_for_test, b1_for_test, 0,
	0,0,0,0, 0,0, 0)
	
lim_bin_cross = bin(all_ell,lim_cross,lmin[:high_ind],lmax[:high_ind])
# Correct for namaster binning
#lim_bin_cross = lim_bin #* clkg_corr[low_ind:high_ind]

lim_bin_auto = bin(all_ell,lim_auto,lmin[:high_ind],lmax[:high_ind])

print('lim_bin_auto',lim_bin_auto)
print('SN',info['params']['SN'])

#+info['params']['SN']

np.savetxt('input/clkg_unbinned.txt',lim_cross)
np.savetxt('input/clgg_unbinned.txt',lim_auto + info['params']['SN'])

cl_input = np.loadtxt('/global/cscratch1/sd/akrolew/unwise-hod/8192-5000/HOD16/QA_Cl/GREEN/clkg_binned_dl50_noiseless_for_chains.txt')

np.savetxt('input/clgg.txt',np.array([cl_input[0,:high_ind],lim_bin_auto,0.1*lim_bin_auto]))
np.savetxt('input/clkg.txt',np.array([cl_input[0,:high_ind],lim_bin_cross,0.1*lim_bin_cross]))