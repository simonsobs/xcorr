from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from settings import *

info['params']['omch2'] = "lambda H0,ombh2,tau,mnu,nnu,num_massive_neutrinos,ns,SN,As,Omegam,b1: Omegam * (H0/100.)**2. - ombh2 - (mnu/93.14)"
info['params']['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Set up log file

logfile = output_name + '_minimize.log'

# Prevent over-wriring
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

info['sampler'] = {'minimize':  minimize_block}

np.random.seed(123+rank)
products = run(info)
