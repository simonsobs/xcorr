from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from settings import *

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

import time

# Need to get the likelihood first
model = get_model(info)

t0 = time.time()
like = model.loglike({'logA': 2.984, 'Omegam': 0.3180, 'logSN': -6.7467, 'b1': 2.2475, 'shift': -0.0256, 'width': 0.9471, 's_wise': 0.6505,'alpha_cross': 0.0, 'alpha_auto': 0.0})
print(time.time()-t0)
