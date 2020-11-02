from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from settings import *

# Set up log file

logfile = output_name + '.log'

# Prevent over-writing
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

sampled_params = np.array(['logA','Omegam','logSN','b1','s_wise', 'shift', 'width','alpha_cross','alpha_auto'])

minimum = np.loadtxt(output_name + '_minimize.minimum.txt')
x = minimum[2:2+len(sampled_params)]
x = np.array([x[0], x[1], x[2], x[3], x[4], x[7], x[8], x[5], x[6]])

################################ FISHER FOR COVMAT ##############################################

# Wrapper around clkg likelihood. We need to define
# this function to pass a *different* function to info
# for the Fisher calculation!
def wrap_clkg_likelihood(s_wise=0.4,
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
	#Function for the likelihood.
	# Using \Delta z = 0.1 doesn't change the runtime
	# kmax = 3.0 allows for faster runtime without sacrificing accuracy
	# hubble units and k_hunit need to be True, otherwise output is in Mpc
	# I want delta_nonu power spectrum
	return clkg_likelihood(s_wise,b1,b2,bs,alpha_auto,alpha_cross,alpha_matter,SN,shift,width,_theory)

# Now we want to use the "evaluate" sampler
info['likelihood']['clkg_likelihood']['external'] = wrap_clkg_likelihood
info['sampler'] = 'evaluate'

# Need to get the likelihood first
model = get_model(info)
like = model.loglike({'logA': x[0], 'Omegam': x[1], 'logSN': x[2], 'b1': x[3],'s_wise':x[4], 'shift': x[5], 'width': x[6], 'alpha_cross': x[7], 'alpha_auto': x[8]})


# Compute the derivatives
cov_derivatives = []
cl_derivatives = []


for i in range(len(sampled_params)):
	param_val = x[i]
	param = sampled_params[i]

	upper = 1.1
	lower = 0.9
	delta = upper -lower

	like_dict = {}
	for j in range(len(sampled_params)):
		if i == j:
			like_dict[sampled_params[j]] = x[j]*upper
		else:
			like_dict[sampled_params[j]] = x[j]
			
	#like_dict['alpha_cross'] = x[7]
	#like_dict['alpha_auto'] = x[8]

	#0 = x[4]
	#0 = x[5]
	#0 = x[5]
	#like_dict['alpha_auto'] = x[6]
	#like_dict['alpha_matter'] = x[7]

	like = model.loglike(like_dict)

	total_cov1 = covariance(model, like_dict)
	
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

	cross1, auto1 = get_angular_power_spectra(halofit,
		model.likelihood.theory,
		like_dict['Omegam'],
		like_dict['b1'],
		1.0,
		1.0,
		like_dict['alpha_auto'],
		like_dict['alpha_cross'],
		0.0,
		10**like_dict['logSN'],
		like_dict['shift'],
		like_dict['width'])

	auto1 += 10**like_dict['logSN']

	cl1 = np.concatenate((cross1[low_ell_kg:high_ell],auto1[low_ell_gg:high_ell]),axis=0)

	like_dict = {}
	for j in range(len(sampled_params)):
		if i == j:
			like_dict[sampled_params[j]] = x[j]*lower
		else:
			like_dict[sampled_params[j]] = x[j]

	#0 = x[4]
	#0 = x[5]
	#0 = x[5]
	#like_dict['alpha_auto'] = x[6]
	#like_dict['alpha_matter'] = x[7]
	
	#like_dict['alpha_cross'] = x[7]
	#like_dict['alpha_auto'] = x[8]

	like = model.loglike(like_dict)

	total_cov2 = covariance(model, like_dict)

	
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

	cross2, auto2 = get_angular_power_spectra(halofit,
		model.likelihood.theory,
		like_dict['Omegam'],
		like_dict['b1'],
		1.0,
		1.0,
		like_dict['alpha_auto'],
		like_dict['alpha_cross'],
		0.0,
		10**like_dict['logSN'],
		like_dict['shift'],
		like_dict['width'])

	auto2 += 10**like_dict['logSN']

	cl2 = np.concatenate((cross2[low_ell_kg:high_ell],auto2[low_ell_gg:high_ell]),axis=0)

	#if param == 'As':
	#	param_val *= 1e9
	#if param == 'SN':
	#	param_val *= 1e7
	#cov_derivatives.append( (total_cov1-total_cov2)/(delta * param_val))
	#cl_derivatives.append( (cl1-cl2)/(delta * param_val))

	ind_cov_derivative = (total_cov1-total_cov2)/(delta * param_val)
	ind_cl_derivative = (cl1-cl2)/(delta * param_val)
	ind_cov_derivative[np.isinf(ind_cov_derivative)] = 0
	ind_cl_derivative[np.isinf(ind_cl_derivative)] = 0
	cov_derivatives.append( ind_cov_derivative)
	#print(i,'ind_cov_derivative shape',np.shape(ind_cov_derivative))
	cl_derivatives.append( ind_cl_derivative)

#print('cov_derivative shape',np.shape(cov_derivatives))


# Get the fiducial covariance	
like_dict = {}
for j in range(len(sampled_params)):
	like_dict[sampled_params[j]] = x[j]


like = model.loglike(like_dict)
cov = covariance(model, like_dict)
invcov = np.linalg.inv(cov)

# Define fisher matrix
fisher = np.zeros((len(sampled_params),len(sampled_params)))

# Fisher calculation
# For reference see https://arxiv.org/pdf/0906.0664.pdf
for i in range(len(sampled_params)):
	for j in range(len(sampled_params)):
		Mab = np.outer(cl_derivatives[i],cl_derivatives[j]) + np.outer(cl_derivatives[i],cl_derivatives[j]).T
		fisher[i,j] = 0.5 * np.trace(np.dot(invcov, np.dot(cov_derivatives[i], np.dot(invcov, cov_derivatives[j]) ) ) + np.dot(invcov, Mab))



shift_width_covariance = np.array([[0.00120118, 0.00210338],
       [0.00210338, 0.00513655]])


prior_dndz = np.zeros_like(fisher)
shift_ind = np.where(sampled_params=='shift')[0][0]
print(sampled_params)
width_ind = np.where(sampled_params=='width')[0][0]
prior_dndz[shift_ind:width_ind+1, shift_ind:width_ind+1] = np.linalg.inv(shift_width_covariance)

fisher_tot = fisher + prior_dndz #+ prior_alpha
print(np.shape(fisher_tot))
print(len(sampled_params))

# Get the covariance matrix
#cov = np.linalg.inv(fisher)
np.savetxt(output_name + '.fisher',fisher_tot)
#cov = np.zeros((9,9))
cov = np.linalg.inv(fisher_tot)

#cov[7,7] = 1./0.1**2.
#cov[8,8] = 1./0.1**2.
#np.savetxt(output_name + '.covmat',cov)
cov = np.loadtxt(output_name + '.covmat')

################################ CHAINS ##############################################

# You need to make a *NEW* function for the next run of cobaya
# So this is a dummy funcion calling my old likelihood!
def wrap_clkg_likelihood(s_wise=0.4,
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
	#Function for the likelihood.
	# Using \Delta z = 0.1 doesn't change the runtime
	# kmax = 3.0 allows for faster runtime without sacrificing accuracy
	# hubble units and k_hunit need to be True, otherwise output is in Mpc
	# I want delta_nonu power spectrum
	out = clkg_likelihood(s_wise,b1,b2,bs,alpha_auto,alpha_cross,alpha_matter,SN,shift,width,_theory)
	#print('out',out)
	return out
	
sampled_params = np.array(['logA','Omegam','logSN','b1','s_wise', 'shift', 'width','alpha_cross','alpha_auto'])


info['likelihood']['clkg_likelihood']['external'] = wrap_clkg_likelihood
info['sampler'] = {'mcmc': {'learn_proposal': True, 'oversample': True,'learn_proposal_Rminus1_max': 10, 'proposal_scale': 1.0, 'Rminus1_stop': 0.05, 'burn_in': '100d', 'max_tries': '100d','covmat': cov, 'covmat_params': list(sampled_params)}}
#info['sampler'] = {'mcmc': {'learn_proposal': True, 'oversample': True, 'Rminus1_stop': 0.05, 'burn_in': '100d', 'max_tries': '100d'}}
info['sampler']['mcmc']['blocking'] = [(1.0, ['logA','Omegam']), (2.0, ['shift','width','s_wise']), (5.0, ['logSN','b1','alpha_cross','alpha_auto'])]


np.random.seed(123+rank)
starting_point = np.random.multivariate_normal(x,0.5*cov)

starting_point[4] = 0.65
starting_point[7] = np.random.uniform(0.9,1.1)
starting_point[8] = np.random.uniform(0.9,1.1)

print('Starting at ', starting_point)
info['params']['logA']['ref'] =   starting_point[0]
info['params']['Omegam']['ref'] = starting_point[1]
info['params']['logSN']['ref'] = starting_point[2]
info['params']['b1']['ref'] = starting_point[3]
info['params']['s_wise']['ref'] = starting_point[4]
info['params']['shift']['ref'] = starting_point[5]
info['params']['width']['ref'] = starting_point[6]
info['params']['alpha_cross']['ref'] = starting_point[7]
info['params']['alpha_auto']['ref'] = starting_point[8]


info['output'] = output_name + ''
info['resume'] = True
info['force'] = False

products = run(info)
