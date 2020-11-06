import matplotlib
matplotlib.use("Agg")

# Export the results to GetDist
from getdist.mcsamples import loadMCSamples
from getdist import gaussian_mixtures
import matplotlib.pyplot as plt
import numpy as np
# Notice loadMCSamples requires a *full path*
import os
from settings import *
from cobaya.model import get_model

info['params']['omch2'] = "lambda H0,ombh2,tau,mnu,nnu,num_massive_neutrinos,ns,SN,As,Omegam,b1: Omegam * (H0/100.)**2. - ombh2 - (mnu/93.14)"
info['params']['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}


#low_ind_gg = 1
#low_ind_kg = 1

name = output_name.split('/')[1]
names = ['logA','Omegam','b1']

gd_sample = loadMCSamples(os.path.abspath(output_name), settings={'ignore_rows':0.5})

mean = gd_sample.getMeans()

chains = gd_sample.getParams()

#ind = np.arange(len(chains.sigma8))

#gd_sample.filter(ind < 38000)

#mean = gd_sample.getMeans()

#chains = gd_sample.getParams()

logpost = np.loadtxt(output_name + '.1.txt')[:,1]
logpost = logpost[int(0.5*len(logpost)):]

#gd_sample.filter(np.abs(chains.b2) < 1)


chains = gd_sample.getParams()
#logpost = 0.5*(chains.chi2)+chains.minuslogprior

print("%12s %12s" % ('Parameter','Mean'))
for i in range(len(names)):
	if names[i] != 'SN' and names[i] != 'As':
		print('%12s %12.4f' % (names[i], mean[i]))
	else:
		print('%12s %12.4e' % (names[i], mean[i]))
		
map = np.argmin(logpost)

names = ['logA','As','Omegam','sigma8','b1']

print('')
print('MAP')
#print('%12s %12s' % ('logpost','chi2'))
#print('%12.4f %12.4f' % (logpost[map],chains.chi2[map]))
print("%12s %12s" % ('Parameter','MAP'))
for i in range(len(names)):
	exec('value = chains.' + names[i] + '[' + str(map) + ']')
	if names[i] != 'SN' and names[i] != 'As':
		print('%12s %12.4f' % (names[i], value))
	else:
		print('%12s %12.4e' % (names[i], value))
print('%12s %12.4f' % ('logpost',logpost[map]))
print('%12s %12.4f' % ('chi2',chains.chi2[map]))

ml = np.argmin(chains.chi2)

print('')
print('ML')
print("%12s %12s" % ('Parameter','ML'))
for i in range(len(names)):
	exec('value = chains.' + names[i] + '[' + str(ml) + ']')
	if names[i] != 'SN' and names[i] != 'As':
		print('%12s %12.4f' % (names[i], value))
	else:
		print('%12s %12.4e' % (names[i], value))
print('%12s %12.4f' % ('logpost',logpost[ml]))
print('%12s %12.4f' % ('chi2',chains.chi2[ml]))


print('')
print("%12s %12s %12s %12s %12s %12s %12s" % ('Parameter','2.5%','16%','50%','84%','97.5%','1-sigma'))
for i in range(len(names)):
	exec('lower = gd_sample.confidence(chains.' + names[i] + ',0.025,upper=False)')
	exec('lower2 = gd_sample.confidence(chains.' + names[i] + ',0.16,upper=False)')
	exec('median = gd_sample.confidence(chains.' + names[i] + ',0.5)')
	exec('upper2 = gd_sample.confidence(chains.' + names[i] + ',0.16,upper=True)')
	exec('upper = gd_sample.confidence(chains.' + names[i] + ',0.025,upper=True)')
	one_sig = 0.25 * (upper-lower)
	if names[i] != 'SN' and names[i] != 'As':
		print('%12s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f' % (names[i], lower, lower2, median, upper2, upper, one_sig))
	else:
		print('%12s %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e' % (names[i], lower, lower2, median, upper2, upper, one_sig))
		
#model = get_model(info)
#x = [chains.logA[map], chains.Omegam[map], chains.logSN[map], chains.b1[map], chains.b2[map], chains.alpha_cross[map]]
#like = model.loglike({'logA': 3.0488, 'Omegam': 0.3092, 'logSN': -6.6656, 'b1': 1.1928, 'b2': -0.4715, 'alpha_cross': -19.82})

#like = model.loglike({'logA': np.log(1e10*2.4984e-9), 'Omegam': 0.2749, 'logSN': -6.6866, 'b1': 1.0658, 'b2': 0.3400, 'alpha_cross': -16.16})


# Plot data and best-fit models
plt.figure()
plt.errorbar(data_auto[0,low_ind_gg:high_ind],1e5*data_auto[1,low_ind_gg:high_ind],1e5*data_auto[2,low_ind_gg:high_ind])
plt.ylim(0,2.0)

model = get_model(info)
x = [chains.logA[map], chains.Omegam[map], chains.b1[map]]
#x = [chains.logA[ml], chains.Omegam[ml], chains.logSN[ml], chains.b1[ml], chains.b2[ml], chains.alpha_cross[ml], chains.shift[ml], chains.width[ml]]

like = model.loglike({'logA': x[0], 'Omegam': x[1], 'b1': x[2]})
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
	x[1], x[2], 0,
	0,0,0,0, 0,0, 1)
	
lim_bin_cross = bin(all_ell,lim_cross,lmin[low_ind_kg:high_ind],lmax[low_ind_kg:high_ind])
# Correct for namaster binning
#lim_bin_cross = lim_bin #* clkg_corr[low_ind:high_ind]

#print('lim_auto immediately before bin',lim_auto)

lim_bin_auto = bin(all_ell,lim_auto,lmin[low_ind_gg:high_ind],lmax[low_ind_gg:high_ind]) + info['params']['SN']

plt.figure()
plt.errorbar(data_auto[0,low_ind_gg:high_ind],1e5*data_auto[1,low_ind_gg:high_ind],1e5*data_auto[2,low_ind_gg:high_ind])
#plt.ylim(0,2.0)


#lim_bin_auto = np.loadtxt('clgg.txt')
#lim_bin_cross = np.loadtxt('clkg.txt')

plt.plot(0.5*(lmin[low_ind_gg:high_ind]+lmax[low_ind_gg:high_ind]),1e5*lim_bin_auto,color='r',linestyle='--',label='MAP binned')
plt.plot(all_ell,1e5*(lim_auto+SN),color='r',label='MAP')

plt.xlim(0,600)

plt.legend(frameon=False)

os.system('mkdir plots/'+name)

plt.title('Green Auto',size=25)
plt.xlabel(r'$\ell$',size=25)
plt.ylabel(r'$C_{\ell}^{gg}$',size=25)
plt.tight_layout()
plt.savefig('plots/'+name+'/clgg.png')

plt.figure()
plt.errorbar(data_cross[0,low_ind_kg:high_ind],1e5*data_cross[0,low_ind_kg:high_ind]* data_cross[1,low_ind_kg:high_ind],1e5*data_cross[0,low_ind_kg:high_ind]*data_cross[2,low_ind_kg:high_ind])

plt.plot(0.5*(lmin[low_ind_kg:high_ind]+lmax[low_ind_kg:high_ind]),1e5*0.5*(lmin[low_ind_kg:high_ind]+lmax[low_ind_kg:high_ind])*lim_bin_cross,color='r',linestyle='--',label='MAP binned')
plt.plot(all_ell,1e5*all_ell*lim_cross,color='r',label='MAP')
plt.ylim(0,4)

plt.xlim(0,600)

plt.legend(frameon=False)

os.system('mkdir plots/'+name)

plt.title('Green Cross',size=25)
plt.xlabel(r'$\ell$',size=25)
plt.ylabel(r'$\ell C_{\ell}^{\kappa g}$',size=25)
plt.tight_layout()
plt.savefig('plots/'+name+'/clkg.png')

#print(5/0)


import getdist.plots as gdplt

gdplot = gdplt.getSubplotPlotter()

gdplot.triangle_plot([gd_sample], ['sigma8','Omegam','b1'],markers={'sigma8': 0.8287, 'Omegam': 0.3},filled=True)
#	markers={'sigma8': 0.937, 'Omegam': 0.359, 'b1': 0.706, 'b2': -0.149, 'alpha_cross': 0.880, 'alpha_auto': -26.045, 'alpha_matter': -20.480})


plt.savefig('plots/'+name+'/contours.png')


os.system('mkdir plots/'+name+'/trace')

plt.figure()
plt.plot(chains.sigma8)
plt.savefig('plots/'+name+'/trace/sigma8.png')

plt.figure()
plt.plot(chains.As)
plt.savefig('plots/'+name+'/trace/As.png')

plt.figure()
plt.plot(chains.Omegam)
plt.savefig('plots/'+name+'/trace/Omegam.png')

plt.figure()
plt.plot(chains.b1)
plt.savefig('plots/'+name+'/trace/b1.png')


plt.figure()
plt.plot(logpost)
plt.savefig('plots/'+name+'/trace/logpost.png')

tests = gd_sample.getConvergeTests(test_confidence=0.95,
	writeDataToFile=True,
	what=['MeanVar','GelmanRubin','SplitTest','RafteryLewis','CorrLengths'],
	filename=name+'.converge',
	feedback=True)
	
gdplot.triangle_plot([gd_sample], ['sigma8','Omegam'],markers={'sigma8': 0.8287, 'Omegam': 0.3},filled=True)
#	markers={'sigma8': 0.937, 'Omegam': 0.359, 'b1': 0.706, 'b2': -0.149, 'alpha_cross': 0.880, 'alpha_auto': -26.045, 'alpha_matter': -20.480})


plt.savefig('plots/'+name+'/contours_sig8_Om.png')

gd_sample.addDerived(gd_sample.getParams().sigma8 * gd_sample.getParams().Omegam**0.5 , name='S8', label='S8')