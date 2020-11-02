export PATH=/global/homes/a/akrolew/miniconda3/bin:$PATH

# minimize.py runs the minimizer, to get to a reasonable starting point
srun -n 1 python minimize.py
# chains.py defines a starting covariance matrix from Fisher and then runs mcmc chains
srun -n 8 python chains.py