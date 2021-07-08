# xcorr
Light scripts for computing cross-correlations for SO

## Requirements:

* `NaMaster` and `pymaster`: [https://namaster.readthedocs.io/en/latest/pymaster.html](https://namaster.readthedocs.io/en/latest/pymaster.html)
* `pixell`: [https://github.com/simonsobs/pixell](https://github.com/simonsobs/pixell) (for flat sky computations)
* `healpy`: [https://healpy.readthedocs.io/en/latest](https://healpy.readthedocs.io/en/latest) (for curved sky computations)
* `pyyaml`: [https://pyyaml.org/wiki/PyYAMLDocumentation](https://pyyaml.org/wiki/PyYAMLDocumentation) (to read input files)
* `astropy`: [https://github.com/astropy/astropy](https://github.com/astropy/astropy)
* `sharedmem`: [https://pypi.org/project/sharedmem/0.3/] (for multiprocessing via forking rather than pickling)

## Installation on cori

#### Modules loaded in my .bashrc.ext (not sure if they are all needed):
```
module load cray-fftw/3.3.8.2
module load python/2.7-anaconda-2019.07
module load intel
module load openmpi
```

#### Installing namaster and pymaster:
Install NaMaster with conda:
```
conda install -c conda-forge namaster
conda install -c conda-forge cfitsio=3.430
```

Install pymaster with pip:
`pip install pymaster --user`


#### Installing pixell:
Download the repo from [https://github.com/simonsobs/pixell](https://github.com/simonsobs/pixell),
unzip it,
cd to its root folder,then:
```
python setup.py build_ext -i
```
Add the following line to your .bashrc.ext
```
export PYTHONPATH=$PYTHONPATH:/path/to/your/copy/of/pixell-master
```

#### Installing other needed packages:
```
conda install pyyaml
conda install astropy
conda install -c conda-forge healpy
```

## How to run the Gaussian simulation code:

* From the xcorr directory, run: `python generate_mocks.py --path2config path/to/your/favorite/config.yaml`
* Config file example: `configs/gg-gdg-dgdg-LSSTxSO-curved.yaml`

## How to compute Cls:

* From the xcorr directory, run: `python compute_cls.py --path2config path/to/your/favorite/config.yaml`
* Config file example: `configs/cls-dgdg-curved.yaml`

## Main contributors (in alphabetical order):
* Simone Ferraro (LBNL)
* Alex Krolewski (Waterloo/PI)
* Andrina Nicola (Princeton)
* Emmanuel Schaan (LBNL)
