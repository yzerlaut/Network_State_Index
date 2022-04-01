# Network State Index (NSI)

#### Classifying Network State from LFP in Neocortex 

This module provides a quantitative characterization of network states in neocortex from extracellular signals. It implements the analysis described in the following article (please cite if you use this code !):
> Network States Classification based on Local Field Potential Recordings in the Awake Mouse Neocortex
> Yann Zerlaut, Stefano Zucca, Tommaso Fellin, Stefano Panzeri
> bioRxiv 2022.02.08.479568; doi: https://doi.org/10.1101/2022.02.08.479568


## Installation

1. Install a python distribution for scientific analysis:

   get the [latest Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html) and install it on your home folder.
   
2. Run the following in the [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal):

```
git clone https://github.com/yzerlaut/Network_State_Index.git
cd Network_State_Index
pip install .
```

If you do not wish to clone the repository you can also directly:
```
pip install git+https://github.com/yzerlaut/Network_State_Index
```


## Usage

### Minimal demo

```
import NSI

# -- let's build a fake LFP signal array
tstop, dt = 10, 1e-4 # 10s @ 10kHz
t = np.arange(int(tstop/dt))*dt
LFP = (1-np.sin(3*t) np.random.randn(len(t))

# -- compute the pLFP first
t_pLFP, pLFP = NSI.compute_pLFP(LFP,
			        freqs = np.linspace(50,300,10),
				new_dt=1e-3,
				smoothing=42e-3)
# -- then compute the NSI from the pLFP
NSI = NSI.compute_NSI(pLFP,
		      freqs = np.linspace(50, 300, ))

```

### Demo on the "Visual Coding - Neuropixels" dataset

see associated notebook

### Demo on the paper's dataset

see associated notebook

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/Network_State_index/issues) interface of Github.


## Notes

My implementation of the continuous wavelet transform (`NSI.my_cwt`) is not very efficient... Any suggestions/ideas to improve this is very welcome :)
