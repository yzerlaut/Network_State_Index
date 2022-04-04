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
import numpy as np
import nsi # the NSI module

# -- let's build a fake LFP signal array (having the code features of an awake LFP signal)
tstop, dt, sbsmpl_dt = 5, 1e-3, 5e-3 # 10s @ 1kHz
t = np.arange(int(tstop/dt))*dt
oscill_part = ((1-np.cos(2*np.pi*3*t))*np.random.randn(len(t))+4*(np.cos(2*np.pi*3*t)-1))*\
    (1-np.sign(t-2))/2.*(2-t)/(tstop-2)
desynch_part = (1-np.sign(2-t))/2*(t-2)/(tstop-2)*2*np.random.randn(len(t))
LFP = (oscill_part+desynch_part)*.1 # a ~ 1mV ammplitude signal

# -- compute the pLFP first
t_pLFP, pLFP = nsi.compute_pLFP(1e3*LFP, 1./dt,
                                freqs = np.linspace(50,300,10),
                                new_dt=sbsmpl_dt,
                                smoothing=42e-3)
p0 = np.percentile(pLFP, 0./100) # first 100th percentile

# -- then compute the NSI from the pLFP
NSI = nsi.compute_NSI(pLFP, 1./sbsmpl_dt,
                      low_freqs = np.linspace(2, 5, 4),
                      p0=p0,
                      alpha=2.85)

# then validate NSI episodes
tvNSI, vNSI = nsi.validate_NSI(t_pLFP, NSI,
                               var_tolerance_threshold=20*p0) # here no noise so we increase the thresh


# let's plot the result
import matplotlib.pylab as plt
fig, ax = plt.subplots(3, 1, figsize=(12,4))
ax[0].plot(t, LFP, color=plt.cm.tab10(7))
ax[1].plot(t_pLFP, pLFP, color=plt.cm.tab10(5))
ax[2].plot(t_pLFP, NSI, label='raw')
ax[2].plot(tvNSI, vNSI, 'o', label='validated', lw=0)
ax[2].legend(frameon=False)

for x, label in zip(ax, ['LFP (mV)', 'pLFP (uV)', 'NSI (uV)']):
    x.set_ylabel(label)
    if 'NSI'in label:
        x.set_xlabel('time (s)')
    else:
        x.set_xticklabels([])
plt.show()
```

<p align="center">
  <img src="./demo/synthetic-example.png"/>
</p>

Execute the above example by running: `python nsi/functions/py`

### Demo on the "Visual Coding - Neuropixels" dataset

see associated notebook

### Demo on the paper's dataset

see associated notebook

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/Network_State_index/issues) interface of Github.


## Notes

My implementation of the continuous wavelet transform (`NSI.my_cwt`) is not very efficient... Any suggestions/ideas to improve this is very welcome :)
