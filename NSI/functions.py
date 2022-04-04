import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

##############################################
########### Wavelet Transform ################
##############################################

def my_cwt(data, frequencies, dt, w0=6.):
    """
    wavelet transform with normalization to catch the amplitude of a sinusoid
    """
    output = np.zeros([len(frequencies), len(data)], dtype=np.complex)

    for ind, freq in enumerate(frequencies):
        wavelet_data = np.conj(get_Morlet_of_right_size(freq, dt, w0=w0))
        sliding_mean = signal.convolve(data,
                                       np.ones(len(wavelet_data))/len(wavelet_data),
                                       mode='same')
        # the final convolution
        wavelet_data_norm = norm_constant_th(freq, dt, w0=w0)
        output[ind, :] = signal.convolve(data-sliding_mean+0.*1j,
                                         wavelet_data,
                                         mode='same')/wavelet_data_norm
    return output

### MORLET WAVELET, definition, properties and normalization
def Morlet_Wavelet(t, f, w0=6.):
    x = 2.*np.pi*f*t
    output = np.exp(1j * x)
    output *= np.exp(-0.5 * ((x/w0) ** 2)) # (Normalization comes later)
    return output

def Morlet_Wavelet_Decay(f, w0=6.):
    return 2 ** .5 * (w0/(np.pi*f))

def from_fourier_to_morlet(freq):
    x = np.linspace(0.1/freq, 2.*freq, 1e3)
    return x[np.argmin((x-freq*(1-np.exp(-freq*x)))**2)]
    
def get_Morlet_of_right_size(f, dt, w0=6., with_t=False):
    Tmax = Morlet_Wavelet_Decay(f, w0=w0)
    t = np.arange(-int(Tmax/dt), int(Tmax/dt)+1)*dt
    if with_t:
        return t, Morlet_Wavelet(t, f, w0=w0)
    else:
        return Morlet_Wavelet(t, f, w0=w0)

def norm_constant_th(freq, dt, w0=6.):
    # from theoretical calculus:
    n = (w0/2./np.sqrt(2.*np.pi)/freq)*(1.+np.exp(-w0**2/2))
    return n/dt

##################################################
########### Processing of the LFP ################
##################################################

def gaussian_smoothing(Signal, idt_sbsmpl=10.):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(Signal, idt_sbsmpl)

def heaviside(x):
    """ heaviside (step) function """
    return (np.sign(x)+1)/2


def compute_pLFP(LFP, sampling_freq,
                 freqs = np.linspace(50, 300, 5),
                 new_dt = 5e-3, # desired subsampling freq.
                 smoothing=42e-3):
    """
    performs continuous wavelet transform and smooth the time-varying high-gamma freq power
    """

    if len(LFP.shape)>1:
        # we compute the pLFP on each channel and sum
        W = np.zeros(LFP.shape[1])
        for i in range(LFP.shape[0]):
            W += np.abs(my_cwt(LFP[i,:], freqs, 1./sampling_freq)).mean(axis=0)
        W /= LFP.shape[0]
    else:
        # performing wavelet transform and taking the mean power over the frequency content considered
        W = np.abs(my_cwt(LFP, freqs, 1./sampling_freq)).mean(axis=0)

    isubsmpl = int(new_dt*sampling_freq)
    
    # then smoothing and subsampling
    pLFP = gaussian_smoothing(np.reshape(W[:int(len(W)/isubsmpl)*isubsmpl],
                                         (int(len(W)/isubsmpl),isubsmpl)).mean(axis=1),
                              int(smoothing/new_dt)).flatten()

    return np.arange(len(pLFP))*new_dt, pLFP
    
def NSI_func(max_low_freqs_power, sliding_mean,
             p0=0.,
             alpha=2.):
    """
    p0 should be the 100th percentile of the signal. It can be a sliding mean.
    """
    X = (p0+alpha*max_low_freqs_power)-sliding_mean # rhythmicity criterion
    return -2*max_low_freqs_power*heaviside(X)+heaviside(-X)*(sliding_mean-p0)


def Validate_Network_States(data, 
                            Tstate=200e-3,
                            Var_criteria=2):
    
    # validate states:
    iTstate = int(Tstate/data['new_dt'])
    # validate the transitions
    data['NSI_validated'] = np.zeros(len(data['pLFP']), dtype=bool)
    data['NSI_unvalidated'] = np.zeros(len(data['pLFP']), dtype=bool)
    for i in np.arange(len(data['pLFP']))[::iTstate][1:-1]:
        if np.array(np.abs(data['NSI'][i-iTstate:i+iTstate]-data['NSI'][i])<=Var_criteria).all():
            data['NSI_validated'][i]=True
        else:
            data['NSI_unvalidated'][i]=True

    data['t_validated'] = data['new_t'][data['NSI_validated']]
    data['i_validated'] = np.arange(len(data['pLFP']))[data['NSI_validated']]
    

def compute_Network_State_Index(signal,
                                low_freqs = np.linspace(2,5,5),
                                Tstate=200e-3,
                                alpha=2.85,
                                T_sliding_mean=0.5):
    
    # compute sliding mean
    sliding_mean = gaussian_smoothing(data[key], int(T_sliding_mean/data['new_dt']))

    # compute low frequency power
    low_freqs = freqs # storing the used-freq
    W_low_freqs = my_cwt(data[key].flatten(), freqs, data['new_dt']) # wavelet transform
    max_low_freqs_power = np.max(np.abs(data['W_low_freqs']), axis=0) # max of freq.

    return NSI_func(max_low_freqs_power, )
    data['NSI']= Network_State_Index(data,
                                     p0 = data['p0'],
                                     alpha=alpha)
    
    Validate_Network_States(data,
                            Tstate=Tstate,
                            # Var_criteria=Var_criteria,
                            Var_criteria=data['p0'])
    
    
    
if __name__=='__main__':

    import matplotlib.pylab as plt
    import NSI
    
    # -- let's build a fake LFP signal array (having the code features of an awake LFP signal)
    tstop, dt = 5, 1e-3 # 10s @ 1kHz
    t = np.arange(int(tstop/dt))*dt
    oscill_part = ((1-np.cos(2*np.pi*3*t))*np.random.randn(len(t))+4*(np.cos(2*np.pi*3*t)-1))*\
        (1-np.sign(t-2))/2.*(2-t)/(tstop-2)
    desynch_part = (1-np.sign(2-t))/2*(t-2)/(tstop-2)*2*np.random.randn(len(t))
    LFP = (oscill_part+desynch_part) # a ~ 1mV ammplitude signal

    # -- compute the pLFP first
    t_pLFP, pLFP = NSI.compute_pLFP(1e3*LFP, 1./dt,
                                    freqs = np.linspace(50,300,10),
                                    new_dt=5e-3,
                                    smoothing=42e-3)

    # -- then compute the NSI from the pLFP
    # NSI = NSI.compute_NSI(pLFP,
    #                       freqs = np.linspace(50, 300, 10))

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(12,4))
    ax[0].plot(t, LFP, color=plt.cm.tab10(7))
    ax[1].plot(t_pLFP, pLFP, color=plt.cm.tab10(5))
    # ax[2].plot(t_NSI, NSI)
    for x, label in zip(ax, ['LFP (mV)', 'pLFP (uV)', 'NSI (uV)']):
        x.set_ylabel(label)
    plt.show()
    
