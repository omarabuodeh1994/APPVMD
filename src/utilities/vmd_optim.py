import numpy as np
from scipy.optimize import minimize  
import matplotlib.pyplot as plt  
from utilities.vmd_post_func import *
from vmdpy import VMD  
import pandas as pd
import pdb
import antropy as ant
from itertools import product
import time
import functools

def min_pow_spec_entr(data):
    """This function is used to compute the maximum power spectrum entropy of each decomposed signal.

    Args:
        data ([[float]]): 2D array of signals with rows as variables and each column as a single observation.

    Returns:
        float: maximum normalized power spectrum entropy
    """
    entr_vec = np.zeros(data.shape[0])
    for i in range(data.shape[0]): 
        entr_vec[i] = ant.spectral_entropy(data[i], sf=1000, method='welch', normalize=True)
    return np.min(entr_vec)

def max_cor_coef(data):
    """This function is used to compute the maximum correlation coefficient.

    Args:
        data ([[float]]): matrix with rows as each variable.

    Returns:
        float: maximum covariance coefficient.
    """
    corr_mat = np.corrcoef(data) # compute correlation matrix of data
    n_elems = len(data)
    cor_seq = np.empty((0,1))
    for i in range(n_elems-1):
        cor_seq = np.append(cor_seq,corr_mat[i,i+1])
    return np.max(cor_seq)

def freq_overlapped(sig_data,tol=0.01,*args):
    """This function is used to decompose a signal and compute the number of modes that contain overlapping frequencies.

    Args:
        sig_data (float): 1D array containing the acceleration signal.
        tol (float): tolerance value for selecting the PSD peaks in each mode relative to the maximum peak of that mode (default is 0.01). 

    Returns:
        int: total number of repeated frequencies.
    """
    freq_res,fs,num_peaks = args
    freq_peaks_list = []
    for i in sig_data:
        freq_peaks_list.append(get_psd_welch_with_peaks(i, freq_res, fs, num_peaks)['peak_f'][0])
    freq_peaks_list = flatten(freq_peaks_list)
    # (2) Compute overlapping modes:
    _,n_uniq_freq = np.unique(freq_peaks_list,return_counts=True) # organize array into a vector of unique elements and store their counts
    return len(np.where(n_uniq_freq>1)[0])

def fitness_VMD(x,sig_data,sig_params):
    """This function is used to compute a fitness function using signals that are decomposed from the VMD function.

    Args:
        x ([int]): integer values moderate bandwidth constraint and number of modes to use in the VMD, respectively.
        sig_data ([float]): acceleration signal.
        sig_params (dict): dictionary containing signal processing parameters like sampling frequency (fs), low pass frequency (low_freq), number of peaks (num_peaks), and frequency resolution (freq_res). 

    Returns:
        float: a fitness function in terms of the maximum correlation and energy loss coefficients.
    """
    # VMD Parameters:
    tau = 0 # noise-tolerance (no strict fidelity enforcement)
    DC = 0 # no DC part imposed
    init = 0 # initialize omegas uniformly
    tol = 1e-6 # tolerance
    # Signal Processing Parameters:
    fs,num_peaks,freq_res = sig_params['fs'],sig_params['num_peaks'],sig_params['freq_res']
    # Analysis:
    alpha, n_modes = x
    n_rows = len(sig_data)-1 if len(sig_data)%2 else len(sig_data) # trimming data based on VMD function (look at VMD code)
    vmd_sigs,_,cen_f_off = VMD(sig_data[:n_rows], alpha, tau, n_modes, 0, 1, 1e-6)
    sort_idx = np.argsort(cen_f_off[-1,:])
    vmd_sigs = vmd_sigs[sort_idx,:]
    entropy_max = min_pow_spec_entr(vmd_sigs) # compute max power spectral entropy of decomposed signals
    n_overlaps = freq_overlapped(vmd_sigs,0.01,freq_res,fs,num_peaks)
    return (entropy_max+n_overlaps)/2.0 # return the maximum correlation coefficient and maximum entropy.

def fitness_VMD_par(alpha,n_modes,sig_data,sig_params):
    """This function is used to compute a fitness function using signals that are decomposed from the VMD function.

    Args:
        x ([int]): integer values moderate bandwidth constraint and number of modes to use in the VMD, respectively.
        sig_data ([float]): acceleration signal.
        sig_params (dict): dictionary containing signal processing parameters like sampling frequency (fs), low pass frequency (low_freq), number of peaks (num_peaks), and frequency resolution (freq_res). 

    Returns:
        float: a fitness function in terms of the maximum correlation and energy loss coefficients.
    """
    # VMD Parameters:
    tau = 0 # noise-tolerance (no strict fidelity enforcement)
    DC = 0 # no DC part imposed
    init = 0 # initialize omegas uniformly
    tol = 1e-6 # tolerance
    # Signal Processing Parameters:
    fs,num_peaks,freq_res = sig_params['fs'],sig_params['num_peaks'],sig_params['freq_res']
    # Analysis:
    n_rows = len(sig_data)-1 if len(sig_data)%2 else len(sig_data) # trimming data based on VMD function (look at VMD code)
    vmd_sigs,_,cen_f_off = VMD(sig_data[:n_rows], alpha, tau, n_modes, 0, 1, 1e-6)
    sort_idx = np.argsort(cen_f_off[-1,:])
    vmd_sigs = vmd_sigs[sort_idx,:]
    entropy_max = min_pow_spec_entr(vmd_sigs) # compute minimum power spectral entropy of decomposed signals
    n_overlaps = freq_overlapped(vmd_sigs,0.01,freq_res,fs,num_peaks)
    return (entropy_max+n_overlaps)/2.0 # return the maximum correlation coefficient and maximum entropy.

def get_opt_vmd_params (sig,alpha_bounds,mode_bounds,sig_params):
        """This function is used to get the optimum parameters that help make the VMD algoritm more efficient.

        Args:
            sig ([float]): 1-D signal of measurements.
            alpha_bounds ([float]): list of minimum and maximum bounds to perform the optimization around.
            mode_bounds ([float]): list of minimum and maximum mode number bounds to perform the optimizaation around.
            sig_params (dict): dictionary containing signal processing parameters like sampling frequency (fs), low pass frequency (low_freq), number of peaks (num_peaks), and frequency resolution (freq_res).

        Returns:
            opt_mode (int): optimum mode number.
            opt_alpha (float): optimum alpha number.
        """
        # (1) generate alpha and mode vectors for minimization search:
        alpha_vec = np.arange(alpha_bounds[0],alpha_bounds[1]+1,25)
        mode_vec = np.arange(mode_bounds[0],mode_bounds[1]+1)
        fit_vec = np.zeros((len(mode_vec),len(alpha_vec)))
        # (2) Construct fitness matrix
        for count, mode_ in enumerate(mode_vec):
            for row, alpha in enumerate(alpha_vec):
                fit_vec[count,row] = fitness_VMD([alpha,mode_],sig,sig_params)
        # (3) select optimum vmd parameters:
        min_fit = np.min(fit_vec,axis=1)
        opt_mode = mode_vec[np.argmin(min_fit)]
        opt_alpha = alpha_vec[np.argmin(fit_vec[np.argmin(min_fit)])]
        return opt_mode,opt_alpha

def vmd_summary(sig_data,sig_params,alpha,K):
    """This function is used to summarize vmd results.

    Args:
        sig_data (float): 1D array containing the acceleration signal.
        sig_params (dict): signal parameters used for low pass filter and vmd ['fs','low_freq','num_peaks','freq_res'].
        alpha (float): penalty factor for frequency bandwidth.
        K (int): number of modes or decompositions. 

    """
    # signal inputs:
    fs,num_peaks,freq_res = sig_params['fs'],sig_params['num_peaks'],sig_params['freq_res']
    # (2) Apply VMD:
    vmd_res = get_VMD_from_vehicle_passes(sig_data,alpha,K,freq_res,fs,num_peaks)
    psd = vmd_res['PSD']
    freq=vmd_res['frequency_vector']
    sigs = vmd_res['deconstructed_signal']
    fig,ax = plt.subplots(K,2)
    for i in range(K):
        ax[i,0].plot(sigs[i])
        ax[i,1].plot(freq,psd[i])
        ax[i,1].set_xlim([0,100])
    fig.show()

if __name__ == '__main__':
    # Inputs:
    low_freq = 100
    freq_res = 0.1
    fs = 1000
    file_ID = 1
    # file inputs:
    file_path = 'parametric_study/case_studies/Rob_CNN/'
    case_study = 'yang'
    # create dictionaries:
    sig_params = {
                    'fs':fs,
                    'low_freq':low_freq,
                    'num_peaks':2,
                    'freq_res':freq_res,
    }
    # -------------------
    # on bridge analysis:
    # -------------------
    # (1) csv file name:
    csv_file = file_path+case_study+'/on_bridge/vehicle_data_'+case_study+'_'
    # (2) extract data:
    sig_data = pd.read_csv(csv_file+str(1)+'.csv')
    # (3) filter signal:
    filt_sig = butter_filter(sig_data['body'], 50, 'low', 1000, 8)
    # (4) get optimum vmd parameters:
    opt_mode, opt_alpha = get_opt_vmd_params (filt_sig,[100,1000],[2,6],sig_params)
    # vmd parameters
    vmd_summary(filt_sig,sig_params,opt_alpha,opt_mode)
    # plt.plot(alpha_bounds,fit_vec)
    pdb.set_trace()
    plt.xlim([0,100])
