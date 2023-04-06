import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter,sosfilt, find_peaks,peak_widths
import scipy.signal as signal
from vmdpy import VMD 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
import pdb
import seaborn as sns

# write an index for each function:
# (1) list_to_df: converts lists to dataframes.
# (2) ...
# Functions used in VMD Analysis:
def cust_range(start,stop,step, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])
    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included
    """
    # process arguments
    # if len(args) == 1:
    #     start = 0
    #     stop = args[0]
    #     step = 1
    # elif len(args) == 2:
    #     start, stop = args
    #     step = 1
    # else:
    #     assert len(args) == 3
    #     start, stop, step = tuple(args)
    
    # determine number of segments
    n = (stop-start)/step + 1
    
    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)
    
    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step
    
    return np.linspace(start, stop, int(n))

def crange(start,stop,step):
    return cust_range(start,stop,step,rtol=1e-05, atol=1e-08, include=[True, True])

def orange(start,stop,step):
    return cust_range(start,stop,step,rtol=1e-05, atol=1e-08, include=[True, False])

def list_to_df(list_arrays,col_labs):
    """This function is used to convert list of dataframes containing the decomposed signals for each sensor (e.g., body, front axle)
    to a single dataframe.

    Args:
        list_arrays ([df]): list of dataframes containing the decomposed signals for each sensor.
        col_labs ([str]): list of strings describing the name of the sensors.
    
    Returns:
        df: dataframe containing the list of dataframes concatenated into one dataframe distinguished by the sensor and mode number.
    """
    df_res = pd.DataFrame()
    mode_col,sensor_col = [],[]
    for idx,sig_group in enumerate(list_arrays):
        for count,sig in enumerate(sig_group):
            df_temp = pd.DataFrame()
            df_temp['acc'] = sig
            df_res = pd.concat((df_res,df_temp))
            mode_str = f'mode_{str(count+1)}'
            mode_col.append(sig.shape[0]*[mode_str])
        sensor_col.append(sig_group.shape[0]*sig_group.shape[1]*[col_labs[idx]])
    df_res.reset_index(inplace=True,drop=True)
    mode_col = flatten(mode_col)
    sensor_col = flatten(sensor_col)
    df_res['mode_num'] = mode_col
    df_res['sensor'] = sensor_col
    return df_res

def df_2_mat(df):
    """This function converts a df with time signal in first column to vector/matrix of amplitudes, excluding the time.

    Args:
        df (df): signal dataframe consisting of time with the measured outputs (could be a single signal or multiple).

    Returns:
        [[float]]: signal measured output in the form of a matrix.
    """
    df_rows,df_cols = df.shape
    col_labs = df.columns
    sig_mat_off = np.zeros((df_rows,df_cols))
    for count, col in enumerate(col_labs):
        sig_mat_off[:,count] = df[col]
    return sig_mat_off

def find_closest_to(fineArray,courseArray):
    """This function is used to find the indeces of elements of a fine array from a course array.

    Args:
        fineArray ([float]): a large array.
        courseArray ([float]): smaller array.

    Returns:
        [int]: a list of indices.
    """
    indexArr = []
    for Value in courseArray:
        diff = abs(fineArray-Value)
        minDiffIndex = diff.argmin()
        indexArr.append(minDiffIndex)
    
    return indexArr

def get_psd_welch_with_peaks(data,freq_res,fs,num_peaks):
    """This function is used to calculate PSDs using the Welch's algorithm and perform peak picking using the Continuous Wavelet Transform algorithm.

    Args:
        data ([float]): signal array.
        freq_res (float): frequency resolution of interest in Hz.
        fs (float): sampling frequency in Hz.
        num_peaks (int): number of peaks of interest.

    Returns:
        dict: dictionary dataframe containing the frequency and PSD ['Frequency','PSD'] with the specified number of peak PSD values with their corresponding frequencies ['peak_f','peak_psd'].
    """
    # compute psd using the Welch algorithm:
    f, psd = signal.welch(data,fs=fs,nperseg=data.shape[0]//2,noverlap=data.shape[0]//2//1.5,nfft=fs/freq_res,window='hanning',axis=0)
    # find peaks of frequency up until 100 Hz:
    peak_psd,peak_f = get_peaks(f,freq_res*0,psd,num_peaks)
    return {'PSD_df':
                            pd.DataFrame({
                                'Frequency': f,
                                'PSD': psd
                            }),
                    'peak_f':peak_f,
                    'peak_PSD':peak_psd
                    }

def get_psd_welch(data,freq_res,fs):
    """This function is used to calculate PSDs using the Welch's algorithm using scipy.signal package.

    Args:
        data ([[float]]): one/two-dimensional array where rows are observations and columns are variables.
        freq_res (float): frequency resolution of interest in Hz.
        fs (float): sampling frequency in Hz.

    Returns:
        frequency [float],Pxx [float]: frequency and PSD of signal.
    """
    if len(data.shape)==3:
        n_rows,n_cols,num_modes = int(fs/freq_res),data.shape[1],data.shape[2]
        psd = np.zeros((n_rows // 2, n_cols,num_modes))
        for m in range(num_modes):
            for i in range(n_cols):
                f, psd[:,i,m] = signal.welch(data[:,i,m],fs=fs,nperseg=data.shape[0]//2,noverlap=data.shape[0]//2//1.5,nfft=fs//freq_res,window='hann',axis=0)
    elif len(data.shape)==2:
        n_rows,n_cols = int(fs/freq_res),data.shape[1]
        psd = np.zeros((n_rows // 2, n_cols))
        for i in range(n_cols):
            f, psd[:,i] = signal.welch(data[:,i],fs=fs,nperseg=data.shape[0]//2,noverlap=data.shape[0]//2//1.5,nfft=fs//freq_res,window='hann',axis=0)
    else:
        f,psd = signal.welch(data,fs=fs,nperseg=len(data)//2,noverlap=len(data)//2//1.5,nfft=fs//freq_res,window='hann',axis=0)
    return np.round(f,1),psd

def get_fft(sig,freq_res,fs):
    """This function is used to compute the fft of a signal using the signal.fft package.

    Args:
        sig ([float]): input signal 1D/2D to fft, where if 2D rows and columns represent observations and variables, respectively.
        freq_res (float): desired frequency resolution in Hz.
        fs (int): sampling frequency in Hz

    Returns:
        frequency,Yxx: frequency and fft amplitudes, respectively.
    """
    if len(sig.shape) == 2:
        N,n_cols = int(fs/freq_res),sig.shape[1]
        Y = np.zeros((N,n_cols))
        for i in range(n_cols):
            Y[:,i] = abs(fft(sig[:,i].T,N))/N
    else:
        # Number of sample points
        N = int(fs//freq_res)
        # compute fft
        Y = abs(fft(sig.T,int(fs/freq_res)))/N
    # sample spacing
    T = 1.0 / fs
    # compute frequency vector
    f = fftfreq(int(fs/freq_res), T)
    return np.round(f[:N//2],1),Y.T[:N//2]

def get_psd_welch_mlab(sig,freq_res,fs):
    """This function is used to calculate PSDs using the Welch's algorithm from the matplotlib.mlab package.

    Args:
        data ([[float]]): one/two-dimensional array where rows are observations and columns are variables.
        freq_res (float): frequency resolution of interest in Hz.
        fs (float): sampling frequency in Hz.

    Returns:
        frequency [float],Pxx [float]: frequency and PSD of signal.
    """
    Pxx,freq = mlb.psd(sig,Fs=fs,window=mlb.window_hanning,pad_to=int(fs//freq_res),NFFT = int(len(sig)//2),noverlap = int(len(sig)//2//1.5))
    return np.round(freq,1), Pxx

def butter_filter(data, cutoff, filter_type, sample_f, order):
    """This function is used to carry out a low pass filter using the butterworth filter.

    Args:
        data ([arr]): numpy array of data from the signal.
        cutoff (float): cutoff frequency of interest.
        filter_type (str): type of filter to be used ('lowpass', 'highpass', 'bandpass', 'bandstop')
        sample_f (float): sampling frequency in Hz.
        order (int): the order of the filter.

    Returns:
        [arr]: returns filtered signal.
    """
    # normal_cutoff = 2*cutoff / fs
    # Get the filter coefficients 
    sos = butter(order, cutoff, btype=filter_type,fs=sample_f,output='sos',analog=False)
    return signal.sosfiltfilt(sos, data)

def get_diff_col(dfA,dfB,bin_width):
    """This function is used to compare the peak frequencies between two dataframes (dfA and dfB) and keep the frequencies that only exist in one dataframe (dfA).

    Args:
        dfA (df): the dataframe consisting of frequencies of interest that are not in the other dataframe.
        dfB (df): the dataframe consisting of frequencies that are used to remove redundant frequencies from dfA.
        bin_width(float): width of frequency bin.
        
    Returns:
        df: dataframe with the frequencies that are only in dfA and not in dfB
    """
    diff_df_ = pd.DataFrame()
    for df_row in range(dfA.shape[0]):
        if ((dfA['bins'].loc[df_row] not in dfB['bins'].values) | (dfA['bins_count'].loc[df_row] > 1.15*dfB[dfB['bins']==dfA['bins'].loc[df_row]]['bins_count']).all()).any():
            diff_df_ = pd.concat((diff_df_,pd.DataFrame(dfA.loc[df_row]).T)).reset_index(drop=True)
    if diff_df_.empty:
        diff_df_['bins'] = [pd.Interval(0,0.5,'left')]
        diff_df_['bins_count'] = 1
        diff_df_['mean_peak_psd'] = 0
    # diff_df.dropna(inplace=True)
    return diff_df_.reset_index(drop=True)

def get_freq_bounds(f,psd,peak_idx,num_modes):
    """This function is used to compute the frequency bounds, using SciPy's peak_widths(), of the most dominant peaks within a psd matrix.

    Args:
        f ([float]): frequency vector.
        psd ([[float]]): one/two dimensional array of power spectrum values [rows,sensors,mode number].
        peak_idx ([int]): array containing location of maximum peaks within a psd array.
        num_modes (int): number of modes to be extracted.

    Returns:
        [arr]: list of arrays containing lower and upper frequency bounds [array_low,array_upper].
    """
    f_low = np.zeros((num_modes,psd.shape[1]))
    f_up = np.zeros((num_modes,psd.shape[1]))
    width_height = np.zeros((num_modes,psd.shape[1]))
    for mode_num in range(num_modes):
        for count, idx in enumerate(peak_idx):
            p_widths = peak_widths(psd[:,count,mode_num].T, [idx[mode_num]], rel_height=0.5)
            idx_low = 1 if np.floor(p_widths[2]).astype(int)[0] == 0 else np.floor(p_widths[2]).astype(int)[0]
            f_low[mode_num,count] = f[idx_low]
            idx_up = 1 if np.floor(p_widths[3]).astype(int)[0] == 0 else np.floor(p_widths[3]).astype(int)[0]
            f_up[mode_num,count] = f[idx_up]
            width_height[mode_num,count] = p_widths[1]
    return f_low,f_up



def gen_mode_num(num_modes):
    """This function is used to create a list of strings describing the mode numbers analyzed during vmd results extraction.

    Args:
        num_modes (int): number of modes the user wants to analyze.

    Returns:
        [str]: list of strings describing mode numbers starting from 1 and ending on num_modes+1.
    """
    return ['mode_'+str(m) for m in range(1,num_modes+1)]



# -----------------------
# VMD Analysis functions:
# -----------------------
def dec_vmd(sigs,alpha,tau,K):
    """This function is used to decompose signals using VMD algorithm.

    Args:
        sigs ([[float]]): two dimensional array of measured signals where rows are observations and columns are variables.
        alpha (float): penalty factor.
        tau (float): langrangian factor.
        K (int): number of decompositions.

    Returns:
        [[float]]: returns array of decomposed signals.
    """
    if len(sigs.shape) == 2:
        n_rows,n_cols = sigs.shape
    else:
        n_rows = len(sigs)
    n_rows = n_rows-1 if n_rows%2 else n_rows # trimming data based on VMD function (look at VMD code)
    if 'n_cols' in locals():
        dec_sigs = np.zeros((n_cols,K,n_rows))
        for i in range(n_cols):
            dec_sigs[i],_,cen_f_off = VMD(sigs[:,i], alpha, tau, K, 0, 1, 1e-6)
            sort_idx = np.argsort(cen_f_off[-1,:])
            dec_sigs[i] = dec_sigs[i,sort_idx,:]
    else:
        dec_sigs = np.zeros((K,n_rows))
        dec_sigs,_,cen_f_off = VMD(sigs, alpha, tau, K, 0, 1, 1e-6)
        sort_idx = np.argsort(cen_f_off[-1,:])
        dec_sigs = dec_sigs[sort_idx,:]
    return dec_sigs



def get_peaks(f,freq_step,psd,num_peaks):
    """This function is used to carry out peak picking using SciPy's find_peaks().

    Args:
        f ([float]): frequency array.
        psd ([[[float]]]): 1D/2D array contain power spectrum of decomposed signals measured from multiple sensors.
        num_peaks (int): number of peaks to pick from the psd values.

    Returns:
        pks_psd([float]): list of peak psd values in an array within a list where each list of arrays corresponds to the peaks of a certain sensor and each array corresponds to the peaks of a particular mode.
        pks_f([float]): list of frequency values corresponding to peak psd values.
    """
    pks_psd = []
    pks_f = []
    if len(psd.shape) == 3:
        for mode_num in range(psd.shape[2]):
            for sensor_id in range(psd.shape[1]):
                max_psd = np.max(psd[:,sensor_id,mode_num])
                pks_idx = find_peaks(psd[:,sensor_id,mode_num],height=0.1*max_psd)
                pks_idx_temp = get_points_around_peaks(f,freq_step,pks_idx[0],psd[:,sensor_id,mode_num])
                pks_psd.append(np.sort(psd[pks_idx_temp,sensor_id,mode_num])[::-1][:num_peaks])
                pks_f.append(f[find_closest_to(psd[:,sensor_id,mode_num],np.sort(psd[pks_idx_temp,sensor_id,mode_num])[::-1][:num_peaks])])
    elif len(psd.shape) == 2:
        for count,p in enumerate(psd.T):
            max_psd = np.max(p)
            pks_idx = find_peaks(p,height=0.1*max_psd)
            pks_idx_temp = get_points_around_peaks(f,freq_step,pks_idx[0],p)
            pks_psd.append(np.sort(p[pks_idx_temp])[::-1][:num_peaks])
            pks_f.append(f[find_closest_to(p,pks_psd[count])])
    else:
        max_psd = np.max(psd)
        pks_idx = find_peaks(psd,height=0.1*max_psd)
        pks_idx_temp = get_points_around_peaks(f,freq_step,pks_idx[0],psd)
        pks_psd.append(np.sort(psd[pks_idx_temp])[::-1][:num_peaks])
        pks_f.append(f[find_closest_to(psd,pks_psd[0])])
    return pks_psd,pks_f



def get_points_around_peaks(f,freq_res,peak_idx,psd_arr):
    """This function is used to extract points around peaks in case of extremely low curvatures.

    Args:
        peak_idx ([int]): indices of peaks.

    Returns:
        [int]: list of indices of peaks including the points around the peaks.
    """
    pks_idx_temp = []
    step_ = np.where(f==freq_res)[0][0]
    for pks_idx_ in peak_idx:
        p_after = pks_idx_ + step_
        if pks_idx_ != 0:
            p_before = pks_idx_ - step_
            if psd_arr[p_before] > 0.8*psd_arr[pks_idx_]:
                pks_idx_temp.append(p_before)
        if psd_arr[p_after] > 0.8*psd_arr[pks_idx_]:
            pks_idx_temp.append(p_after)
        pks_idx_temp.append(pks_idx_)
    return pks_idx_temp

def find_common(df_A,df_B):
    """This function is used to find the common frequencies between two dataframes and combining them into one dataframe by taking the peak frequency
    as the common factor and summation over the rest of the columns.

    Args:
        df_A (df): first dataframe (axle reading)
        df_B (df): second dataframe (axle reading)

    Returns:
        df: a dataframe containing the combined frequencies tha are in common between both dataframes with their corresponding columns being added in terms of their rows.
    """
    # perform deep copy of dfs:
    dfA_ = df_A.copy()
    dfB_ = df_B.copy()
    # get intersection between dfs:
    elem_intersect = np.unique(dfA_[dfA_['bins'].isin(dfB_['bins'])]['bins'].to_numpy())
    # concatenate and extract the elements that are within the intersection:
    df_concat = pd.concat((dfA_,dfB_))
    df_concat = df_concat.loc[df_concat['bins'].isin(elem_intersect)].reset_index(drop=True)
    # group by bins and vehicle pass to distinguish between them:
    df_group = df_concat.groupby(['pass_num','peak_f']).agg(bins=('bins',lambda x: x.iloc[0]),mean_peak_psd=('peak_psd','mean')).dropna().reset_index()
    return df_group.groupby('peak_f').agg(peak_f_count=('peak_f','count'),bins=('bins',lambda x: x.iloc[0]),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()

def separate_dfs(df,sensor_names):
    """This function is used to separate dataframe of peaks based on sensor location into separate dfs in a list.

    Args:
        df (df): dataframe containing peak frequencies and psd for each sensor.
        sensor_names ([str]): list of strings describing name of sensor.

    Returns:
        [df]: list of dfs containing the df of each sensor.
    """
    df_sep = []
    for sensor_name in sensor_names:
        df_sep.append(df[df['sensor'] == sensor_name].copy().reset_index(drop=True))
    return df_sep
    
def bin_df(df_list,input_params):
    """This function is used to bin the peak_f column according to a user-defined binning width.

    Args:
        df ([df]): list of dataframes with peak_f column.
        input_params (objects): class objects containing input parameters defined in input_params.py.

    Returns:
        df: a binned dataframe containing counts of each bin.
    """
    df_bins = []
    for idx,df in enumerate(df_list):
        min_bin = np.floor(df['peak_f'].min())
        max_bin = df['peak_f'].max()+1 if ((df['peak_f'].max()-int(df['peak_f'].max()))==0 or (df['peak_f'].max()-int(df['peak_f'].max()))==0.5 ) else df['peak_f'].max()
        df_bins.append(df.copy())
        df_bins[idx]['bins'],_ = pd.cut(df['peak_f'], bins = np.arange(min_bin,max_bin+input_params.bin_width,input_params.bin_width),right=False,retbins=True)
    return df_bins

def drop_rep_freq(df,num_rep):
    """This function is used to drop the rows of a dataframe that have frequency values repeated less than a user-defined number of times.

    Args:
        df (df): dataframe containing the peak psd with their corresponding frequency values.
        num_rep (int): number of times the frequency is repeated.

    Returns:
        df: filtered dataframe with frequency values repeated more than the number repetitions specified.
    """
    counts = df['peak_f'].value_counts()
    return df[~df['peak_f'].isin(counts[counts <= num_rep].index)].reset_index(drop=True)

def flatten(t):
    """Function flattens a list of lists into a single list.

    Args:
        t (list): list of lists.

    Returns:
        list: a single list.
    """
    return [item for sublist in t for item in sublist]

def concat_dfs(df_list,col_lab,list_str):
    """This function is used to concatenate a list of dfs and create a separate column to distinguish them.

    Args:
        df_list ([df]): list of dataframe containing measured outputs from the vehicle-bridge extraction test.
        col_lab (str): name of column label for column used for distinguishing dataframe.
        list_str ([str]): list of strings describing the column.

    Returns:
        df: a dataframe containing the list of dataframes concatenated on top of each other and distinguished based on the sensor.
    """
    df_res = pd.DataFrame()
    for df in df_list:
        df_res = pd.concat((df_res,df))
    sensor_list = [df_list[idx].shape[0] * [sens_label] for idx, sens_label in enumerate(list_str)]
    df_res[col_lab] = flatten(sensor_list)
    df_res.reset_index(drop=True,inplace=True)
    return df_res 

def get_hist_freq(df_,sensor_lab,bin_width,plot_title):
        """This function is used to create a histogram that plots frequency values from a dataframe based on sensor labels of interest.

        Args:
            df (df_): dataframe containing frequency values with the corresponding sensors.
            sensor_lab ([str]): list of strings pertaining which sensors to view.
            bin_width (float): frequency resolution used in Hz.
            plot_title (str): title of the plot

        Returns:
            Axes: A matplot.axes object.
        """
        df = df_.copy()
        fig,axs = plt.subplots(1)
        min_f = round(df['peak_f'].loc[df['sensor'].isin(sensor_lab)].min()/bin_width)*bin_width
        max_f = round((df['peak_f'].loc[df['sensor'].isin(sensor_lab)].max()+bin_width)/bin_width)*bin_width
        hist_axes = sns.histplot(df[df['sensor'].isin(sensor_lab)],x='peak_f',hue='sensor',multiple='stack',binwidth=bin_width,
                                 binrange=(min_f,max_f),ax=axs)
        bin_range = np.arange(min_f,max_f+bin_width,bin_width)
        hist_axes.set_xticks(bin_range,bin_range)
        hist_axes.set_xlim([min_f,max_f+bin_width])
        hist_axes.set_xlabel('Frequencies (Hz)'); hist_axes.set_ylabel('Counts')
        hist_axes.set_title(plot_title)
        df_binned = bin_df(df.loc[df['sensor'].isin(sensor_lab)],bin_width)
        min_count = 0
        max_count = int(np.ceil(df_binned['bins'].value_counts().max() / 5.0)) * 5
        hist_axes.set_yticks([int(i) for i in np.arange(min_count,max_count+5,5)],label=[int(i) for i in np.arange(min_count,max_count+5,5)])
        return fig,hist_axes

def get_VMD_from_vehicle_passes(signal_data,alpha,K,freq_bin_width,samp_freq,num_peaks):
    """This function is used to carry out VMD on vehicle data to extract PSD, peak frequencies, and centered frequencies for each vehicle pass.

    Args:
        signal_data ([float]): output signal.
        vmd_parameters ([float]): parameters used in the VMD package.
        freq_bin_width (float): frequency resolution of interest in Hz.
        samp_freq (float): sampling frequency in Hz.
        num_peaks (int): number of peaks to extract using the peak picking continuous wavelet transform algorithm.

    Returns:
        dict: dictionary containing results like the PSD (Kxlen(freq)), peak frequencies (Kxnum_peaks), peak_PSD (Kxnum_peaks), and centered frequencies ((Kx1)) ['PSD','peak_frequencies','indices_peak_frequencies','centered_frequencies']
    """
    tau, DC, init, tol = [0,0,0,1e-6]
    deconstructed_signal,_,centered_frequencies = VMD(signal_data, alpha, tau, K, DC, init, tol)
    # sort signals from lowest to highest frequencies
    sortIndex = np.argsort(centered_frequencies[-1,:])
    centered_frequencies = centered_frequencies[:,sortIndex]
    deconstructed_signal = deconstructed_signal[sortIndex,:]
    # --- PSD and Peak picking:
    # calculate psd for first mode to know the psd and freq array sizes:
    freq_vec = np.arange(0,samp_freq/2.0+freq_bin_width,freq_bin_width)
    psd_matrix = np.empty(shape=(0,len(freq_vec)))
    peak_f, peak_psd = [],[]
    for i in range(K):
        psd_df = get_psd_welch_with_peaks(deconstructed_signal[i,:],freq_bin_width,samp_freq,num_peaks)
        psd_matrix = np.vstack([psd_matrix,psd_df['PSD_df']['PSD'].values]) # store psd values for following modes
        peak_f.append(psd_df['peak_f'])
        peak_psd.append(psd_df['peak_PSD'])
    results_dict = {
                    'deconstructed_signal': deconstructed_signal,
                    'PSD':psd_matrix,
                    'frequency_vector':freq_vec,
                    'peak_frequencies':peak_f,
                    'peak_PSD':peak_psd,
                    'centered_frequencies':centered_frequencies
                    }
    return results_dict

if __name__ == '__main__':
    x = 1