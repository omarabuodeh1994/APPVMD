from statistics import mean
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  
import pandas as pd
from vmd_post_func import *
import seaborn as sns
import pdb
from scipy.signal import find_peaks,peak_widths
from functools import reduce
from vmd_optim import *
import os
import time
from joblib import Parallel, delayed

def vmd_each_veh_pass(file_path,veh_class,case_study,pass_num,fs,low_freq,freq_res,num_peaks,bridge_len,vel):
    # (1) csv file name for off and on bridge data:
    csv_file_off = off_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
    csv_file_on = on_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
    # (2) load data into dataframes:
    # off-bridge:
    sig_data_off = pd.read_csv(csv_file_off)
    sig_len_off = sig_data_off.shape[0]-1 if sig_data_off.shape[0] % 2 else sig_data_off.shape[0]  # length of signal
    col_labs = sig_data_off.columns
    num_sensors = len(col_labs[1:]) # number of sensors
    time_off = sig_data_off['time'].loc[:sig_len_off-1]
    # on-bridge:
    sig_data_on = pd.read_csv(csv_file_on)
    sig_len_on = sig_data_on.shape[0]-1 if sig_data_on.shape[0] % 2 else sig_data_on.shape[0]
    time_on = sig_data_on['time'].loc[:sig_len_on-1]
    # (3) convert dataframes into matrices:
    # off-bridge:
    sig_mat_off = df_2_mat(sig_data_off)
    # on-bridge:
    sig_mat_on = df_2_mat(sig_data_on)
    # (4) Execute low-pass filter:
    # off-bridge:
    filt_sig_off = np.zeros(sig_mat_off.shape)
    for i in range(num_sensors):
        filt_sig_off[:,i] = butter_filter(sig_mat_off[:,i], low_freq, 'lowpass', fs, 8)
    # on-bridge:
    filt_sig_on = np.zeros(sig_mat_on.shape)
    for i in range(num_sensors):
        filt_sig_on[:,i] = butter_filter(sig_mat_on[:,i], low_freq, 'lowpass', fs, 8)
    # (5) Extract optimum VMD parameters using off-bridge data:
    opt_mode_off,opt_alpha_off = [],[]
    opt_mode_on,opt_alpha_on = [],[]
    sig_params_opt = {
                    'fs':fs,
                    'low_freq':low_freq,
                    'num_peaks':3,
                    'freq_res':freq_res,
    }
    for sens in range(num_sensors):
        opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_off[:,sens],[100,1000],[2,6],sig_params_opt)
        opt_mode_off.append(opt_mode_temp)
        opt_alpha_off.append(opt_alpha_temp)
        opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_on[:,sens],[100,1000],[2,6],sig_params_opt)
        opt_mode_on.append(opt_mode_temp)
        opt_alpha_on.append(opt_alpha_temp)
    # (6) Decompose signals:
    dec_sig_off,dec_sig_on = [],[]
    for i in range(num_sensors):
        dec_sig_off.append(dec_vmd(filt_sig_off[:,i],opt_alpha_off[i],0,opt_mode_off[i]))
        dec_sig_on.append(dec_vmd(filt_sig_on[:,i],opt_alpha_on[i],0,opt_mode_on[i]))
    # save list of decomposed signals into separate dfs:
    df_dec_off = list_to_df(dec_sig_off,col_labs[1:])
    df_dec_off.to_csv(file_path+veh_class+'/'+case_study+'/vmd_sigs/dec_sigs_off_'+str(pass_num)+'.csv',index=False)
    df_dec_on = list_to_df(dec_sig_on,col_labs[1:])
    df_dec_on.to_csv(file_path+veh_class+'/'+case_study+'/vmd_sigs/dec_sigs_on_'+str(pass_num)+'.csv',index=False)
    # (7) Select first mode:
    # off-bridge:
    sel_dec_off = np.zeros((len(time_off),num_sensors))
    sel_dec_on= np.zeros((len(time_on),num_sensors))
    for count,sig_mat in enumerate(dec_sig_off):
        sel_dec_off[:,count] = sig_mat[0]
        sel_dec_on[:,count] = dec_sig_on[count][0]
    # (8) Compute psd:
    # preallocate lists to store psd matrices of decomposed signals:
    f,sel_psd_off = get_psd_welch(sel_dec_off,freq_res,fs)
    max_psd_off = np.max(sel_psd_off,axis=0)
    idx_max_psd_off = np.argmax(sel_psd_off,axis=0)
    df_dom_veh = pd.DataFrame({'peak_f': f[idx_max_psd_off],'peak_psd':max_psd_off})
    # (9) Compute peak widths:
    p_widths = [peak_widths(sel_psd_off[:,count].T, [idx], rel_height=0.5) for count, idx in enumerate(idx_max_psd_off)]
    f_low = [1 if int(np.floor(p_widths[i][-2])) == 0 else int(np.floor(p_widths[i][-2])) for i in range(num_sensors)] # lower frequency
    f_up = [int(np.ceil(p_widths[i][-1])) for i in range(num_sensors)] # upper frequency
    f_bounds = [f[f_low],f[f_up]]
    p_widths[0]
    # (10) Execute bandstop filter:
    # off-bridge:
    sig_final_off = np.zeros(sel_dec_off.shape)
    for i in range(num_sensors):
        sig_final_off[:,i] = butter_filter(sel_dec_off[:,i], [f_bounds[0][i],f_bounds[1][i]], 'bandstop', fs, 8)
    # on-bridge:
    sig_final_on = np.zeros(sel_dec_on.shape)
    for i in range(num_sensors):
        sig_final_on[:,i] = butter_filter(sel_dec_on[:,i], [f_bounds[0][i],f_bounds[1][i]], 'bandstop', fs, 8)
    # (11) Compute psd:
    # Off-bridge:
    f_off,psd_final_off = get_psd_welch(sig_final_off,freq_res,fs)
    # On-bridge:
    f_on,psd_final_on = get_psd_welch(sig_final_on,freq_res,fs)
    # (12) Peak picking:
    # off-bridge:
    peak_psd_off,peak_f_off = get_peaks(f_off,freq_res*2,psd_final_off,num_peaks)
    # on-bridge:
    peak_psd_on,peak_f_on = get_peaks(f_off,freq_res*2,psd_final_on,num_peaks)
    # (13) Store final results into dataframes:
    # off-bridge:
    res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'pass_num':[]}
    for idx_out,p_psd in enumerate(peak_psd_off):
        for idx_in,elem in enumerate(p_psd):
            res_dict['peak_f'].append(peak_f_off[idx_out][idx_in])
            res_dict['peak_psd'].append(elem)
            res_dict['sensor'].append(col_labs[idx_out+1])
            res_dict['pass_num'].append(pass_num)
    df_off = pd.DataFrame(res_dict)
    # on-bridge:
    res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'pass_num':[]}
    for idx_out,p_psd in enumerate(peak_psd_on):
        for idx_in,elem in enumerate(p_psd):
            res_dict['peak_f'].append(peak_f_on[idx_out][idx_in])
            res_dict['peak_psd'].append(elem)
            res_dict['sensor'].append(col_labs[idx_out+1])
            res_dict['pass_num'].append(pass_num)
    df_on = pd.DataFrame(res_dict)
    return df_off,df_on

if __name__ == '__main__':
    # seaborn plot dictionary:
    plot_style = {
        'font.size': 18.0,
        'font.family':'Times New Roman',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,
        'axes.grid':True,
        'grid.linewidth': 0.8,
        'grid.linestyle':'--',
        'grid.color':'k',
        'lines.linewidth': 2,
        'lines.markersize': 8.0,
        'patch.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 5.5,
        'ytick.major.size': 5.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'legend.title_fontsize': None
    }
    # Inputs:
    low_freq = 50
    bin_width = 0.5
    freq_res = 0.1
    fs = 1000
    file_ID = 1
    num_peaks = 5
    num_passes = 10
    V = 10
    # file inputs:
    file_path = '../../parametric_study/case_studies/Rob_CNN/'
    L = [16,30,40,16,16,16,16,16,16,20,38,56,20,24,28]
    bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num','med_num',
                    'high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
    veh_classes = ['hatchback','SUV','truck']
    # create dictionaries:
    sig_params = {
                    'fs':fs,
                    'low_freq':low_freq,
                    'num_peaks':num_peaks,
                    'freq_res':freq_res,
    }
    for veh_class in veh_classes:
        # print status of case study in txt file:
        status_path = file_path+veh_class+'/status_python.txt'
        for span_idx,bridge_name in enumerate(bridge_names):
            # create directories to store results:
            # off-bridge directory:
            off_bridge_path = file_path+veh_class+'/'+bridge_name+'/off_bridge'
            os.makedirs(off_bridge_path,exist_ok=True)
            # on-bridge directory:
            on_bridge_path = file_path+veh_class+'/'+bridge_name+'/on_bridge'
            os.makedirs(on_bridge_path,exist_ok=True)
            # vmd signal directory:
            vmd_sigs_path = file_path+veh_class+'/'+bridge_name+'/vmd_sigs'
            os.makedirs(vmd_sigs_path,exist_ok=True)
            with open (status_path, "a") as file_object:
                file_object.write(f'case study: {bridge_name}\n')
            # run vmd and peak picking analyses in parallel:
            vmd_runs_dict = Parallel(n_jobs=10,prefer='processes')(delayed(vmd_each_veh_pass)(file_path,veh_class,bridge_name,pass_num,fs,low_freq,freq_res,num_peaks,L[span_idx],V) for pass_num in range(1,num_passes+1))
            # concatenate final results into separate dfs:
            df_off_concat,df_on_concat = pd.DataFrame(),pd.DataFrame()
            for df in vmd_runs_dict:
                df_off_concat = pd.concat((df_off_concat,df[0]))
                df_on_concat = pd.concat((df_on_concat,df[1]))
            # (15) reindex dfs:
            df_on_concat.reset_index(drop=True,inplace=True)
            df_off_concat.reset_index(drop=True,inplace=True)
            # (16) drop frequencies that are repeated less than 20%:
            df_on_concat = drop_rep_freq(df_on_concat,2)
            df_off_concat = drop_rep_freq(df_off_concat,2)
            # (17) separate dfs into respective sensors:
            # on-bridge:
            df_body_on = df_on_concat[df_on_concat['sensor'] == 'body'].copy().reset_index(drop=True)
            df_front_on = df_on_concat[df_on_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True)
            df_rear_on = df_on_concat[df_on_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True)
            # off-bridge:
            df_body_off = df_off_concat[df_off_concat['sensor'] == 'body'].copy().reset_index(drop=True)
            df_front_off = df_off_concat[df_off_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True)
            df_rear_off = df_off_concat[df_off_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True)    
            # (18) Binning dfs for on-bridge data
            # body:
            df_body_on_bin = bin_df(df_body_on,bin_width)
            # front:
            df_front_on_bin = bin_df(df_front_on,bin_width)
            # rear:
            df_rear_on_bin = bin_df(df_rear_on,bin_width)
            # (20) Find common frequencies between front and rear axle sensors
            df_on_common = find_common(df_front_on_bin,df_rear_on_bin)
            df_bins_on = df_on_common.groupby('bins').agg(counts=('count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
            # (21) Binning dfs for on-bridge data
            # body:
            df_body_off_bin = bin_df(df_body_off,bin_width)
            # front:
            df_front_off_bin = bin_df(df_front_off,bin_width)
            # rear:
            df_rear_off_bin = bin_df(df_rear_off,bin_width)
            # (22) Find common frequencies between front and rear axle sensors
            df_off_common = find_common(df_front_off_bin,df_rear_off_bin)
            df_bins_off = df_off_common.groupby('bins').agg(counts=('count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
            # (23) get set difference between on and off bridge data
            df_diff = get_diff_col(df_bins_on,df_bins_off,'bins')
            # (24) Extract exact frequencies from the binned data
            df_freq_final = df_on_common[df_on_common['bins'].isin(df_diff['bins'])].reset_index(drop=True)
            # (25) Compute statistics
            # probability of detecting bridge freq
            df_stat = df_freq_final.copy()
            df_stat['prob'] = df_freq_final['count']/10
            # compute bridge to vehicle mean psd ratio:
            df_stat['mean_ratio'] = df_freq_final['mean_peak_psd']/df_bins_off['mean_peak_psd'].max()
            # add column to distinguish between analyses:
            df_stat['veh_class'] = df_stat.shape[0]*[veh_class]
            df_stat['case'] = df_stat.shape[0]*[bridge_name]
            # (26) Plot histograms:
            fig_on,hist_on = get_hist_freq(df_on_concat,['front_axle','rear_axle'],bin_width,'On-bridge')
            fig_on.savefig(file_path+veh_class+'/'+bridge_name+'/hist_on_'+bridge_name+'.pdf')
            fig_on.clf();hist_on.cla()
            plt.close(plt.gcf())
            fig_off,hist_off = get_hist_freq(df_off_concat,['front_axle','rear_axle'],bin_width,'Off-bridge')
            fig_off.savefig(file_path+veh_class+'/'+bridge_name+'/hist_off_'+bridge_name+'.pdf')
            fig_off.clf();hist_off.cla()
            plt.close(plt.gcf())
            # (27) Save results to csv file:
            # save set difference:
            df_diff.to_csv(file_path+veh_class+'/'+bridge_name+'/set_diff_'+bridge_name+'.csv',index=False)
            # save statistics:
            df_stat.to_csv(file_path+veh_class+'/'+bridge_name+'/stat_'+bridge_name+'.csv',index=False)