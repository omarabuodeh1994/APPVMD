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

def vmd_each_veh_pass(file_path,bridge_name,pass_num,fs,low_freq,freq_res,num_peaks,modes_included):
    mode_names = gen_mode_num(modes_included)
    off_bridge_path = file_path+'truck/' + bridge_name + '/off_bridge'
    on_bridge_path = file_path+'truck/' + bridge_name + '/on_bridge'
    # (1) csv file name for off and on bridge data:
    csv_file_off = off_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
    csv_file_on = on_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
    # (2) load data into dataframes:
    # off-bridge:
    sig_data_off = pd.read_csv(csv_file_off)
    sig_len_off = sig_data_off.shape[0]-1 if sig_data_off.shape[0] % 2 else sig_data_off.shape[0]  # length of signal
    sensors = sig_data_off.columns[2:]
    num_sensors = len(sensors) # number of sensors
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
        opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_off[:,sens],[100,2000],[3,7],sig_params_opt)
        opt_mode_off.append(opt_mode_temp)
        opt_alpha_off.append(opt_alpha_temp)
        opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_on[:,sens],[100,2000],[3,7],sig_params_opt)
        opt_mode_on.append(opt_mode_temp)
        opt_alpha_on.append(opt_alpha_temp)
    # opt_mode_off=[3,3]
    # opt_alpha_off=[1000,1000]
    # opt_mode_on=[3,3]
    # opt_alpha_on=[1000,1000]
    # (6) Decompose signals:
    dec_sig_off,dec_sig_on = [],[]
    for i in range(num_sensors):
        dec_sig_off.append(dec_vmd(filt_sig_off[:,i],opt_alpha_off[i],0,opt_mode_off[i]))
        dec_sig_on.append(dec_vmd(filt_sig_on[:,i],opt_alpha_on[i],0,opt_mode_on[i]))
    # save list of decomposed signals into separate dfs:
    df_dec_off = list_to_df(dec_sig_off,sensors)
    df_dec_off.to_csv(file_path+'truck/'+bridge_name+'/vmd_sigs/dec_sigs_off_'+str(pass_num)+'.csv',index=False)
    df_dec_on = list_to_df(dec_sig_on,sensors)
    df_dec_on.to_csv(file_path+'truck/'+bridge_name+'/vmd_sigs/dec_sigs_on_'+str(pass_num)+'.csv',index=False)
    # (7) Select first mode:
    # off-bridge:
    sel_dec_off = np.zeros((len(time_off),num_sensors,modes_included))
    sel_dec_on= np.zeros((len(time_on),num_sensors,modes_included))
    for idx_mode,mode_num in enumerate(mode_names):
        for count,sens in enumerate(sensors):
            sel_dec_off[:,count,idx_mode] = df_dec_off[(df_dec_off['sensor']==sens) & (df_dec_off['mode_num'] == mode_num)]['acc'].values
            sel_dec_on[:,count,idx_mode] = df_dec_on[(df_dec_on['sensor']==sens) & (df_dec_on['mode_num'] == mode_num)]['acc'].values
    # (8) Compute psd:
    # preallocate lists to store psd matrices of decomposed signals:
    f,sel_psd_off = get_psd_welch(sel_dec_off,freq_res,fs)
    idx_max_psd_off = np.argmax(sel_psd_off,axis=0)
    # (9) Compute peak widths:
    f_bounds = []
    for mode_num in range(modes_included):
        f_bounds.append(get_freq_bounds(f,sel_psd_off[:,:,mode_num],idx_max_psd_off[:,mode_num]))
    # (10) Execute bandstop filter:
    # off-bridge:
    sig_final_off = np.zeros(sel_dec_off.shape)
    # on-bridge:
    sig_final_on = np.zeros(sel_dec_on.shape)
    for mode_num in range(modes_included):
        for i in range(num_sensors):
            sig_final_off[:,i,mode_num] = butter_filter(sel_dec_off[:,i,mode_num], [f_bounds[mode_num][0][i],f_bounds[mode_num][1][i]], 'bandstop', fs, 8)
        for i in range(num_sensors):
            sig_final_on[:,i,mode_num] = butter_filter(sel_dec_on[:,i,mode_num], [f_bounds[mode_num][0][i],f_bounds[mode_num][1][i]], 'bandstop', fs, 8)      
    
    # (11) Compute psd:
    # Off-bridge:
    f_off,psd_final_off = get_psd_welch(sig_final_off,freq_res,fs)
    # On-bridge:
    f_on,psd_final_on = get_psd_welch(sig_final_on,freq_res,fs)
    # (12) Peak picking:
    peak_psd_off,peak_f_off = [],[]
    peak_psd_on,peak_f_on = [],[]
    for mode_num in range(modes_included):
        # off-bridge:
        peak_psd_temp,peak_f_temp = get_peaks(f_off,freq_res*2,psd_final_off[:,:,mode_num],num_peaks)
        peak_psd_off.append(peak_psd_temp)
        peak_f_off.append(peak_f_temp)
        # on-bridge:
        peak_psd_temp,peak_f_temp = get_peaks(f_on,freq_res*2,psd_final_on[:,:,mode_num],num_peaks)
        peak_psd_on.append(peak_psd_temp)
        peak_f_on.append(peak_f_temp)
    # (13) Store final results into dataframes:
    # off-bridge:
    res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'mode_num':[],'pass_num':[]}
    for idx_mode,mode_num in enumerate(mode_names):
        for idx_out,p_psd in enumerate(peak_psd_off[idx_mode]):
            for idx_in,elem in enumerate(p_psd):
                res_dict['peak_f'].append(peak_f_off[idx_mode][idx_out][idx_in])
                res_dict['peak_psd'].append(elem)
                res_dict['sensor'].append(sensors[idx_out])
                res_dict['mode_num'].append(mode_num)
                res_dict['pass_num'].append(pass_num)
    df_off = pd.DataFrame(res_dict)
    # on-bridge:
    res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'mode_num':[],'pass_num':[]}
    for idx_mode,mode_num in enumerate(mode_names):
        for idx_out,p_psd in enumerate(peak_psd_on[idx_mode]):
            for idx_in,elem in enumerate(p_psd):
                res_dict['peak_f'].append(peak_f_on[idx_mode][idx_out][idx_in])
                res_dict['peak_psd'].append(elem)
                res_dict['sensor'].append(sensors[idx_out])
                res_dict['mode_num'].append(mode_num)
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
    num_peaks = 10
    num_passes = 10
    modes_included = 3
    V = [10]
    # file inputs:
    L = [20,30,40,40,40,40,40,40,40,40,44,46,15,27,42]
    case_studies = ['fatigue/','Rob_CNN/','Rob_thesis/']
    bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num','med_num',
                    'high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
    # create dictionaries:
    sig_params = {
                    'fs':fs,
                    'low_freq':low_freq,
                    'num_peaks':num_peaks,
                    'freq_res':freq_res,
    }
    for case_study in case_studies:
        file_path = '../parametric_study_stiffness/case_studies/'+case_study
        for bridge_name in bridge_names:
            # print status of case study in txt file:
            status_path = file_path+'truck/status_python.txt'
            # create directories to store results:
            # off-bridge directory:
            off_bridge_path = file_path+'truck/' + bridge_name + '/off_bridge'
            os.makedirs(off_bridge_path,exist_ok=True)
            # on-bridge directory:
            on_bridge_path = file_path+'truck/' + bridge_name + '/on_bridge'
            os.makedirs(on_bridge_path,exist_ok=True)
            # vmd signal directory:
            vmd_sigs_path = file_path + 'truck/' + bridge_name + '/vmd_sigs'
            os.makedirs(vmd_sigs_path,exist_ok=True)
            with open (status_path, "a") as file_object:
                file_object.write(f'case study: {case_study} and bridge name: {bridge_name}\n')
            # run vmd and peak picking analyses in parallel:
            vmd_runs_dict = Parallel(n_jobs=10,prefer='processes')(delayed(vmd_each_veh_pass)(file_path,bridge_name,pass_num,fs,low_freq,freq_res,num_peaks,modes_included) for pass_num in range(1,num_passes+1))
            # concatenate final results into separate dfs:
            df_off_concat,df_on_concat = pd.DataFrame(),pd.DataFrame()
            for df in vmd_runs_dict:
                df_off_concat = pd.concat((df_off_concat,df[0]))
                df_on_concat = pd.concat((df_on_concat,df[1]))
            # vmd_runs_dict = []
            # for pass_num in range(1,num_passes+1):
            #     df_off_temp,df_on_temp = vmd_each_veh_pass(file_path,bridge_name,pass_num,fs,low_freq,freq_res,num_peaks,modes_included)
            #     df_off_concat = pd.concat((df_off_concat,df_off_temp))
            #     df_on_concat = pd.concat((df_on_concat,df_on_temp))
            # (15) reindex dfs:
            df_on_concat.reset_index(drop=True,inplace=True)
            df_off_concat.reset_index(drop=True,inplace=True)
            # (16) drop frequencies that are repeated less than 20%:
            df_on_concat = drop_rep_freq(df_on_concat,2)
            df_off_concat = drop_rep_freq(df_off_concat,2)
            # (17) separate dfs into respective sensors:
            # on-bridge:
            df_front_on = df_on_concat[df_on_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True)
            df_rear_on = df_on_concat[df_on_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True)
            # off-bridge:
            df_front_off = df_off_concat[df_off_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True)
            df_rear_off = df_off_concat[df_off_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True)
            # (18) Binning dfs for on-bridge data
            # front:
            df_front_on_bin = bin_df(df_front_on,bin_width)
            # rear:
            df_rear_on_bin = bin_df(df_rear_on,bin_width)
            # (20) Find common frequencies between front and rear axle sensors
            df_on_common = find_common(df_front_on_bin,df_rear_on_bin)
            df_freq_on = df_on_common.drop('bins',axis=1)
            df_bins_on = df_on_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
            # (21) Binning dfs for on-bridge data
            # front:
            df_front_off_bin = bin_df(df_front_off,bin_width)
            # rear:
            df_rear_off_bin = bin_df(df_rear_off,bin_width)
            # (22) Find common frequencies between front and rear axle sensors
            df_off_common = find_common(df_front_off_bin,df_rear_off_bin)
            df_freq_off = df_off_common.drop('bins',axis=1)
            df_bins_off = df_off_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
            # (23) find difference between on and off bridge peaks:
            df_diff = get_diff_col(df_bins_on,df_bins_off,'bins')
            # (24) Extract exact frequencies from binned data:
            df_freq_final = df_on_common[df_on_common['bins'].isin(df_diff['bins'])].reset_index(drop=True)
            # (25) Compute statistics:
            # probability of detecting bridge freq
            df_stat = df_freq_final.copy()
            df_stat['peak_f_prob'] = df_freq_final['peak_f_count']/10
            # compute bridge to vehicle mean psd ratio:
            df_stat['mean_ratio'] = df_freq_final['mean_peak_psd']/df_bins_off['mean_peak_psd'].max()
            # add column to distinguish between analyses:
            df_stat['veh_class'] = df_stat.shape[0]*['truck']
            df_stat['bridge_name'] = df_stat.shape[0]*[bridge_name]
            # (26) select bin with highest number
            df_diff_sorted = df_diff.sort_values(['bins_count','mean_peak_psd'], ascending = False).reset_index(drop=True)
            dominant_bin = pd.DataFrame(df_diff_sorted.loc[0]).T
            df_stat_dominant = df_stat[df_stat['bins'].isin(dominant_bin['bins'])]
            # (27) Plot histograms:
            fig_on,hist_on = get_hist_freq(df_on_concat,['front_axle','rear_axle'],bin_width,'On-bridge')
            fig_on.savefig(file_path+'truck/'+bridge_name+'/hist_on_'+bridge_name+'.pdf')
            fig_on.clf()
            hist_on.cla()
            plt.close(plt.gcf())
            fig_off,hist_off = get_hist_freq(df_off_concat,['front_axle','rear_axle'],bin_width,'Off-bridge')
            fig_off.savefig(file_path+'truck/'+bridge_name+'/hist_off_'+bridge_name+'.pdf')
            fig_off.clf()
            hist_off.cla()
            plt.close(plt.gcf())
            # (28) Save results to csv file:
            # save set difference:
            df_diff.to_csv(file_path+'truck/'+bridge_name+'/set_diff_'+bridge_name+'.csv',index=False)
            # save statistics:
            df_stat.to_csv(file_path+'truck/'+bridge_name+'/stat_'+bridge_name+'.csv',index=False)
            # save dominant frequencies:
            df_stat_dominant.to_csv(file_path+'truck/'+bridge_name+'/stat_dominant_'+bridge_name+'.csv',index=False)