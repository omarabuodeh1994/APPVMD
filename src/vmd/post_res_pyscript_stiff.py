from statistics import mean
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlb 
from vmdpy import VMD  
import pandas as pd
from vmd_post_func import *
import seaborn as sns
import pdb
from scipy.signal import find_peaks,peak_widths
from functools import reduce
import csv 
from vmd_optim import *

# VMD Analysis:
def vmd_analysis(file_path,veh_class,bridge_name,num_passes,modes_included):
    df_on_concat = pd.DataFrame()
    df_off_concat = pd.DataFrame()
    mode_names = gen_mode_num(modes_included)
    for pass_num in range(1,num_passes+1):  
        # (1) csv file name for decomposed signals of off and on bridge data:
        csv_file_off = file_path+veh_class+'/'+bridge_name+'/vmd_sigs/dec_sigs_off_'+str(pass_num)+'.csv'
        csv_file_on = file_path+veh_class+'/'+bridge_name+'/vmd_sigs/dec_sigs_on_'+str(pass_num)+'.csv'
        
        # (2) read csv file into dataframe:
        df_vmd_sigs_off = pd.read_csv(csv_file_off)
        len_sig_off = len(df_vmd_sigs_off[(df_vmd_sigs_off['mode_num'] == 'mode_1') & (df_vmd_sigs_off['sensor'] == 'front_axle')]['acc'].values)
        df_vmd_sigs_on = pd.read_csv(csv_file_on)
        len_sig_on = len(df_vmd_sigs_on[(df_vmd_sigs_on['mode_num'] == 'mode_1') & (df_vmd_sigs_on['sensor'] == 'front_axle')]['acc'].values)
        
        # (3) Select two modes:
        sel_dec_off = np.zeros((len_sig_off,num_sensors,modes_included))
        sel_dec_on= np.zeros((len_sig_on,num_sensors,modes_included))
        for idx_mode,mode_num in enumerate(mode_names):
            for count,sens in enumerate(sensors):
                sel_dec_off[:,count,idx_mode] = df_vmd_sigs_off[(df_vmd_sigs_off['sensor']==sens) & (df_vmd_sigs_off['mode_num'] == mode_num)]['acc'].values
                sel_dec_on[:,count,idx_mode] = df_vmd_sigs_on[(df_vmd_sigs_on['sensor']==sens) & (df_vmd_sigs_on['mode_num'] == mode_num)]['acc'].values

        # (4) Compute psd:
        # preallocate lists to store psd matrices of decomposed signals:
        f,sel_psd_off = get_psd_welch(sel_dec_off,freq_res,fs)
        idx_max_psd_off = np.argmax(sel_psd_off,axis=0)
        
        # (5) Compute peak widths:
        f_bounds = []
        for mode_num in range(modes_included):
            f_bounds.append(get_freq_bounds(f,sel_psd_off[:,:,mode_num],idx_max_psd_off[:,mode_num]))
        
        # (6) Execute bandstop filter:
        # off-bridge:
        sig_final_off = np.zeros(sel_dec_off.shape)
        # on-bridge:
        sig_final_on = np.zeros(sel_dec_on.shape)
        for mode_num in range(modes_included):
            for i in range(num_sensors):
                sig_final_off[:,i,mode_num] = butter_filter(sel_dec_off[:,i,mode_num], [f_bounds[mode_num][0][i],f_bounds[mode_num][1][i]], 'bandstop', fs, 8)
            for i in range(num_sensors):
                sig_final_on[:,i,mode_num] = butter_filter(sel_dec_on[:,i,mode_num], [f_bounds[mode_num][0][i],f_bounds[mode_num][1][i]], 'bandstop', fs, 8)      
        
        # (7) Compute psd:
        # Off-bridge:
        f_off,psd_final_off = get_psd_welch(sig_final_off,freq_res,fs)
        # On-bridge:
        f_on,psd_final_on = get_psd_welch(sig_final_on,freq_res,fs)
        
        # (8) Peak picking:
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
        
        # (9) Store final results into dataframes:
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
        # (10) Concatenate dfs into one final df:
        # off-bridge:
        df_off_concat = pd.concat((df_off_concat,df_off))
        # on-bridge:
        df_on_concat = pd.concat((df_on_concat,df_on))
        
    # reset indices:
    return df_off_concat.reset_index(drop=True),df_on_concat.reset_index(drop=True)

import itertools
if __name__ == '__main__':
    # (1) Inputs:
    V = 10
    num_passes = 10
    freq_res = 0.1
    bin_width = 0.5
    fs = 1000
    num_peaks = 10
    sensors = ['front_axle','rear_axle']
    num_sensors = 2
    modes_included = 4

    # (2) File Inputs:
    file_path = '../../parametric_studies/parametric_study_tire_stiffness_1/case_studies/'
    case_studies = ['Rob_CNN','Rob_thesis','fatigue']
    L = [16,30,40,16,16,16,16,16,16,20,38,56,20,24,28]
    bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num','med_num',
                    'high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
    veh_classes = ['truck']

    # (3) Seaborn Plotting Style:
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

    # (4) Execute VMD analysis and extraction:
    for case_study in case_studies:
        for veh_class, bridge_name in itertools.product(veh_classes, bridge_names):
            file_path_final = file_path + case_study + '/'
            df_off_concat,df_on_concat = vmd_analysis(file_path_final,veh_class,bridge_name,num_passes,modes_included)
            print(f'case study: {case_study}, vehicle class: {veh_class}, bridge name: {bridge_name}')
            # (5) Drop frequencies that are repeated less than 20% of the time:
            df_on_concat = drop_rep_freq(df_on_concat,2).reset_index(drop=True)
            df_off_concat = drop_rep_freq(df_off_concat,2).reset_index(drop=True)

            # (6) Separate dfs into respective sensors off-bridge:
            # on-bridge:
            df_front_on = df_on_concat[df_on_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True).drop('sensor',axis=1)
            df_rear_on = df_on_concat[df_on_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True).drop('sensor',axis=1)
            # off-bridge:
            df_front_off = df_off_concat[df_off_concat['sensor'] == 'front_axle'].copy().reset_index(drop=True).drop('sensor',axis=1)
            df_rear_off = df_off_concat[df_off_concat['sensor'] == 'rear_axle'].copy().reset_index(drop=True).drop('sensor',axis=1)

            # (7) Bin data for on-bridge:
            # on-bridge:
            df_front_on_bin = bin_df(df_front_on,bin_width)
            df_rear_on_bin = bin_df(df_rear_on,bin_width)
            # off-bridge:
            df_front_off_bin = bin_df(df_front_off,bin_width)
            df_rear_off_bin = bin_df(df_rear_off,bin_width)

            # (8) get common frequencies between front and rear axles:
            # on-bridge:
            df_on_common = find_common(df_front_on_bin,df_rear_on_bin)
            df_freq_on = df_on_common.drop('bins',axis=1)
            df_bins_on = df_on_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()

            # off-bridge:
            df_off_common = find_common(df_front_off_bin,df_rear_off_bin)
            df_freq_off = df_off_common.drop('bins',axis=1)
            df_bins_off = df_off_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()

            # (9) find difference between on and off bridge peaks:
            df_diff = get_diff_col(df_bins_on,df_bins_off,'bins')

            # (10) Extract exact frequencies from binned data:
            df_freq_final = df_on_common[df_on_common['bins'].isin(df_diff['bins'])].reset_index(drop=True)

            # (11) Compute statistics:
            # probability of detecting bridge freq
            df_stat = df_freq_final.copy()
            df_stat['peak_f_prob'] = df_freq_final['peak_f_count']/10
            
            # compute bridge to vehicle mean psd ratio:
            df_stat['mean_ratio'] = df_freq_final['mean_peak_psd']/df_bins_off['mean_peak_psd'].max()
            # add column to distinguish between analyses:
            df_stat['veh_class'] = df_stat.shape[0]*[veh_class]
            df_stat['bridge_name'] = df_stat.shape[0]*[bridge_name]
            
            # (12) select bin with highest number
            df_diff_sorted = df_diff.sort_values(['bins_count','mean_peak_psd'], ascending = False).reset_index(drop=True)
            dominant_bin = pd.DataFrame(df_diff_sorted.loc[0]).T
            df_stat_dominant = df_stat[df_stat['bins'].isin(dominant_bin['bins'])]
            # (12) save results in csv files:
            # save set difference:
            df_diff.to_csv(file_path_final+veh_class+'/'+bridge_name+'/set_diff_'+bridge_name+'_'+str(modes_included)+'_modes.csv',index=False)
            # save statistics:
            df_stat.to_csv(file_path_final+veh_class+'/'+bridge_name+'/stat_'+bridge_name+'_'+str(modes_included)+'_modes.csv',index=False)
            # save dominant frequencies:
            df_stat_dominant.to_csv(file_path_final+veh_class+'/'+bridge_name+'/stat_dominant_'+bridge_name+'_'+str(modes_included)+'_modes.csv',index=False)