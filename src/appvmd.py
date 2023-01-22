import sys
# sys.path('./utilities/')
from utilities.vmd_post_func import *
from utilities.vmd_optim import *
import os
import pandas as pd
import numpy as np
import seaborn as sns
import pdb
from joblib import Parallel, delayed
from input_params import *
from dir_projects import *

class APPVMD:
    def __init__(self):
        return True
    
    def appvmd(dir_project,dir_idx, input_params,veh_class, bridge_name, pass_num):
        """This function is used to employ the first portion of the APPVMD algorithm, where the peaks of each vehicle pass (on and off bridge) are computed and stored in dfs.

        Args:
            dir_project (object): class object containing the input parameters and defined in input_params.py.
            input_params (object): class object containing the input parameters and defined in input_params.py.
            veh_class (str): string describing vehicle class or model used in vbi test.
            bridge_name (str): string describing bridge case used in vbi test.
            pass_num (int): ID describing the pass number of vehicle.

        Returns:
            [df,df]: two dataframes containing peak frequency, peak psd, sensor location, pass number, and mode number for each vehicle pass.
        """
        # off-bridge directory:
        off_bridge_path = dir_project.dir + veh_class + '/' + bridge_name + dir_project.off_bridge_dir[dir_idx]
        os.makedirs(off_bridge_path,exist_ok=True)
        
        # on-bridge directory:
        on_bridge_path = dir_project.dir + veh_class + '/' + bridge_name + dir_project.on_bridge_dir[dir_idx]
        os.makedirs(on_bridge_path,exist_ok=True)
        
        # vmd signal directory:
        vmd_sigs_path = dir_project.dir + veh_class + '/' + bridge_name + dir_project.vmd_sigs_dir[dir_idx]
        os.makedirs(vmd_sigs_path,exist_ok=True)
        # csv file name for off and on bridge data:
        csv_file_off = off_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
        csv_file_on = on_bridge_path+'/veh_pass_'+str(pass_num)+'.csv'
        
        # load data into dataframes:
        # off-bridge:
        sig_data_off = pd.read_csv(csv_file_off)
        sig_len_off = sig_data_off.shape[0]-1 if sig_data_off.shape[0] % 2 else sig_data_off.shape[0]  # length of signal
        num_sensors = len(input_params.sensor_names) # number of sensors
        # on-bridge:
        sig_data_on = pd.read_csv(csv_file_on)
        sig_len_on = sig_data_on.shape[0]-1 if sig_data_on.shape[0] % 2 else sig_data_on.shape[0]
        
        # convert dataframes into matrices:
        # off-bridge:
        sig_mat_off = df_2_mat(sig_data_off[input_params.sensor_names])
        # on-bridge:
        sig_mat_on = df_2_mat(sig_data_on[input_params.sensor_names])
        
        # Execute low-pass filter:
        # off-bridge:
        filt_sig_off = np.zeros(sig_mat_off.shape)
        for i in range(num_sensors):
            filt_sig_off[:,i] = butter_filter(sig_mat_off[:,i], input_params.cut_off_freq, 'lowpass', input_params.fs, 8)
        # on-bridge:
        filt_sig_on = np.zeros(sig_mat_on.shape)
        for i in range(num_sensors):
            filt_sig_on[:,i] = butter_filter(sig_mat_on[:,i], input_params.cut_off_freq, 'lowpass', input_params.fs, 8)
        
        ### Debug
        # fig,axs = plt.subplots(2,1)
        # for idx in range(2):
        #     axs[idx].plot(sig_mat_on[:,idx],ls='-',c='red'); axs[idx].set_title(f'{input_params.sensor_names[idx]}')
        #     axs[idx].plot(filt_sig_on[:,idx],ls='--',c='black'); axs[idx].set_title(f'{input_params.sensor_names[idx]}')
        #     axs[idx].legend(['raw acceleration','filtered acceleration'], loc='upper left')

        # Extract optimum VMD parameters using off-bridge data:
        opt_mode_off,opt_alpha_off = [],[]
        opt_mode_on,opt_alpha_on = [],[]
        sig_params_opt = {
                    'fs':input_params.fs,
                    'low_freq':input_params.cut_off_freq,
                    'num_peaks':3,
                    'freq_res':input_params.freq_res,
        }

        # loop through each sensors and obtain vmd parameters using get_opt_vmd_params()
        # for sens in range(num_sensors):
        #     opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_off[:,sens],[100,2000],[4,7],sig_params_opt)
        #     opt_mode_off.append(opt_mode_temp)
        #     opt_alpha_off.append(opt_alpha_temp)
        #     opt_mode_temp, opt_alpha_temp = get_opt_vmd_params (filt_sig_on[:,sens],[100,2000],[4,7],sig_params_opt)
        #     opt_mode_on.append(opt_mode_temp)
        #     opt_alpha_on.append(opt_alpha_temp)
        
        # opt_mode_off = [8,8]
        # opt_mode_on = [8,8]
        # opt_alpha_off = [1000,1000]
        # opt_alpha_on = [1000,1000]

        # # Decompose signals:
        dec_sig_off,dec_sig_on = [],[]
        # for i in range(num_sensors):
        #     dec_sig_off.append(dec_vmd(filt_sig_off[:,i],opt_alpha_off[i],0,opt_mode_off[i]))
        #     dec_sig_on.append(dec_vmd(filt_sig_on[:,i],opt_alpha_on[i],0,opt_mode_on[i]))
        
        df_dec_off = pd.read_csv(dir_project.dir + veh_class + '/' + bridge_name + dir_project.vmd_sigs_dir[dir_idx] +'/dec_sigs_off_' + str(pass_num) + '.csv')
        df_dec_on = pd.read_csv(dir_project.dir + veh_class + '/' + bridge_name + dir_project.vmd_sigs_dir[dir_idx] +'/dec_sigs_on_' + str(pass_num) + '.csv')
        for sens_idx,sensor_name in enumerate(input_params.sensor_names):
            arr_off = np.zeros((4,sig_len_off))
            arr_on = np.zeros((4,sig_len_on))
            for mode_idx,mode_num in enumerate([f'mode_{i}' for i in range(1,5)]):
                arr_off[mode_idx,:] = df_dec_off[(df_dec_off['mode_num'] == mode_num) & (df_dec_off['sensor'] == sensor_name)]['acc'].to_numpy()
                arr_on[mode_idx,:] = df_dec_on[(df_dec_on['mode_num'] == mode_num) & (df_dec_on['sensor'] == sensor_name)]['acc'].to_numpy()
            dec_sig_off.append(arr_off)
            dec_sig_on.append(arr_on)
        # save list of decomposed signals into separate dfs:
        # df_dec_off = list_to_df(dec_sig_off,input_params.sensor_names)
        # df_dec_off.to_csv(dir_project.dir + veh_class + '/' + bridge_name + dir_project.vmd_sigs_dir[dir_idx] +'/dec_sigs_off_' + str(pass_num) + '.csv', index=False)

        # df_dec_on = list_to_df(dec_sig_on,input_params.sensor_names)
        # df_dec_on.to_csv(dir_project.dir + veh_class + '/' +bridge_name + dir_project.vmd_sigs_dir[dir_idx] +'/dec_sigs_on_' + str(pass_num) + '.csv', index=False)
        
        ## Debug
        # for dec_sig_on_arr in dec_sig_on:
        #     fig,axs = plt.subplots(dec_sig_on_arr.shape[0],1)
        #     for idx,sig_mode_i in enumerate(dec_sig_on_arr):
        #         f,psd_i = get_psd_welch(sig_mode_i,input_params.freq_res,input_params.fs)
        #         axs[idx].plot(f,psd_i);axs[idx].set_xlim([0,50])
        #     plt.show()
        # off-bridge:
        sel_dec_off = np.zeros((sig_len_off,num_sensors,input_params.num_modes))
        for count,sig_mat in enumerate(dec_sig_off):
            sel_dec_off[:,count,:] = sig_mat[:input_params.num_modes,:].T
        
        # on-bridge:
        sel_dec_on= np.zeros((sig_len_on,num_sensors,input_params.num_modes))
        for count,sig_mat in enumerate(dec_sig_on):
            sel_dec_on[:,count,:] = sig_mat[:input_params.num_modes,:].T
        
        
        # Compute psd for off-bridge data and compute indices of maximum psd amplitudes:
        f_off,sel_psd_off = get_psd_welch(sel_dec_off,input_params.freq_res,input_params.fs)
        f_on,sel_psd_on = get_psd_welch(sel_dec_on,input_params.freq_res,input_params.fs)
        idx_max_psd_off = np.argmax(sel_psd_off,axis=0)
        
        # Compute peak widths using indices of max psd during off-bridge portion:
        f_low,f_up = get_freq_bounds(f_off,sel_psd_off,idx_max_psd_off,input_params.num_modes)
        
        ### Debug
        # fig,axs = plt.subplots(input_params.num_modes,input_params.num_sensors)
        # count = 0
        # for mode_num in range(input_params.num_modes):
        #     for sensor_id in range(num_sensors):
        #         idx_psd_low = np.where(f_off == f_low[mode_num,sensor_id])[0][0];idx_psd_up = np.where(f_off == f_up[mode_num,sensor_id])[0][0]
        #         axs[mode_num,sensor_id].plot(f_off,sel_psd_off[:,sensor_id,mode_num],ls='--',color='green')
        #         axs[mode_num,sensor_id].scatter(f_off[idx_max_psd_off[sensor_id,mode_num]],sel_psd_off[idx_max_psd_off[sensor_id,mode_num],sensor_id,mode_num],marker='x',color='blue')
        #         axs[mode_num,sensor_id].scatter([f_low[mode_num,sensor_id],f_up[mode_num,sensor_id]],[sel_psd_off[idx_psd_low,sensor_id,mode_num],sel_psd_off[idx_psd_up,sensor_id,mode_num]],marker='x',color='red')
        #         axs[mode_num,sensor_id].set_xlim([0,50])
        # plt.show()

        
        # Execute bandstop filter:
        # off-bridge:
        sig_final_off = np.zeros(sel_dec_off.shape)
        for mode_num in range(input_params.num_modes):
            for i in range(num_sensors):
                sig_final_off[:,i,mode_num] = butter_filter(sel_dec_off[:,i,mode_num], [f_low[mode_num,i],f_up[mode_num,i]], 'bandstop', input_params.fs, 8)
        # on-bridge:
        sig_final_on = np.zeros(sel_dec_on.shape)
        for mode_num in range(input_params.num_modes):
            for i in range(num_sensors):
                sig_final_on[:,i,mode_num] = butter_filter(sel_dec_on[:,i,mode_num], [f_low[mode_num,i],f_up[mode_num,i]], 'bandstop', input_params.fs, 8)
        
        # Compute psd:
        # Off-bridge:
        f_off,psd_final_off = get_psd_welch(sig_final_off,input_params.freq_res,input_params.fs)
        # On-bridge:
        f_on,psd_final_on = get_psd_welch(sig_final_on,input_params.freq_res,input_params.fs)
        
        # Peak picking:
        # off-bridge:
        peak_psd_off,peak_f_off = get_peaks(f_off,input_params.freq_res*2,psd_final_off,input_params.num_peaks)
        # on-bridge:
        peak_psd_on,peak_f_on = get_peaks(f_off,input_params.freq_res*2,psd_final_on,input_params.num_peaks)
        
        # Uncomment to check if the peaks are being detected.
        # fig,axs = plt.subplots(input_params.num_modes,input_params.num_sensors)
        # count = 0
        # for mode_num in range(input_params.num_modes):
        #     for sensor_id in range(num_sensors):
        #         axs[mode_num,sensor_id].plot(f_on,sel_psd_on[:,sensor_id,mode_num],ls='--',color='grey')
        #         axs[mode_num,sensor_id].plot(f_on,psd_final_on[:,sensor_id,mode_num],ls='-')
        #         axs[mode_num,sensor_id].scatter(peak_f_on[count],peak_psd_on[count],marker='x',color='red')
        #         axs[mode_num,sensor_id].set_xlim([0,50])
        #         count += 1
        # plt.show()
        
        # Store final results into dataframes:
        # off-bridge:
        res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'pass_num':[],'mode_num':[]}
        for idx_out,p_psd in enumerate(peak_psd_off):
            for idx_in,elem in enumerate(p_psd):
                res_dict['peak_f'].append(peak_f_off[idx_out][idx_in])
                res_dict['peak_psd'].append(elem)
                res_dict['sensor'].append(input_params.sensor_names[idx_out%2])
                res_dict['pass_num'].append(pass_num)
                res_dict['mode_num'].append(int(np.floor(idx_out/2)+1))
        df_off = pd.DataFrame(res_dict)

        # on-bridge:
        res_dict = {'peak_f':[],'peak_psd':[],'sensor':[],'pass_num':[],'mode_num':[]}
        for idx_out,p_psd in enumerate(peak_psd_on):
            for idx_in,elem in enumerate(p_psd):
                res_dict['peak_f'].append(peak_f_on[idx_out][idx_in])
                res_dict['peak_psd'].append(elem)
                res_dict['sensor'].append(input_params.sensor_names[idx_out%2])
                res_dict['pass_num'].append(pass_num)
                res_dict['mode_num'].append(int(np.floor(idx_out/2)+1))
        df_on = pd.DataFrame(res_dict)
        return df_off,df_on

    def post_appvmd(input_params,dir_params,parallel):
        for idx,vel_i in enumerate(input_params.vel):
            # loop through each vehicle class
            for veh_class in input_params.veh_classes:
                
                # print status of case study in txt file:
                status_path = dir_params.dir+veh_class+'/status_python.txt'
                
                # loop through each bridge name
                for bridge_name in input_params.bridge_names:
                    
                    # create directories to store results:
                    # off-bridge directory:
                    off_bridge_path = dir_params.dir+veh_class+'/'+bridge_name+dir_params.off_bridge_dir[idx]
                    os.makedirs(off_bridge_path,exist_ok=True)
                    # on-bridge directory:
                    on_bridge_path = dir_params.dir+veh_class+'/'+bridge_name+dir_params.on_bridge_dir[idx]
                    os.makedirs(on_bridge_path,exist_ok=True)
                    # vmd signal directory:
                    vmd_sigs_path = dir_params.dir+veh_class+'/'+bridge_name+dir_params.vmd_sigs_dir[idx]
                    os.makedirs(vmd_sigs_path,exist_ok=True)
                    
                    # open status file
                    with open (status_path, "a") as file_object:
                        file_object.write(f'case study: {bridge_name}\n')
                        print(f'case study: {bridge_name}\n')
                    
                    # run vmd and peak picking analyses in parallel:
                    vmd_runs_dict = parallel(delayed(APPVMD.appvmd)(dir_params, idx, input_params,veh_class, bridge_name, pass_num) for pass_num in range(1,input_params.num_passes+1))
                    
                    # concatenate final results into separate dfs:
                    df_off_concat,df_on_concat = pd.DataFrame(),pd.DataFrame()
                    for df in vmd_runs_dict:
                        df_off_concat = pd.concat((df_off_concat,df[0]))
                        df_on_concat = pd.concat((df_on_concat,df[1]))
                    # reindex dfs:
                    df_on_concat.reset_index(drop=True,inplace=True)
                    df_off_concat.reset_index(drop=True,inplace=True)
                    
                    # drop frequencies that are repeated less than 20%:
                    df_on_concat = drop_rep_freq(df_on_concat,2)
                    df_off_concat = drop_rep_freq(df_off_concat,2)
                    
                    # separate dfs into respective sensors:
                    df_sensors_on = separate_dfs(df_on_concat,input_params.sensor_names) # on-bridge:
                    df_sensors_off = separate_dfs(df_off_concat,input_params.sensor_names) # off-bridge:
                    
                    # Binning dfs:
                    df_bin_on = bin_df(df_sensors_on,input_params) # on-bridge:
                    df_bin_off = bin_df(df_sensors_off,input_params) # off-bridge:

                    # Find common frequencies between front and rear axle sensors
                    # on-bridge 
                    df_on_common = find_common(df_bin_on[0],df_bin_on[1])
                    df_bins_on = df_on_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
                    
                    # Find common frequencies between front and rear axle sensors
                    df_off_common = find_common(df_bin_off[0],df_bin_off[1])
                    df_bins_off = df_off_common.groupby('bins').agg(bins_count=('peak_f_count','sum'),mean_peak_psd=('mean_peak_psd','mean')).dropna().reset_index()
                    
                    # get set difference between on and off bridge data
                    df_diff = get_diff_col(df_bins_on,df_bins_off,'bins')
                    
                    # Extract exact frequencies from the binned data
                    if df_diff['mean_peak_psd'].loc[0] == 0:
                        df_freq_final = pd.DataFrame(columns=df_on_common.columns)
                        df_freq_final['peak_f'] = [0]
                        df_freq_final['peak_f_count'] = [1]
                        df_freq_final['bins'] = df_diff['bins']
                        df_freq_final['mean_peak_psd'] = [0]
                    else:
                        df_freq_final = df_on_common[df_on_common['bins'].isin(df_diff['bins'])].reset_index(drop=True)

                    # Compute statistics
                    # probability of detecting bridge freq
                    df_stat = df_freq_final.copy()
                    df_stat['peak_f_prob'] = df_freq_final['peak_f_count']/10
                    # compute bridge to vehicle mean psd ratio:
                    df_stat['mean_ratio'] = df_freq_final['mean_peak_psd']/df_bins_off['mean_peak_psd'].max()
                    # add column to distinguish between analyses:
                    df_stat['veh_class'] = df_stat.shape[0]*[f'{veh_class}']
                    df_stat['veh_velocity'] = df_stat.shape[0]*[f'{vel_i}']
                    df_stat['bridge_name'] = df_stat.shape[0]*[bridge_name]
                    # Save results to csv file:
                    # save set difference:
                    df_diff.to_csv(dir_params.dir+veh_class+'/'+bridge_name+'/set_diff_'+bridge_name+'_'+str(vel_i)+'.csv',index=False)
                    # save statistics:
                    df_stat.to_csv(dir_params.dir+veh_class+'/'+bridge_name+'/stat_'+bridge_name+'_'+str(vel_i)+'.csv',index=False)

    def main_regular(dir_params,input_params):
        """This function is used to call the appvmd and then bin the picked peaks based on taking the common/differences between front and rear sensors.
        Function can be used for a single vehicle run or multiple vehicle runs.

        Args:
            dir_params (object): object containing directory strings defined in dir_projects.py
            input_params (object): object containing input parameters defined in input_params.py
        """
        if input_params.parll_bool == True:
            with Parallel(n_jobs=input_params.num_jobs_appvmd,prefer='processes') as parallel:
                APPVMD.post_appvmd(input_params,dir_params,parallel)
        else: # no parallelization
            with Parallel(n_jobs=1,prefer='processes') as parallel:
                APPVMD.post_appvmd(input_params,dir_params,parallel)

    def multi_veh_study(case_study,study_type):
        input_params = InputParams(case_study,study_type)
        dir_project = DirectoryProjects(case_study,study_type)
        APPVMD.main_regular(dir_project,input_params)  

if __name__ == '__main__':
    # case_study = sys.argv[1] # for passing case studies through the cluster
    for case_study in [f'Case_{i}' for i in range(1,4)]:
        print(case_study)
        APPVMD.multi_veh_study(case_study,'mult_veh')