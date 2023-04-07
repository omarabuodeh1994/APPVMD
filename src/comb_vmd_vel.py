import pandas as pd
import pdb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import sys
sys.path.insert(1,'../')
from utilities.vmd_post_func import flatten
from functools import reduce
import math 
from input_params import *
from dir_projects import *

def read_appvmd_results(list_inputs,list_dir,cases_to_ignore):
    """This function is used to read results from the vmd_main_par.py script and summarize them in dataframes. 
    Args:
        list_inputs ([objects]): list of objects containing user-defined input parameters defined in input_params.py.
        list_dir ([objects]): list of objects containing directory path of parameter study defined in dir_projects.py.
        cases_to_ignore ([str]): list of strings describing name of case studies to ignore (if possible).
    Returns:
        df: returns a dataframes containing the appvmd results and binned appvmd results.
    """
    vmd_results = pd.DataFrame()
    vmd_binned_results = pd.DataFrame()
    vmd_dominant_results = pd.DataFrame()
    for idx,input_params in enumerate(list_inputs):
        # extract results from vmd analysis
        for veh_vel in input_params.vel:
            for veh_class in input_params.veh_classes:
                bridge_idx = 0
                for bridge_name in input_params.bridge_names:
                    if len(cases_to_ignore) != 0:
                        if bridge_idx < 2:
                            if input_params.case_study+'_'+bridge_name == cases_to_ignore[0][idx]+'_'+cases_to_ignore[idx+1][bridge_idx]:
                                bridge_idx += 1
                                continue
                    vmd_res_temp = pd.read_csv(list_dir[idx].dir+'/'+veh_class+'/'+bridge_name+'/'+'stat_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    vmd_binned_temp = pd.read_csv(list_dir[idx].dir+'/'+veh_class+'/'+bridge_name+'/'+'set_diff_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    # vmd_dom_temp = pd.read_csv(file_path+case_study+'/'+'sedan'+'/'+bridge_name+'/'+'stat_dominant_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    if vmd_res_temp.empty:
                        col_name = vmd_res_temp.columns
                        series_dict = {col_name[0]:[0],col_name[1]:[pd.Interval(0.0,0.5,closed='right')],col_name[2]:[10],col_name[3]:[0],col_name[4]:[0.0],col_name[5]:[0],col_name[6]:[veh_class],col_name[7]:[bridge_name]}
                        vmd_res_temp = pd.concat((vmd_res_temp,pd.DataFrame(series_dict)),axis=0)
                    vmd_res_temp['case_study'] = input_params.case_study 
                    vmd_res_temp['veh_class'] = veh_class
                    vmd_res_temp['velocity'] = veh_vel
                    vmd_binned_temp['case_study'] = [input_params.case_study]*vmd_binned_temp.shape[0]
                    vmd_binned_temp['bridge_name'] = [bridge_name]*vmd_binned_temp.shape[0]
                    vmd_binned_temp['veh_class'] = [veh_class] * vmd_binned_temp.shape[0]
                    vmd_binned_temp['velocity'] = veh_vel
                    vmd_results = pd.concat((vmd_results,vmd_res_temp)).reset_index(drop=True)
                    # vmd_dominant_results = pd.concat((vmd_dominant_results,vmd_dom_temp)).reset_index(drop=True)
                    vmd_binned_results = pd.concat((vmd_binned_results,vmd_binned_temp)).reset_index(drop=True)  
    return vmd_results,vmd_binned_results # ,vmd_dominant_results

def map_with_true_bridge_freq(df,list_input_params):
    """This function is used to match the analytically calculated bridge frequencies using the Euler beam formulations.

    Args:
        df (df): dataframe containing the picked frequencies for each case study.
        case_studies ([object]): list of objects that are defined in input_params.py.

    Returns:
        df: returns dataframe with the frequencies that match the most.
    """
    final_df = pd.DataFrame()
    for input_params in list_input_params:
        for veh_vel in input_params.vel:
            for veh_class in input_params.veh_classes:
                # extract results from vmd analysis
                for bridge_name in input_params.bridge_names:
                    if df[(df['bridge_name'] == bridge_name) & (df['veh_class'] == veh_class) & (df['case_study'] == input_params.case_study) & (df['velocity'] == veh_vel)].empty:
                        continue
                    temp_df = df[(df['bridge_name'] == bridge_name) & (df['veh_class'] == veh_class) & (df['case_study'] == input_params.case_study) & (df['velocity'] == veh_vel)].copy()
                    temp_df['diff'] = (temp_df['peak_f'] - temp_df['bridge_freq_true']).abs()
                    temp_df.sort_values(by=['diff','peak_f_count','mean_ratio'],ascending=[True,False,False],inplace=True)
                    final_df = pd.concat((final_df,pd.DataFrame(temp_df.loc[temp_df.index[0]]).T))
    return final_df.reset_index(drop=True)


def get_flexural_idx_and_frequency(input_params):
    """This function is used to compute the flexural index and frequency of simply supported beams based on geometric and mechanical properties.

    Args:
        input_params ([object]): list of class object defined by the user in input_params.py (it contains bridge properties like L,I,M,E).

    Returns:
        [df]: returns a dataframe containing frequency values and flexural indexes, respectively.
    """
    # bridge properties:
    L = flatten([input_param.bridge_spans for input_param in input_params])
    I = flatten([input_param.bridge_Is for input_param in input_params])
    M = flatten([input_param.bridge_masses for input_param in input_params])
    E = flatten([input_param.bridge_Es for input_param in input_params])

    # calculation:
    bridge_comp_prop_df = pd.DataFrame()
    # closed-form solutions to obtain natural frequencies of beam based on boundary condition type:
        # frequency for pinned-pinned condition
    if input_params[0].boundary_condition == 'pp':
        bridge_comp_prop_df['bridge_freq_true'] = [round(round(np.pi/2*np.sqrt((E[idx]*10**9*I[idx])/(M[idx]*L[idx]**4))/0.1)*0.1,3) for idx in range(len(L))]
    elif input_params[0].boundary_condition == 'ff':
        # frequency for fixed-fixed condition
        bridge_comp_prop_df['bridge_freq_true'] = [round(round((1.5**2*np.pi)/2*np.sqrt((E[idx]*10**9*I[idx])/(M[idx]*L[idx]**4))/0.1)*0.1,3) for idx in range(len(L))]
    elif input_params[0].boundary_condition == 'fp':
        # frequency for fixed-pinned condition
        bridge_comp_prop_df['bridge_freq_true'] = [round(round((1.25**2*np.pi)/2*np.sqrt((E[idx]*10**9*I[idx])/(M[idx]*L[idx]**4))/0.1)*0.1,3) for idx in range(len(L))]
    
    # flexural index computation:
    bridge_comp_prop_df['flexural_index'] = [E[idx]*I[idx]/L[idx] for idx in range(len(L))]

    return bridge_comp_prop_df

def compute_drive_freq(vel,dist,freq_res):
    return round(round(vel/dist/freq_res)*freq_res,3)

def get_dominant_freq(df_binned,df_exact,list_input_params):
    """This function is used to extract the frequencies with the highest counts and psd ratio.

    Args:
        df_binned (df): dataframe containing the peak frequencies and psd ratios in binned structure.
        veh_classes ([str]): list of strings describing the vehicle class used.
        bridge_names ([str]): list of strings describing bridge names during analysis.
        case_studies ([str]): list of strings describing the bridge case study analyzed.

    Returns:
        df: dataframe containing the most dominant frequency for each case study and vehicle class.
    """
    df_binned_final = pd.DataFrame()
    df_exact_final = pd.DataFrame()
    # get the bin frequencies that were repeated the most and had the highest amplitude
    for input_params in list_input_params:
        for veh_class in input_params.veh_classes:
            for bridge_name in input_params.bridge_names:
                if df_binned[(df_binned['bridge_name'] == bridge_name) & (df_binned['veh_class'] == veh_class) & (df_binned['case_study'] == input_params.case_study)].empty:
                    continue
                df_binned_final = pd.concat((df_binned_final,df_binned[(df_binned['case_study']==input_params.case_study) & (df_binned['bridge_name']==bridge_name) & (df_binned['veh_class']==veh_class)].sort_values(['bins_count','mean_peak_psd'],ascending=False,ignore_index=True).loc[0].to_frame()),axis=1)
                df_exact_final = pd.concat((df_exact_final,df_exact[(df_exact['case_study']==input_params.case_study) & (df_exact['bridge_name']==bridge_name) & (df_exact['veh_class']==veh_class)].sort_values(['peak_f_count','mean_peak_psd'],ascending=False,ignore_index=True).loc[0].to_frame()),axis=1)
    df_exact_final = df_exact_final.T.reset_index(drop=True)
    df_binned_final = df_binned_final.T.reset_index(drop=True)
    df_binned_final['peak_f'] = df_exact_final['peak_f']
    df_binned_final['peak_f_count'] = df_exact_final['peak_f_count']
    df_binned_final['peak_f_prob'] = df_binned_final['peak_f_count']/input_params.num_passes
    return df_binned_final

def plot_actual_vs_predicted(data,x_str,y_str,col_lab,col_val,axes_units):
    """This function is used to create a plot of the actual versus predicted variables.

    Args:
        data (df): dataframe of the data.
        x_str (str): string value of the column to plot as x values.
        y_str (str): string value of the column to plot as y values.
        col_lab (str): string values of column label to plot from.
        col_val (str): string value under the specified column label.
        axes_units (str): string value of the units to print for the axes labels.

    Returns:
        fig,axes: returns a matplotlib.pyplot.fig and matplotlib.pyplot.axes objects.
    """
    fig,axs = plt.subplots(1)
    sns.scatterplot(x=x_str,y=y_str,data=data[data[col_lab]==col_val],ax=axs,color='red')
    max_x = data[data[col_lab]==col_val][x_str].max()
    max_y = data[data[col_lab]==col_val][y_str].max()
    max_x_y = int(max([max_x,max_y])) + 1 if int(max([max_x,max_y])) % 2 == 0 else int(max([max_x,max_y]))
    axs.set_xlim([0,max_x_y])
    axs.set_ylim([0,max_x_y])
    plt.plot((0,max_x_y+2),(0,max_x_y+2),ls='--',c='k')
    axs.set_xticks(range(0,max_x_y+2,2))
    axs.set_yticks(range(0,max_x_y+2,2))
    axs.set_xlabel(f'Actual ({axes_units})');axs.set_ylabel(f'Predicted ({axes_units})')
    axs.legend(['Data','Perfect Fit'])
    return fig,axs

def get_bridge_detection(df,freq_threshold):
    """This function is used to create a column that indicates whether the bridge frequency was detected based on the computed error between the predicted and actual.

    Args:
        df (df): dataframe containing the actual and predicted frequencies.
        freq_threshold (float): bridge frequency threshold.

    Returns:
        df: returns the dataframe with a new column that indicates whether the bridge frequency is detected or not detected.
    """
    # check if absolute error is greater than 0.5:
    delta = ((df['bridge_freq_true']-df['peak_f'])).abs() > freq_threshold
    # convert booleans to string:
    delta_str = delta.map({True:'not detected', False:'detected'})
    # compute idx of peak_f col:
    peak_f_idx = df.columns.get_loc('peak_f')
    df.insert(peak_f_idx+1,'bridge_freq_detection',delta_str.to_list())
    return df

def drop_freq(df,freq_value):
    """This function is used to drop frequencies that are less than a certain magnitude.

    Args:
        df (df): dataframe containing the frequency value.
        freq_value (float): frequency value in Hz.

    Returns:
        df: result dataframe after dropping the rows with frequency values less than the the user-specified one.
    """
    idx_to_drop = df[(df['peak_f'] <= freq_value) & (df['peak_f'] > 0.0)].index
    df.drop(idx_to_drop,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def get_number_detections(df,veh_classes,velocities,detection_str,detection_prop,range_to_accept):
    """This function is used to get the number of detection from APPVMD results.

    Args:
        df (df): dataframe containing the aggregated results from the APPVMD algorithm.
        veh_classes ([str]): list of strings describing vehicle names.
        velocities ([float]): list of vehicle velocities used in FE simulation in m/s.
        detection_str (str): string containing whether the function should consider index the series with 'detected' or 'not detected'.
        detection_prop ([float]): float or list depending on the range_to_accept. If range_to_accept = 'greater' or 'less' than only define this variable as a float, else if 'between' then define it as a list with the first element being the lower bound and the second being upper.
        range_to_accept (str): string describing the range to search for (e.g., greater, betwee, less)

    Returns:
        list: list of number of times the function found the detected or not detected.
    """
    if detection_str == 'not detected':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (df['velocity'] == v)].shape[0] for veh_class in veh_classes for v in velocities]

    elif range_to_accept == 'between':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (df['velocity'] == v) & ((dom_freq_df['peak_f_prob'] >= detection_prop[0]) & (dom_freq_df['peak_f_prob'] < detection_prop[1]))].shape[0] for veh_class in veh_classes for v in velocities]

    elif range_to_accept == 'greater':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (df['velocity'] == v) & (dom_freq_df['peak_f_prob'] >= detection_prop)].shape[0] for veh_class in veh_classes for v in velocities]

    else:
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (df['velocity'] == v) & (dom_freq_df['peak_f_prob'] < detection_prop)].shape[0] for veh_class in veh_classes for v in velocities]

def get_detection_summary(veh_classes,velocities,not_detection,low_detection,int_detection,high_detection):
    """This function is used to summarize detection lists into a dataframe.

    Args:
        veh_classes ([str]): list of strings describing vehicle classes used.
        not_detection ([int]): number of false detections.
        low_detection ([int]): number of detections in the low confidence region.
        int_detection ([int]): number of detections in the intermediate confidence region.
        high_detection ([int]): number of detections in the high confidence region.

    Returns:
        df: dataframe summarizing number of detections.
    """
    veh_classes_edit = [f'{veh_class}_{vel}' for veh_class in veh_classes for vel in velocities]
    df_summary = pd.DataFrame({'veh_class':veh_classes_edit,'not_detected':not_detection,'low_count':low_detection,'int_count':int_detection,'high_count':high_detection})
    df_summary['total'] = df_summary[['low_count','int_count','high_count']].sum(axis=1)/df_summary.loc[:,df_summary.columns != 'veh_class'].sum(axis=1)
    return df_summary

def get_statistics(seaborn_plot_format,vehicle_classes,velocities,df,x_str,y_str,col_lab,axes_units):
    with sns.plotting_context(seaborn_plot_format):
    # initialize empty lists
        # lists for figures and axes objects:
        fig_act_vs_pred, axs_act_vs_pred = [],[]

        # lists for r2 and mse values:
        r2,mse = [],[]
        # vehicle classes with velocity:
        veh_classes_edit = [f'{veh_class}_{v}' for veh_class in vehicle_classes for v in velocities]
        # loop through each vehicle class
        for v in velocities:
            for veh_class in vehicle_classes:
                fig_temp,ax_temp = plot_actual_vs_predicted(df[df['velocity'] == v],x_str,y_str,col_lab,veh_class,axes_units)
                ax_temp.set_title(f'{veh_class}_{v}')
                fig_act_vs_pred.append(fig_temp); axs_act_vs_pred.append(ax_temp)
                # compute statistics quantities (e.g., r2 and mse)
                r2.append(r2_score(df[(df['veh_class'] == veh_class) & (df[y_str] > 0) & (df['velocity'] == v)][x_str],df[(df['veh_class'] == veh_class) & (df[y_str] > 0) & (df['velocity'] == v)][y_str]))
                mse.append(mean_squared_error(df[(df['veh_class'] == veh_class) & (df['velocity'] == v)][x_str],df[(df['veh_class'] == veh_class) & (df['velocity'] == v)][y_str]))
        stat_df = pd.DataFrame({'veh_class':veh_classes_edit,'r2':r2,'mse':mse})
        return fig_act_vs_pred,axs_act_vs_pred,stat_df

def plot_hist_prob(plot_format,df,col_labs,bin_width,bar_color):
    """This function is used to plot histogram of probability of success in vbi simulations.

    Args:
        plot_format (dict): dictionary containing the plot format to follow for seaborn plots.
        df (df): df containing the probabilities for each simulation.
        col_labs ([str]): list of strings for which to plot the histogram subplots.
        bin_width ([float]): list of bin widths for each histogram subplot.
        bar_color (str): string describing color of bars used in histogram.

    Returns:
        [fig,ax]: list of figure and axes objects.
    """
    with sns.plotting_context(plot_format):
        fig_hist,ax_hist = plt.subplots(1,3,figsize=(24,14))
        for idx,col_lab in enumerate(col_labs):
            min_val = math.floor(df[col_lab].min()/bin_width[idx])*bin_width[idx]
            max_val = math.ceil(df[col_lab].max()/bin_width[idx])*bin_width[idx]
            hist_sns = sns.histplot(data=df,x=col_lab,bins=np.arange(min_val,max_val+bin_width[idx],bin_width[idx]),ax=ax_hist[idx],color = bar_color)
            hist_sns.set_xticks(np.arange(min_val,max_val+bin_width[idx],bin_width[idx]))
            hist_sns.set_yticks(np.arange(int(min(hist_sns.get_yticks())), int(max(hist_sns.get_yticks())) + 1))
    return fig_hist,ax_hist

def get_bridge_df(list_input_params,flexural_index,frequency):
    """Create dataframe containing bridge properties and frequency values.

    Args:
        list_input_params ([object]): list of objects containing input parameters defined in input_params.py.
        flexural_index ([float]): list of flexural index (EI/L) values of each bridge.
        frequency ([float]): list of analytically calculated frequency values.

    Returns:
        df: dataframe containing bridge properties and frequency values
    """
    n_rows = len(flatten([input_params.bridge_masses for input_params in list_input_params])) # get number of rows of returned dataframe for bridge_id column
    return pd.DataFrame({       'bridge_id': [f'B_{i}' for i in range(1,n_rows+1)],
                                'mass':flatten([input_params.bridge_masses for input_params in list_input_params]),
                                'span':flatten([input_params.bridge_spans for input_params in list_input_params]),
                                'E': flatten([input_params.bridge_Es for input_params in list_input_params]),
                                'I': flatten([input_params.bridge_Is for input_params in list_input_params]),
                                'flexural_index': flexural_index, f'bridge_freq_true': frequency,
                                'bridge_name': flatten([input_params.bridge_names for input_params in list_input_params]),
                                'case_study': flatten([[input_params.case_study]*len(input_params.bridge_names) for input_params in list_input_params])})

def get_span_type(bridge_df,input_params):
    """This function is used to categorize bridge spans based on three categories (short, medium, and long span).

    Args:
        bridge_df (df): dataframe containing computed bridge properties.
        bounds (object): object containing input parameters defined in the input_params.py.

    Returns:
        df: bridge dataframe containing an addition column of span type.
    """
    bridge_df.loc[(bridge_df['span'] <= input_params.bridge_bounds[0]),'span_type'] = 'short_span'
    bridge_df.loc[(bridge_df['span'] > input_params.bridge_bounds[0]) & (bridge_df['span'] <= input_params.bridge_bounds[1]),'span_type'] = 'med_span'
    bridge_df.loc[(bridge_df['span'] > input_params.bridge_bounds[1]),'span_type'] = 'long_span'
    return bridge_df

def get_confidence_regions(df,input_params):
    """This function is used to compute the confidence regions based on the probability of successful bridge frequency extraction.

    Args:
        df (df): dataframe containing the bridge frequency values.
        input_params (object): input parameters defined by the user in input_params.py.

    Returns:
        df: dataframes containing a new column "CFR" (confidence region).
    """
    df.loc[(df['peak_f_prob'] < input_params.freq_cfr_bounds[0]),'CFR'] = 'low CFR'
    df.loc[(df['peak_f_prob'] >= input_params.freq_cfr_bounds[0]) & (df['peak_f_prob'] < input_params.freq_cfr_bounds[1]),'CFR'] = 'inter. CFR'
    df.loc[(df['peak_f_prob'] >= input_params.freq_cfr_bounds[1]),'CFR'] = 'high CFR'
    return df

def get_cr_dfs(df,veh_class):
    """creates separate dataframes for each confidence region.

    Args:
        df (df): dataframe containing data for all confidence regions.
        veh_class (str): string describing which vehicle class is of interest. 

    Returns:
        [df]: list of dataframes that are categorized as not detected, low confidence region, medium confidence region, high confidence region.
    """
    high_cr = df[(df['veh_class']==veh_class) & (df['bridge_freq_detection']=='detected') & (df['CFR'] == 'high CFR')].reset_index(drop=True)
    med_cr = df[(df['veh_class']==veh_class) & (df['bridge_freq_detection']=='detected') & (df['CFR'] == 'inter. CFR')].reset_index(drop=True)
    low_cr = df[(df['veh_class']==veh_class) & (df['bridge_freq_detection']=='detected') & (df['CFR'] == 'low CFR')].reset_index(drop=True)
    not_detected = df[(df['veh_class']==veh_class) & (df['bridge_freq_detection']=='not detected')].reset_index(drop=True)
    return not_detected,low_cr,med_cr,high_cr

def save_figs(list_fig,path,plot_names):
    """This function is used to save a list of figures to a specific path.

    Args:
        list_fig ([matplotlib.figure]): list of matplotlib figures.
        path (str): string describing where figures are saved.
    """
    for idx,fig in enumerate(list_fig):
        fig.savefig(f'{path}/{plot_names[idx]}.pdf',bbox_inches = 'tight')

class LoadInputs:
    def __init__(self):
        # cases to ignore incase of duplicate bridge properties:
        self.cases_to_ignore = [['Case_1','Case_2','Case_3'],['small_thick','med_num'],['low_num','low_num_constant_flexure'],['small_thick','med_num']]
        # case studies of interest
        self.case_studies = [f'Case_{i}' for i in range(1,4)]
        # type of study:
        self.type_of_study = 'mult_veh'
    
    def get_inputs_dir(self):
        list_inputs, list_dir = [],[]
        for case_study in self.case_studies:
            list_inputs.append(InputParams(case_study,self.type_of_study))
            list_dir.append(DirectoryProjects(case_study,self.type_of_study))
        return list_inputs, list_dir
    
if __name__ == '__main__':
    
    # Inputs:
    input_obj = LoadInputs()
    list_inputs,list_dir = input_obj.get_inputs_dir()

    # create dataframe for vehicles:
    veh_df = pd.DataFrame({'veh_class':list_inputs[0].veh_classes,'veh_mass':list_inputs[0].veh_masses})
    # get frequency and flexural index values for each bridge:
    bridge_comp_prop_df = get_flexural_idx_and_frequency(list_inputs)
    
    # store frequency and flexural index vectors:
    frequency,flexural_index = bridge_comp_prop_df['bridge_freq_true'].to_numpy(),bridge_comp_prop_df['flexural_index'].to_numpy()

    # convert frequency vector to df:
    bridge_freq_df = pd.DataFrame({'peak_f':frequency})

    # extract vmd results from parametric study folder:
    vmd_results,vmd_binned_results = read_appvmd_results(list_inputs,list_dir,input_obj.cases_to_ignore)
    
    # vmd_results,vmd_binned_results,vmd_dominant_results = read_vmd_results(file_path,case_studies,bridge_names,veh_classes,V,modes_included,cases_to_ignore)
    vmd_results = pd.merge(vmd_results,veh_df,how='left',on='veh_class')
    
    # create bridge dataframe
    bridge_df = get_bridge_df(list_inputs,flexural_index,frequency)

    # label spans as short, medium and long spans
    bridge_df = get_span_type(bridge_df,list_inputs[0])

    # merge bridge_df with vmd_results to get the frequency and flexural index lined up with the different case studies and vehicle classes:
    merge_vmd_df = pd.merge(bridge_df,vmd_results,how='right',on=['bridge_name','case_study'])
    merge_vmd_df['mass_ratio'] = merge_vmd_df['veh_mass']/(merge_vmd_df['mass']*merge_vmd_df['span'])
    merge_vmd_df['EI'] = merge_vmd_df['E']*merge_vmd_df['I']

    # merge vmd binned results with bridge_df:
    merge_vmd_binned_df = pd.merge(bridge_df,vmd_binned_results,how='right',on=['bridge_name','case_study'])
    dom_count_binned_df = get_dominant_freq(merge_vmd_binned_df,merge_vmd_df,list_inputs)
    
    # map with bridge frequencies based on the Euler beam frequency equation:
    dom_freq_df = map_with_true_bridge_freq(merge_vmd_df,list_inputs) # according to the exact frequency

    # create column specifying whether bridge is detected or not:
    dom_freq_df = get_bridge_detection(dom_freq_df,1) # according to the exact frequency
    dom_count_binned_df = get_bridge_detection(dom_count_binned_df,1) # according to binned frequency

    # Add confidence regions
    dom_freq_df = get_confidence_regions(dom_freq_df,list_inputs[0]) # according to the exact frequency
    dom_freq_df[(dom_freq_df['veh_class'] == 'hatchback') & (dom_freq_df['CFR'] == 'high CFR') & (dom_freq_df['bridge_freq_detection'] == 'detected')]
    dom_count_binned_df = get_confidence_regions(dom_count_binned_df,list_inputs[0]) # according to binned frequency
    dom_count_binned_df[(dom_count_binned_df['veh_class'] == 'hatchback') & (dom_count_binned_df['CFR'] == 'high CFR') & (dom_count_binned_df['bridge_freq_detection'] == 'detected')]

    # count the number of probabilities within certain ranges for the df with exact frequency:
    # not detected:
    n_not_detected_list = get_number_detections(dom_freq_df,list_inputs[0].veh_classes,list_inputs[0].vel,'not detected',0.0,'')

    # greater than 70 % 
    n_high_detected_list = get_number_detections(dom_freq_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',0.7,'greater')
    
    # 40% - 70%: 
    n_int_detected_list = get_number_detections(dom_freq_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',[0.4,0.7],'between')

    # less than 40%
    n_low_detected_list = get_number_detections(dom_freq_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',0.4,'less')

    # count the number of probabilities within certain ranges for the df with binned frequency:
    # not detected:
    n_not_detected_list_binned = get_number_detections(dom_count_binned_df,list_inputs[0].veh_classes,list_inputs[0].vel,'not detected',0.0,'')

    # greater than 70 % 
    n_high_detected_list_binned = get_number_detections(dom_count_binned_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',0.7,'greater')
    
    # 40% - 70%: 
    n_int_detected_list_binned = get_number_detections(dom_count_binned_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',[0.4,0.7],'between')

    # less than 40%
    n_low_detected_list_binned = get_number_detections(dom_count_binned_df,list_inputs[0].veh_classes,list_inputs[0].vel,'detected',0.4,'less')

    # summarize results into dataframe:
    df_counts_exact = get_detection_summary(list_inputs[0].veh_classes,list_inputs[0].vel,n_not_detected_list,n_low_detected_list,n_int_detected_list,n_high_detected_list)
    print(df_counts_exact)
    # df_counts_binned = get_detection_summary(list_inputs[0].veh_classes,list_inputs[0].vel,n_not_detected_list_binned,n_low_detected_list_binned,n_int_detected_list_binned,n_high_detected_list_binned)
    # print(df_counts_binned)
    
    # get statistics summary of dataframes:
    fig_act_vs_pred, axs_act_vs_pred, stat_df = get_statistics(list_inputs[0].plot_format,list_inputs[0].veh_classes,list_inputs[0].vel,dom_freq_df,f'bridge_freq_true','peak_f','veh_class','Hz')
    print(stat_df)
    pdb.set_trace()
    # fig_act_vs_pred_binned, axs_act_vs_pred_binned, stat_df_binned = get_statistics(list_inputs[0].plot_format,list_inputs[0].veh_classes,list_inputs[0].vel,dom_count_binned_df,'bridge_freq_true','peak_f','veh_class','Hz')
    # print(stat_df_binned)
    # save_figs(fig_act_vs_pred,'./results/sedan_velocity',['sedan_9','sedan_13.4','sedan_17.9','sedan_22.4'])
    # plot relational plots of dataframe:
    with sns.plotting_context(list_inputs[0].plot_format):
        g = sns.relplot(data=dom_freq_df,x='flexural_index',y='peak_f_prob',hue='mass',style='span_type',row='bridge_freq_detection',col='veh_class',palette='flare',facet_kws={"margin_titles": True,'despine':False},s=250)
        g.set(ylim=(0,1.1),yticks=np.arange(0,1.1,0.1))
        g.set_xlabels('Flexural Index (GN/m)',clear_inner=False)
        g.set_ylabels('Confidence Region',clear_inner=False)
        g.set_titles(row_template='{row_name}',col_template='{col_name}')
    pdb.set_trace()
    # save_figs([g],'./results/general',['rel_plot_veh'])
    

    # histogram plots
    # Store confidence region df for trucks:
    not_detected_tr,low_cr_tr,med_cr_tr,high_cr_tr = get_cr_dfs(dom_freq_df,'truck')
    pdb.set_trace()
    
    # high cfr:
    fig_hist_high,ax_hist_high = plot_hist_prob(list_inputs[0].plot_format,high_cr_tr,['EI','span','mass'],[5,10,2000],'green')
    fig_hist_high.suptitle('High Confidence Region')
    ax_hist_high[0].set_title('Histogram of Bridge EI');ax_hist_high[1].set_title('Histogram of Bridge Span Lengths');ax_hist_high[2].set_title('Histogram of Bridge Masses')
    ax_hist_high[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_high[1].set_xlabel('Span Length (m)');ax_hist_high[2].set_xlabel('Bridge Mass (kg/m)')
    print(high_cr_tr[['bridge_id','mass','span','EI','flexural_index','bridge_name','case_study']])
    # intermediate cfr:
    fig_hist_int,ax_hist_int = plot_hist_prob(list_inputs[0].plot_format,med_cr_tr,['EI','span','mass'],[20,10,2000],'yellow')
    fig_hist_int.suptitle('Intermediate Confidence Region')
    ax_hist_int[0].set_title('Histogram of Bridge EI');ax_hist_int[1].set_title('Histogram of Bridge Span Lengths');ax_hist_int[2].set_title('Histogram of Bridge Masses')
    ax_hist_int[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_int[1].set_xlabel('Span Length (m)');ax_hist_int[2].set_xlabel('Bridge Mass (kg/m)')
    print(med_cr_tr[['bridge_id','mass','span','EI','flexural_index','bridge_name','case_study']])

    # low cfr:
    fig_hist_low,ax_hist_low = plot_hist_prob(list_inputs[0].plot_format,low_cr_tr,['EI','span','mass'],[30,10,2000],'orange')
    fig_hist_low.suptitle('Low Confidence Region')
    ax_hist_low[0].set_title('Histogram of Bridge EI');ax_hist_low[1].set_title('Histogram of Bridge Span Lengths');ax_hist_low[2].set_title('Histogram of Bridge Masses')
    ax_hist_low[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_low[1].set_xlabel('Span Length (m)');ax_hist_low[2].set_xlabel('Bridge Mass (kg/m)')
    print(low_cr_tr[['bridge_id','mass','span','EI','flexural_index','bridge_name','case_study']])
    # save_figs([fig_hist_low],'./results/histograms',['low_cr'])

    # Undetected:
    fig_hist_undetected,ax_hist_undetected = plot_hist_prob(list_inputs[0].plot_format,not_detected_tr,['EI','span','mass'],[30,10,2000],'red')
    fig_hist_undetected.suptitle('Low Confidence Region')
    ax_hist_undetected[0].set_title('Histogram of Bridge EI');ax_hist_undetected[1].set_title('Histogram of Bridge Span Lengths');ax_hist_undetected[2].set_title('Histogram of Bridge Masses')
    ax_hist_undetected[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_undetected[1].set_xlabel('Span Length (m)');ax_hist_undetected[2].set_xlabel('Bridge Mass (kg/m)')
    print(not_detected_tr[['bridge_id','mass','span','EI','flexural_index','bridge_name','case_study']])
    save_figs([fig_hist_undetected],'./results/histograms',['undetected'])