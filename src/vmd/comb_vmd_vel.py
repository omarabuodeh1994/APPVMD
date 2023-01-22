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

def read_vmd_results(file_path,case_studies,bridge_names,veh_classes,veh_vels,modes_included,cases_to_ignore):
    """This function is used to read results from the vmd_main_par.py script and summarize them in dataframes. 
    Args:
        file_path (str): string describing file path of parametric study.
        case_studies ([str]): list of strings describing case study names during analysis.
        veh_classes ([str]): list of strings describing vehicle models used during analysis.
        bridge_names ([str]): list of strings describing name of bridges investigated during analysis.
    Returns:
        df: returns a dataframes containing the vmd results.
    """
    vmd_results = pd.DataFrame()
    vmd_binned_results = pd.DataFrame()
    vmd_dominant_results = pd.DataFrame()
    for case_idx,case_study in enumerate(case_studies):
        # extract results from vmd analysis
        for veh_vel in veh_vels:
            for veh_class in veh_classes:
                bridge_idx = 0
                for bridge_name in bridge_names:
                    if bridge_idx < 2:
                        if case_study+'_'+bridge_name == cases_to_ignore[0][case_idx]+'_'+cases_to_ignore[case_idx+1][bridge_idx]:
                            bridge_idx += 1
                            continue
                    # vmd_res_temp = pd.read_csv(file_path+case_study+'/'+'sedan'+'/'+bridge_name+'/'+'stat_'+bridge_name+'_V_'+str(veh_vel)+'_'+str(modes_included)+'_modes'+'.csv')
                    # vmd_binned_temp = pd.read_csv(file_path+case_study+'/'+'sedan'+'/'+bridge_name+'/'+'set_diff_'+bridge_name+'_V_'+str(veh_vel)+'_'+str(modes_included)+'_modes'+'.csv')
                    # vmd_dom_temp = pd.read_csv(file_path+case_study+'/'+'sedan'+'/'+bridge_name+'/'+'stat_dominant_'+bridge_name+'_V_'+str(veh_vel)+'_'+str(modes_included)+'_modes'+'.csv')
                    vmd_res_temp = pd.read_csv(file_path+case_study+'/'+veh_class+'/'+bridge_name+'/'+'stat_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    vmd_binned_temp = pd.read_csv(file_path+case_study+'/'+veh_class+'/'+bridge_name+'/'+'set_diff_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    # vmd_dom_temp = pd.read_csv(file_path+case_study+'/'+'sedan'+'/'+bridge_name+'/'+'stat_dominant_'+bridge_name+'_'+str(veh_vel)+'.csv')
                    if vmd_res_temp.empty:
                        col_name = vmd_res_temp.columns
                        series_dict = {col_name[0]:[0],col_name[1]:[pd.Interval(0.0,0.5,closed='right')],col_name[2]:[10],col_name[3]:[0],col_name[4]:[0.0],col_name[5]:[0],col_name[6]:[veh_class],col_name[7]:[bridge_name]}
                        vmd_res_temp = pd.concat((vmd_res_temp,pd.DataFrame(series_dict)),axis=0)
                    vmd_res_temp['case_study'] = case_study 
                    vmd_res_temp['veh_class'] = f'{veh_class}_{str(veh_vel)}'
                    # vmd_dom_temp['case_study'] = case_study
                    # vmd_dom_temp['veh_class'] = f'sedan_{str(veh_vel)}'
                    vmd_binned_temp['case_study'] = [case_study]*vmd_binned_temp.shape[0]
                    vmd_binned_temp['bridge_name'] = [bridge_name]*vmd_binned_temp.shape[0]
                    vmd_binned_temp['veh_class'] = [f'{veh_class}_{str(veh_vel)}'] * vmd_binned_temp.shape[0]

                    vmd_results = pd.concat((vmd_results,vmd_res_temp)).reset_index(drop=True)
                    # vmd_dominant_results = pd.concat((vmd_dominant_results,vmd_dom_temp)).reset_index(drop=True)
                    vmd_binned_results = pd.concat((vmd_binned_results,vmd_binned_temp)).reset_index(drop=True)  
    return vmd_results,vmd_binned_results # ,vmd_dominant_results

def map_with_true_bridge_freq(df,case_studies,veh_classes,bridge_names):
    """This function is used to match the analytically calculated bridge frequencies using the Euler beam formulations.

    Args:
        df (df): dataframe containing the picked frequencies for each case study.
        case_studies ([str]): list of strings describing the case studies analyzed.
        veh_classes ([str]): list of vehicle classes used.
        bridge_names ([str]): list of strings describing each bridge analyzed.

    Returns:
        df: returns dataframe with the frequencies that match the most.
    """
    final_df = pd.DataFrame()
    for case_study in case_studies:
        for veh_class in veh_classes:
            # extract results from vmd analysis
            for bridge_name in bridge_names:
                if df[(df['bridge_name'] == bridge_name) & (df['veh_class'] == veh_class) & (df['case_study'] == case_study)].empty:
                    continue
                temp_df = df[(df['bridge_name'] == bridge_name) & (df['veh_class'] == veh_class) & (df['case_study'] == case_study)].copy()
                temp_df['diff'] = (temp_df['peak_f'] - temp_df['bridge_freq_true']).abs()
                temp_df.sort_values(by=['diff','peak_f_count','mean_ratio'],ascending=[True,False,False],inplace=True)
                final_df = pd.concat((final_df,pd.DataFrame(temp_df.loc[temp_df.index[0]]).T))
    return final_df.reset_index(drop=True)

def get_flexural_idx_and_frequency(bridge_properties,freq_res):
    """This function is used to compute the flexural index and frequency of simply supported beams based on geometric and mechanical properties.

    Args:
        bridge_properties ([float]): list of bridge properties where [list of bridge spans, list of bridge MOI, list of bridge masses, elastic modulus]
        freq_res (float): frequency resolution used to analyze the signals in Hz.

    Returns:
        [[float],[float]]: returns a list of two lists where the first list is frequency values and the second is flexural indexes.
    """
    L = bridge_properties[0]
    I = bridge_properties[1]
    M = bridge_properties[2]
    E = bridge_properties[3]
    return [round(round(np.pi/2*np.sqrt((E[idx]*10**9*I[idx])/(M[idx]*L[idx]**4))/freq_res)*freq_res,3) for idx in range(len(L))],[E[idx]*I[idx]/L[idx] for idx in range(len(L))]

def compute_drive_freq(vel,dist,freq_res):
    return round(round(vel/dist/freq_res)*freq_res,3)

def get_dominant_freq(df_exact,veh_classes,bridge_names,case_studies):
    """This function is used to extract the frequencies with the highest counts and psd ratio.

    Args:
        df_binned (df): dataframe containing the peak frequencies and psd ratios in binned structure.
        df_exact (df): dataframe containing the exact peak frequencies.
        veh_classes ([str]): list of strings describing the vehicle class used.
        bridge_names ([str]): list of strings describing bridge names during analysis.
        case_studies ([str]): list of strings describing the bridge case study analyzed.

    Returns:
        df: dataframe containing the most dominant frequency for each case study and vehicle class.
    """
    df_exact_final = pd.DataFrame()
    # get the bin frequencies that were repeated the most and had the highest amplitude
    for case_study in case_studies:
        for veh_class in veh_classes:
            for bridge_name in bridge_names:
                if df_exact[(df_exact['bridge_name'] == bridge_name) & (df_exact['veh_class'] == veh_class) & (df_exact['case_study'] == case_study)].empty:
                    continue
                df_exact_final = pd.concat((df_exact_final,df_exact[(df_exact['case_study']==case_study) & (df_exact['bridge_name']==bridge_name) & (df_exact['veh_class']==veh_class)].sort_values(['peak_f_count','mean_peak_psd'],ascending=False,ignore_index=True).loc[0].to_frame()),axis=1)
    df_exact_final = df_exact_final.T.reset_index(drop=True)
    return df_exact_final

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

def get_number_detections(df,veh_classes,detection_str,detection_prop,range_to_accept):
    """This function is used to get the number of detection from APPVMD results.

    Args:
        df (df): dataframe containing the aggregated results from the APPVMD algorithm.
        veh_classes ([str]): list of strings describing vehicle names.
        detection_str (str): string containing whether the function should consider index the series with 'detected' or 'not detected'.
        detection_prop ([float]): float or list depending on the range_to_accept. If range_to_accept = 'greater' or 'less' than only define this variable as a float, else if 'between' then define it as a list with the first element being the lower bound and the second being upper.
        range_to_accept (str): string describing the range to search for (e.g., greater, betwee, less)

    Returns:
        list: list of number of times the function found the detected or not detected.
    """
    if detection_str == 'not detected':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str)].shape[0] for veh_class in veh_classes]

    elif range_to_accept == 'between':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & ((dom_freq_df['peak_f_prob'] >= detection_prop[0]) & (dom_freq_df['peak_f_prob'] < detection_prop[1]))].shape[0] for veh_class in veh_classes]

    elif range_to_accept == 'greater':
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (dom_freq_df['peak_f_prob'] >= detection_prop)].shape[0] for veh_class in veh_classes]

    else:
        return [df[(df['veh_class'] == f'{veh_class}') & (df['bridge_freq_detection'] == detection_str) & (dom_freq_df['peak_f_prob'] < detection_prop)].shape[0] for veh_class in veh_classes]

def get_detection_summary(veh_classes,not_detection,low_detection,int_detection,high_detection):
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
    df_summary = pd.DataFrame({'veh_class':veh_classes,'not_detected':not_detection,'low_count':low_detection,'int_count':int_detection,'high_count':high_detection})
    df_summary['total'] = df_summary[['low_count','int_count','high_count']].sum(axis=1)/df_summary.loc[:,df_summary.columns != 'veh_class'].sum(axis=1)
    return df_summary

def get_statistics(seaborn_plot_format,vehicle_classes,df,x_str,y_str,col_lab,axes_units):
    with sns.plotting_context(seaborn_plot_format):
    # initialize empty lists
        # lists for figures and axes objects:
        fig_act_vs_pred, axs_act_vs_pred = [],[]

        # lists for r2 and mse values:
        r2,mse = [],[]
        # loop through each vehicle class
        for veh_class in vehicle_classes:
            fig_temp,ax_temp = plot_actual_vs_predicted(df,x_str,y_str,col_lab,veh_class,axes_units)
            ax_temp.set_title(veh_class)
            fig_act_vs_pred.append(fig_temp); axs_act_vs_pred.append(ax_temp)
            # compute statistics quantities (e.g., r2 and mse)
            r2.append(r2_score(df[(df['veh_class'] == veh_class) & (df[y_str] > 0)][x_str],df[(df['veh_class'] == veh_class)& (df[y_str] > 0)][y_str]))
            mse.append(mean_squared_error(df[df['veh_class'] == veh_class][x_str],df[df['veh_class'] == veh_class][y_str]))
        stat_df = pd.DataFrame({'veh_class':vehicle_classes,'r2':r2,'mse':mse})
    return fig_act_vs_pred,axs_act_vs_pred,stat_df

def plot_hist_prob(plot_format,df,col_labs,bin_width):
    with sns.plotting_context(plot_format):
        fig_hist,ax_hist = plt.subplots(1,3,figsize=(24,10))
        for idx,col_lab in enumerate(col_labs):
            min_val = math.floor(df[col_lab].min()/bin_width[idx])*bin_width[idx]
            max_val = math.ceil(df[col_lab].max()/bin_width[idx])*bin_width[idx]
            hist_sns = sns.histplot(data=df,x=col_lab,bins=range(min_val,max_val+bin_width[idx],bin_width[idx]),ax=ax_hist[idx])
            hist_sns.set_xticks(range(min_val,max_val+bin_width[idx],bin_width[idx]))
            hist_sns.set_yticks(range(int(min(hist_sns.get_yticks())), int(max(hist_sns.get_yticks())) + 1))
    return fig_hist,ax_hist

if __name__ == '__main__':
    # seaborn plot dictionary:
    plot_style = {
        'font.size': 14,
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
    # file inputs:
    # file_path = '../../parametric_studies/parametric_study_velocity/case_studies/'
    file_path = '../../parametric_studies/velocity/case_studies/'
    case_studies = ['Case_1','Case_3','Case_2']
    bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num',
                    'med_num','high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
    cases_to_ignore = [['Case_1','Case_2','Case_3'],['small_thick','med_num'],['low_num','low_num_constant_flexure'],['small_thick','med_num']]
    veh_classes = ['sedan']
    # veh_classes = ['sedan']
    veh_masses = [1707]
    # veh_masses = [1707]
    freq_res = 0.1
    # bridge parameters:
    L = [
        16,30,40,16,16,16,16,16,16,20,38,56,20,24,28,
        20,30,40,40,40,40,40,40,40,40,44,46,15,27,42,
        21.3,30,40,21.3,21.3,21.3,21.3,21.3,21.3,20,38,56,13,28,44]
    I = [
        0.0621,0.0621,0.0621,0.0621,0.0744,0.0863,0.0324,0.0621,0.0915,0.0324,0.0621,0.0915,0.0621,0.0744,0.0863,
        0.451,0.451,0.451,0.253,0.451,0.714,0.451,0.753,0.997,0.451,0.753,0.997,0.253,0.451,0.714,
        0.0837,0.0837,0.0837,0.0837,0.177,0.284,0.0837,0.159,0.235,0.0837,0.159,0.235,0.0837,0.177,0.284]
    M = [
        11600,11600,11600,11600,11700,11750,9040,11600,15500,9040,11600,15500,11600,11700,11750,
        10400,10400,10400,10302,10400,10600,10400,12265,14080,10400,12265,14080,10302,10400,10600,
        5600,5600,5600,5600,5790,5840,5600,6700,9800,5600,6700,9800,5600,5790,5840]
    E = [
        211,211,211,211,211,211,211,211,211,211,211,211,211,211,211,
        205,205,205,205,205,205,205,205,205,205,205,205,205,205,205,
        200,200,200,200,200,200,200,200,200,200,200,200,200,200,200]
    bridge_properties = [L,I,M,E]
    V = [9,13.4,17.9,22.4]
    bin_width = 0.5
    modes_included = 4
    analysis_type = 1 # 1 for mapping with true bridge frequency and 0 for using maximum number and psd ratio to make bridge frequency extraction
    
    # Analysis:
    # create new vehicle class and mass names to include velocity:
    veh_classes_ = [f'{veh_class}_{v_i}' for veh_class in veh_classes for v_i in V]
    veh_masses_ = [veh_mass for veh_mass in veh_masses for v_i in V]

    # create dataframe for vehicles:
    # veh_df = pd.DataFrame({'veh_class':['sedan_9','sedan_13.4','sedan_17.9','sedan_22.4'],'veh_mass':veh_masses*4})
    veh_df = pd.DataFrame({'veh_class':veh_classes_,'veh_mass':veh_masses_})

    # get frequency and flexural index values for each bridge:
    frequency,flexural_index = get_flexural_idx_and_frequency(bridge_properties,freq_res)

    # convert frequency vector to df:
    bridge_freq_df = pd.DataFrame({'peak_f':frequency})

    # extract vmd results from parametric study folder:
    vmd_results,vmd_binned_results = read_vmd_results(file_path,case_studies,bridge_names,veh_classes,V,modes_included,cases_to_ignore)
    # vmd_results,vmd_binned_results,vmd_dominant_results = read_vmd_results(file_path,case_studies,bridge_names,veh_classes,V,modes_included,cases_to_ignore)
    vmd_results = pd.merge(vmd_results,veh_df,how='left',on='veh_class')


    # create bridge dataframe
    bridge_df = pd.DataFrame({'mass':M,'span':L,'E':E,'I':I,'flexural_index':flexural_index,'bridge_freq_true':frequency,'bridge_name':bridge_names*len(case_studies),'case_study':flatten([[case_study]*len(bridge_names) for case_study in case_studies])})
    
    # label spans as short, medium and long spans
    bridge_df.loc[(bridge_df['span'] <= 20),'span_type'] = 'short_span'
    bridge_df.loc[(bridge_df['span'] > 20) & (bridge_df['span'] <= 30),'span_type'] = 'med_span'
    bridge_df.loc[(bridge_df['span'] > 30),'span_type'] = 'long_span'

    # merge bridge_df with vmd_results to get the frequency and flexural index lined up with the different case studies and vehicle classes:
    merge_vmd_df = pd.merge(bridge_df,vmd_results,how='right',on=['bridge_name','case_study'])
    merge_vmd_df['mass_ratio'] = merge_vmd_df['veh_mass']/(merge_vmd_df['mass']*merge_vmd_df['span'])
    merge_vmd_df['EI'] = merge_vmd_df['E']*merge_vmd_df['I']
    # merge vmd binned results with bridge_df:
    merge_vmd_binned_df = pd.merge(bridge_df,vmd_binned_results,how='right',on=['bridge_name','case_study'])
    # merge dominant vmd results:
    # merge_vmd_dom_df = pd.merge(bridge_df,vmd_dominant_results,how='right',on=['bridge_name','case_study'])
    # map with bridge frequencies based on the Euler beam frequency equation:
    dom_freq_df = map_with_true_bridge_freq(merge_vmd_df,case_studies,veh_classes_,bridge_names)
    # get bridge frequencies based on high count of bins:
    # dom_freq_df_count = get_dominant_freq(merge_vmd_dom_df,['sedan_9','sedan_13.4','sedan_17.9','sedan_22.4'],bridge_names,case_studies)
    # create column specifying whether bridge is detected or not:
    if analysis_type == 1:
        dom_freq_df = get_bridge_detection(dom_freq_df,1)
    # else:
        # dom_freq_df = get_bridge_detection(dom_freq_df_count,1)
    
    # Add confidence regions
    dom_freq_df.loc[(dom_freq_df['peak_f_prob'] < 0.4),'CFR'] = 'low CFR'
    dom_freq_df.loc[(dom_freq_df['peak_f_prob'] >= 0.4) & (dom_freq_df['peak_f_prob'] < 0.7),'CFR'] = 'inter. CFR'
    dom_freq_df.loc[(dom_freq_df['peak_f_prob'] >= 0.7),'CFR'] = 'high CFR'

    # count the number of probabilities within certain ranges:
    # not detected:
    n_not_detected_list = get_number_detections(dom_freq_df,veh_classes_,'not detected',0.0,'')

    # greater than 70 % 
    n_high_detected_list = get_number_detections(dom_freq_df,veh_classes_,'detected',0.7,'greater')
    
    # 40% - 70%: 
    n_int_detected_list = get_number_detections(dom_freq_df,veh_classes_,'detected',[0.4,0.7],'between')

    # less than 40%
    n_low_detected_list = get_number_detections(dom_freq_df,veh_classes_,'detected',0.4,'less')
    
    # summarize results into dataframe:
    df_counts = get_detection_summary(veh_classes_,n_not_detected_list,n_low_detected_list,n_int_detected_list,n_high_detected_list)
    print(df_counts)
    
    # get statistics summary of dataframes:
    fig_act_vs_pred, axs_act_vs_pred, stat_df = get_statistics(plot_style,veh_classes_,dom_freq_df,'bridge_freq_true','peak_f','veh_class','Hz')
    print(stat_df)

    pdb.set_trace()
    # plot relational plots of dataframe:
    g = sns.relplot(data=dom_freq_df,x='flexural_index',y='CFR',hue='mass',style='span_type',row='bridge_freq_detection',col='veh_class',palette='flare',facet_kws={"margin_titles": True,'despine':False},s=100)
    g.set_xlabels('Flexural Index (GN/m)',clear_inner=False)
    g.set_ylabels('Confidence Region',clear_inner=False)
    g.set_titles(row_template='{row_name}',col_template='{col_name}')

    # histogram plots
    # Store confidence region df for trucks:
    high_cr_tr = dom_freq_df[(dom_freq_df['veh_class']=='truck_10') & (dom_freq_df['bridge_freq_detection']=='detected') & (dom_freq_df['CFR'] == 'high CFR')].reset_index(drop=True)
    med_cr_tr = dom_freq_df[(dom_freq_df['veh_class']=='truck_10') & (dom_freq_df['bridge_freq_detection']=='detected') & (dom_freq_df['CFR'] == 'inter. CFR')].reset_index(drop=True)
    low_cr_tr = dom_freq_df[(dom_freq_df['veh_class']=='truck_10') & (dom_freq_df['bridge_freq_detection']=='detected') & (dom_freq_df['CFR'] == 'low CFR')].reset_index(drop=True)
    not_detected_tr = dom_freq_df[(dom_freq_df['veh_class']=='truck_10') & (dom_freq_df['bridge_freq_detection']=='not detected')].reset_index(drop=True)
    
    # high cfr:
    fig_hist_high,ax_hist_high = plot_hist_prob(plot_style,high_cr_tr,['EI','span','mass'],[20,10,4000])
    fig_hist_high.suptitle('High Confidence Region')
    ax_hist_high[0].set_title('Histogram of Bridge EI');ax_hist_high[1].set_title('Histogram of Bridge Span Lengths');ax_hist_high[2].set_title('Histogram of Bridge Masses')
    ax_hist_high[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_high[1].set_xlabel('Span Length (m)');ax_hist_high[2].set_xlabel('Bridge Mass (kg/m)')

    # intermediate cfr:
    fig_hist_int,ax_hist_int = plot_hist_prob(plot_style,med_cr_tr,['EI','span','mass'],[20,10,4000])
    fig_hist_int.suptitle('Intermediate Confidence Region')
    ax_hist_int[0].set_title('Histogram of Bridge EI');ax_hist_int[1].set_title('Histogram of Bridge Span Lengths');ax_hist_int[2].set_title('Histogram of Bridge Masses')
    ax_hist_int[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_int[1].set_xlabel('Span Length (m)');ax_hist_int[2].set_xlabel('Bridge Mass (kg/m)')

    # low cfr:
    fig_hist_low,ax_hist_low = plot_hist_prob(plot_style,low_cr_tr,['EI','span','mass'],[20,10,4000])
    fig_hist_low.suptitle('Low Confidence Region')
    ax_hist_low[0].set_title('Histogram of Bridge EI');ax_hist_low[1].set_title('Histogram of Bridge Span Lengths');ax_hist_low[2].set_title('Histogram of Bridge Masses')
    ax_hist_low[0].set_xlabel('EI ($\mathregular{GN-m^2}$)');ax_hist_low[1].set_xlabel('Span Length (m)');ax_hist_low[2].set_xlabel('Bridge Mass (kg/m)')

    
    