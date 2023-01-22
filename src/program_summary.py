import pandas as pd
import numpy as np
import seaborn as sns
from input_params import *
import pdb
from comb_vmd_vel import plot_hist_prob
import matplotlib.pyplot as plt

def edit_axes_xlabels(axs_subplots,x_label):
    # loop through each subplot
    for idx,axs_subplot in enumerate(axs_subplots):
        # edit x and y labels
        axs_subplot.set_xlabel(x_label[idx])

if __name__ == '__main__':
    # load input parameters into a list based on case study:
    input_params = [InputParams(case_study, 'mult_veh') for case_study in [f'Case_{case_id}' for case_id in range(1, 4)]] 

    # plot histograms
    bridge_df = pd.DataFrame()
    for input_param in input_params:
        bridge_df = pd.concat((bridge_df,pd.DataFrame({
                                  'bridge_spans': input_param.bridge_spans,
                                  'bridge_masses': input_param.bridge_masses,
                                  'bridge_Es': input_param.bridge_Es,
                                  'bridge_Is': input_param.bridge_Is,
                                  'case_study': input_param.case_study})))
    bridge_df.reset_index(drop=True,inplace=True)
    fig,axs = plot_hist_prob(input_param.plot_format,bridge_df,['bridge_spans','bridge_masses','bridge_Is'],[10,2000,0.1])
    edit_axes_xlabels(axs,['Bridge Span Lengths (m)','Bridge Masses (kg/m)', 'Bridge MOI (m$^4$)'])
    fig.show()
    fig.savefig('./results/model_description/program_summary.pdf')
    pdb.set_trace()

