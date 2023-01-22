import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import itertools
if __name__ == '__main__':
    
    # Plotting style
    plot_style = {
        'font.size': 20.0,
        'font.family':'Times New Roman',
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
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
    # Import data in df:
    sns.set_context(plot_style)
    bridge_df = pd.read_csv('./filtered_bridge_data.csv')
    fig,axs = plt.subplots(1)
    # combine all mat types into concrete and steel
    bridge_df.loc[bridge_df['MAT_NAME'] == 'Concrete Continuous','MAT_NAME'] = 'Concrete'
    bridge_df.loc[bridge_df['MAT_NAME'] == 'Steel Continuous','MAT_NAME'] = 'Steel'
    # remove zeros
    bridge_df = bridge_df.loc[(bridge_df['MAX_SPAN_LEN_MT_048'] >= 10)].reset_index(drop=True)
    # take span lengths and store into separate df:
    bridge_spans_df = pd.DataFrame(bridge_df.loc[:,['MAT_NAME','MAX_SPAN_LEN_MT_048']].to_numpy(),columns=['mat_name','span_len'])
    # plot histogram of span lengths based on material name:
    min_bin = round(bridge_spans_df['span_len'].min()/10)*10
    max_bin = round(bridge_spans_df['span_len'].max()/10)*10
    fig = sns.histplot(data=bridge_spans_df,x='span_len',hue='mat_name',bins = range(min_bin,max_bin+10,10),legend=False)
    plt.legend(title='Material Type', loc='upper right', labels=['Steel','Concrete'])
    fig.set_xlabel('Span Lengths (m)'); fig.set_ylabel('Counts')
    fig.set_xticks(range(min_bin,max_bin+10,10))
    plt.savefig('./vmd/results/model_description/bridge_hist.pdf',bbox_inches='tight')
    plt.show()
    pdb.set_trace()