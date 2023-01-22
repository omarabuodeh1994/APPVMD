import pandas as pd
import pdb
case_studies = ['Case_1','Case_2','Case_3']
veh_classes = ['hatchback','sedan','SUV','truck']
bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num',
                    'med_num','high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
V = [10]
files_dir = '../parametric_studies/study/mult_veh/case_studies/'

# loop through each case study:
for case_study in case_studies:
    # loop through each velocity:
    for vel_i in V:
        # loop through each vehicle class:
        for veh_class in veh_classes:
            for bridge_name in bridge_names:
                dir_csv = f'{files_dir}{case_study}/{veh_class}/{bridge_name}/stat_{bridge_name}_{vel_i}.csv'
                print(dir_csv)
                df_stat = pd.read_csv(dir_csv)
                df_stat['veh_class'] = f'{veh_class}'
                df_stat.to_csv(f'{files_dir}{case_study}/{veh_class}/{bridge_name}/stat_{bridge_name}_4_modes.csv',index=False)
                # set difference
                dir_csv = f'{files_dir}{case_study}/{veh_class}/{bridge_name}/set_diff_{bridge_name}_{vel_i}.csv'
                print(dir_csv)
                df_set_diff = pd.read_csv(dir_csv)
                df_set_diff.to_csv(f'{files_dir}{case_study}/{veh_class}/{bridge_name}/set_diff_{bridge_name}_4_modes.csv',index=False)
