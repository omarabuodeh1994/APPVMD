from single_truck_sim import *
import pdb 
# inputs for truck simulation:
bridge_properties = {
    'E':211.0e9,
    'poisson': 0.2,
    'beam_mass': 11600.0,
    'beam_height': 1.3140,
    'beam_width': 0.3285, 
    'approach_slab_length': 40.0,
    'bridge_length': 16.0,  
    'exit_slab_length': 8.0,
    'number_elements': 100
}
vehicle_properties = {
    'velocity':10.0,
    'dt':0.001,
    'vehicle_length': 7.8,
}
time_lag = 10 # time lag to account for previous steps (adding/removing dashpot elements to allow the vehicle to reach)
num_passes = 1
for itr in range(1,num_passes+1):
    # ---------------------
    # run truck simulation:
    # ---------------------
    truck_sim_res = truck_sim(bridge_properties,vehicle_properties,f'truck_run_{itr}')
    brge_nodes = truck_sim_res['bridge_nodes_inter']
    veh_nodes = truck_sim_res['veh_cm_nodes']
    # ----------------
    # extract results:
    # ----------------
    # from inputs:
    brge_len = bridge_properties['bridge_length'] # bridge length in m
    aprch_slb_len = bridge_properties['approach_slab_length'] # approach slab length in m
    exit_slb_len = bridge_properties['exit_slab_length'] # exit slab length in m
    Lv = vehicle_properties['vehicle_length'] # length of vehicle in m
    v = vehicle_properties['velocity'] # vehicle driving speed in m/s
    # calculations:
    time_period = (aprch_slb_len+brge_len+exit_slb_len-Lv)/v
    # ----------------------------------
    # extract vehicle acceleration data:
    # ----------------------------------
    veh_res_on,veh_res_off = pd.DataFrame(),pd.DataFrame()
    for count, i in enumerate(veh_nodes, start=1):
        veh_res_dummy = abqs_results('truck_run_1.rep',i,'TRUCKPART','DynamicStep',time_period)
        # ---------------
        # on-bridge data:
        # ---------------
        start_time = (aprch_slb_len-Lv)/v
        end_time = (aprch_slb_len+brge_len)/v
        on_brg_data = time_history(veh_res_dummy.copy(),start_time,end_time,0) # sprung mass
        on_brg_data.reset_index(drop=True,inplace=True)
        veh_res_on[f'truck_{count}'] = on_brg_data['Response'].values
        # ----------------
        # off-bridge data:
        # ----------------
        start_time = 0
        end_time = (aprch_slb_len-Lv)/v
        off_brg_data= time_history(veh_res_dummy.copy(),start_time,end_time,0) # sprung mass
        off_brg_data.reset_index(drop=True,inplace=True)
        veh_res_off[f'truck_{count}'] = off_brg_data['Response'].values
    # add column times to both dataframes:
    veh_res_on.insert(0,'time',on_brg_data['Frame']) # on bridge
    veh_res_off.insert(0,'time',off_brg_data['Frame']) # on bridge
    # save to csv:
    veh_res_on.to_csv(f'vehicle_on_results_{itr}',index=False)
    veh_res_off.to_csv(f'vehicle_off_results_{itr}',index=False)
    # ---------------------------------
    # extract bridge acceleration data:
    # ---------------------------------
    brge_res_on,brge_res_off = pd.DataFrame(),pd.DataFrame()
    for count, i in enumerate(brge_nodes, start=1):
        brge_res_dummy = abqs_results('truck_run_1.rep',i,'BRIDGE_PART','DynamicStep',time_period)
        # ---------------
        # on-bridge data:
        # ---------------
        start_time = (aprch_slb_len-Lv)/v
        end_time = (aprch_slb_len+brge_len)/v
        on_brge_data = time_history(brge_res_dummy.copy(),start_time,end_time,0) # sprung mass
        on_brge_data.reset_index(drop=True,inplace=True)
        brge_res_on[f'bridge_{count}'] = on_brge_data['Response'].values
        # ----------------
        # off-bridge data:
        # ----------------
        start_time = 0
        end_time = (aprch_slb_len-Lv)/v
        off_brge_data= time_history(brge_res_dummy.copy(),start_time,end_time,0) # sprung mass
        off_brge_data.reset_index(drop=True,inplace=True)
        brge_res_off[f'bridge_{count}'] = off_brg_data['Response'].values
    # add column times to both dataframes:
    brge_res_on.insert(0,'time',on_brg_data['Frame']) # on bridge
    brge_res_off.insert(0,'time',off_brg_data['Frame']) # on bridge
    # save to csv:
    brge_res_on.to_csv(f'bridge_on_results_{itr}',index=False)
    brge_res_off.to_csv(f'bridge_off_results_{itr}',index=False)
    pdb.set_trace()