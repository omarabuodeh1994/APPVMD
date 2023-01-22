from tkinter import Y
from input_file_commands import*
from python_functions import*
from vehicle_library import*
from computation_functions import*
import numpy as np
import os
import os.path
import time
from post_process_FDD import *
import matplotlib.pyplot as plt

# This script is developed to model a simple sprung mass example with mass M and spring stiffness K
# An input file will be generated and submitted to the Abaqus solver

# Given:
# Bridge:
E = 200.0e9 # 27.5e09
PRatio = 0.2
BeamMass = 5600.0 # 4800.0 
Height = 1.9282 # (12.0*0.12/2.0)**0.5
Width = 0.1409 # 2.0/Height 
Lapproach = 7.0 # length of approach slab in m
BridgeLength = 21.3 # length of suspended bridge in m
Lexit = 7.0 # length of exit slab in m 
NumberOfElements = 100
V = 10.0 # velocity of vehicle in m/s
bridge_frequency = 2.08 # 6.0 # frequency corresponding to first model in bridge (Hz)

# Signal inputs:
time_step = 0.001 # time step in analysis in sec
frequency_bin_width = 0.5 # frequency bin width in Hz
frequency_cut_off = 50 # cut-off frequency for low-pass filter in Hz
# Finite Element inputs
DamageLocation = [] # crack location across beam in m 
CrackRatio = [] # crack ratio

surface_roughness_bool = 1 # boolean to include surface (0 = No and 1 = Yes)
road_class = 'A' # ISO road class (A,B,C)

BeamGeometry = [Width,Height,BridgeLength,Lapproach,Lexit] 
FEInput = [NumberOfElements,DamageLocation,CrackRatio]

# Vehicle:
DimensionMass = [0.1,0.1,0.1] # Dimensions of mass block
axle_label = ['Rear_Axle','Front_Axle']
[a,b] = [3.0,3.0] # length of CM [rear,front]
vehicle_frequencies = [1.3,2.2,9.7,15.4] # vehicle frequencies [body bounce, body pitch, front bounce, rear bounce]
# Position of center of mass node: 
XPosition = [-Lapproach, -Lapproach] 
YPosition = [0.0, 1.05]
# Name of parts:
PartNames = ['bridge_part','VehiclePart']
MatName = ['BridgeMat','VehicleMat']
# File information:
file_name = 'passenger_robs_bridge_surf_'
######################################################################################
#-----------------------------------HEADER--------------------------------------------
frequency_peaks_off_bridge = [] # predefine empty list to store frequency peaks when vehicle is off bridge
frequency_peaks_on_bridge = [] # predefine empty list to store frequency peaks when vehicle is on bridge
iterations = 1
for i in range(1,iterations+1):
    file_ID = i
    v = [0,0,0,0]
    HeaderLines = input_header('JobTest',PartNames[0],v)
    FinalLines = HeaderLines
    ######################################################################################
    #-----------------------------------PARTS---------------------------------------------
    # ----- Vehicle part:
    # FinalLines += ['** PARTS VEHICLE'] + SprungMassPart(XPosition,YPosition,DimensionMass,PartNames[1],K,C,MatName[1],DimensionMass[2])
    LinesVehicle = passenger_vehicle_part(PartNames[1],-Lapproach) # passenger vehicle model
    FinalLines += ['** PARTS VEHICLE'] + LinesVehicle['Lines']
    # Vehicle properties:
    length_rear,length_center,length_front = LinesVehicle['beam_lengths'][0], LinesVehicle['beam_lengths'][1], LinesVehicle['beam_lengths'][2]
    radius_rear, radius_center, radius_front = LinesVehicle['radius'][0], LinesVehicle['radius'][1], LinesVehicle['radius'][2]
    mass_rear,mass_center,mass_front = LinesVehicle['beam_masses'][0],LinesVehicle['beam_masses'][1],LinesVehicle['beam_masses'][2]
    mass_rear_axle,mass_front_axle = LinesVehicle['axle_masses'][0],LinesVehicle['axle_masses'][1]
    k_unsprung = LinesVehicle['unsprung_stiffness']
    c_sprung = LinesVehicle['sprung_damping']
    Lv = LinesVehicle['vehicle_length']
    mass_vehicle = sum([mass_rear,mass_center,mass_front])
    axle_position = LinesVehicle['axle_position']
    ######################################################################################
    #-----------------------------------CALCULATION---------------------------------------
    total_bridge_length = Lapproach + BridgeLength + Lexit
    time_period = (total_bridge_length-Lv)/V
    time_vehicle_driving = np.round(crange(0,time_period,0.001),3)
    vehicle_position_path = pd.DataFrame(np.vstack((crange(axle_position[0],total_bridge_length-Lv,V*time_step),
                                                    crange(axle_position[1],total_bridge_length,V*time_step))).T,
                                                    columns=axle_label)

    # ----- Bridge part:
    bridge_part_lines = bridge_part(PartNames[0],E,PRatio,BeamGeometry,FEInput, MatName[0], 'No')
    FinalLines += ['** PARTS BRIDGE'] + bridge_part_lines['Lines']
    bridge_nodes_of_interest = bridge_part_lines['node_ID_of_interest'] 
    ######################################################################################
    #-----------------------------------ASSEMBLY------------------------------------------
    # ----- Assemble parts:
    FinalLines += assemble_parts(PartNames)

    # ----- Assign rigid body definition to sprung mass part:
    FinalLines += rigid_body (PartNames[1],'EntireBeam',['Tie','BeamTie'],'BeamCM') # vehicle body
    FinalLines += rigid_body (PartNames[1],'RearAxle',['',''],'RearAxleCM') # rigid body rear axle
    FinalLines += rigid_body (PartNames[1],'FrontAxle',['',''],'FrontAxleCM') # rigid body front axle

    # ----- Define Node sets for results of interest:
    FinalLines += nset('Results',PartNames[1]+'.VehicleCMs,'+PartNames[0]+'.NodesOfInterest',0)

    # ----- End Assembly:
    FinalLines += ['*End assembly']

    ######################################################################################
    #-----------------------------------MATERIAL DEFINITION-------------------------------
    # ----- Sprung mass definition:
    density_rear = compute_density('Circle',[radius_rear,length_rear],mass_rear)
    density_center = compute_density('Circle',[radius_center,length_center],mass_center)
    density_front = compute_density('Circle',[radius_front,length_front],mass_front)
    FinalLines += mat_definition(PartNames[1]+'RearBeam',1.0e10,0.2,density_rear,[])
    FinalLines += mat_definition(PartNames[1]+'CenterBeam',1.0e10,0.2,density_center,[])
    FinalLines += mat_definition(PartNames[1]+'FrontBeam',1.0e10,0.2,density_front,[])
    FinalLines += mat_definition(PartNames[1]+'RearAxle',1.0e10,0.2,mass_rear_axle/(0.1**3),[])
    FinalLines += mat_definition(PartNames[1]+'FrontAxle',1.0e10,0.2,mass_front_axle/(0.1**3),[])

    # ----- Bridge definition:
    FinalLines += mat_definition(MatName[0],E,PRatio,BeamMass/(Width*Height),[])

    ######################################################################################
    #-----------------------------------SURFACE DEFINITION--------------------------------
    FinalLines += surface_interaction('Surface',PartNames[1]+'.tire_surface',PartNames[0]+'.BeamSurf')

    ######################################################################################
    #-----------------------------------SUPPORT BOUNDARY CONDITIONS-----------------------
    # ----- Apply BC on bridge:
    # Assign Pin:
    FinalLines += boundary_condition(PartNames[0],'VerticalSupport','Support',[],[1,2],'MOD')

    # Assign Roller:
    FinalLines += boundary_condition(PartNames[0],'AllNodes','Support',[],[1],'MOD')

    FinalLines += boundary_condition(PartNames[1],'VehicleCMs','Support',[],[1],'MOD')
    FinalLines += boundary_condition(PartNames[1],'VehicleCMsForRotation','Support',[],[6],'MOD')

    ######################################################################################
    #-----------------------------------step DEFINITION-----------------------------------
    # ----- Define implicit dynamic step:
    FinalLines += step('Dynamic',time_period,time_step,-0.41421,'Dynamicstep',0)
    # ---- Lumped Mass:
    # Rear:
    lumped_rear = a/Lv*mass_vehicle+mass_rear_axle
    # Front
    lumped_front = b/Lv*mass_vehicle+mass_front_axle
    # ----- Surface definition:
    phase_angles = pd.read_csv('phase_angles2.csv',sep=',',header=None)
    print('Started generating surface profile \n')
    surface_roughness_df = surface_roughness_given_phase_angles(surface_roughness_bool,'A', total_bridge_length, time_step,phase_angles.values[0])
    print('Finished generating surface profile \n')
    amp_rear_surface_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,axle_label[0],[k_unsprung[0],0.0,lumped_rear])
    print('Finished generating rear amplitude dataframe \n')
    amp_front_surface_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,axle_label[1],[k_unsprung[1],0.0,lumped_front])
    print('Finished generating front amplitude dataframe \n')
    # ----- Amplitude definition:
    # Save amplitude csv files for rear side:
    amp_rear_surface_df['amplitude_mass'].to_csv('amplitude_rear_axle.inp',columns=['Time_sec_','Rear_Axle'],header=False,index=False) # rear axle
    amp_rear_surface_df['amplitude_contact'].to_csv('amplitude_rear_contact.inp',columns=['Time_sec_','Rear_Axle'],header=False,index=False) # rear contact
    # Save amplitude csv files for front side:
    amp_front_surface_df['amplitude_mass'].to_csv('amplitude_front_axle.inp',columns=['Time_sec_','Front_Axle'],header=False,index=False) # front axle
    amp_front_surface_df['amplitude_contact'].to_csv('amplitude_front_contact.inp',columns=['Time_sec_','Front_Axle'],header=False,index=False) # front contact
    # Include amplitude:
    # Rear:
    FinalLines += amplitude('RearAxleAmp','amplitude_rear_axle.inp') # rear axle
    FinalLines += amplitude('RearContactAmp','amplitude_rear_contact.inp') # rear contact
    # Front:
    FinalLines += amplitude('FrontAxleAmp','amplitude_front_axle.inp') # front axle
    FinalLines += amplitude('FrontContactAmp','amplitude_front_contact.inp') # front contact
    # ----- Define concentrated force to simulate surface roughness:
    # Rear:
    FinalLines += concentrated_load('RearAxleCF','RearAxleAmp') # Defining concentrated force on rear axle
    FinalLines += [PartNames[1]+'.RearAxleCM,2,1'] 
    FinalLines += concentrated_load('RearContactCF','RearContactAmp') # Defining concentrated force on rear contact
    FinalLines += [PartNames[1]+'.RearSurface,2,-1']
    # Front:
    FinalLines += concentrated_load('FrontAxleCF','FrontAxleAmp') # Defining concentrated force on rear axle
    FinalLines += [PartNames[1]+'.FrontAxleCM,2,1']
    FinalLines += concentrated_load('FrontContactCF','FrontContactAmp') # Defining concentrated force on rear contact
    FinalLines += [PartNames[1]+'.FrontSurface,2,-1']
    # ----- Define horizontal displacement of vehicle:
    # Passenger Vehicle:
    FinalLines += boundary_condition(PartNames[1],'VehicleCMs','Applied',[Lapproach+BridgeLength+Lexit-Lv],[1],'MOD') # Beam CM 
    # ----- Output requests
    # Field output:
    FinalLines += output_request(1,'Field', 1,'Results','U, V, A,CF')
    # History output:
    FinalLines += output_request(0,'History', 1,'Results','U, V, A,CF')
    # End of step
    FinalLines += ['*End step']

    # ----- Save input file in separate text file:
    np.savetxt(file_name+str(file_ID)+'.inp',FinalLines,fmt="%s")

    ######################################################################################
    #-----------------------------------RUN ABAQUS----------------------------------------
    os.system('abaqus job='+file_name+str(file_ID))
    time.sleep(5)
    start_time = time.time()
    while os.path.exists(file_name+str(file_ID)+'.lck'):
        pass
    end_time = time.time()
    duration = end_time - start_time
    print(duration)
    bridge_properties = [Lapproach,BridgeLength,Lexit]
    # #-----------------------------------GENERATE ABAQUS REPORT----------------------------
#     os.system('abaqus ODBreport job='+file_name+str(file_ID)+' history')
#     time.sleep(3)

#     # Employ FDD
#     frequency_results = passenger_FDD(bridge_properties,Lv,frequency_cut_off,frequency_bin_width,V,file_name+str(file_ID))
#     frequency_peaks_on_bridge.append(frequency_results[0])
#     frequency_peaks_off_bridge.append(frequency_results[1])
#     # Save FDD plots:
#     fig_on_bridge = frequency_results[2]
#     fig_on_bridge.savefig('FDD_on_bridge_'+str(file_ID)+'.png')
#     fig_off_bridge = frequency_results[3]
#     fig_off_bridge.savefig('FDD_off_bridge_'+str(file_ID)+'.png')
#     # Remove files generated by abaqus:
#     remove_files(file_name,file_ID) 
#     # plt.show()
#     # print(frequency_peaks)

# # Flatten list of lists
# frequency_peaks_flattened_on_bridge = flatten(frequency_peaks_on_bridge)
# frequency_peaks_flattened_off_bridge = flatten(frequency_peaks_off_bridge)
# # Calculate maximum frequencies
# max_frequencies = max(flatten([frequency_peaks_flattened_on_bridge,frequency_peaks_flattened_off_bridge,[bridge_frequency],vehicle_frequencies[2:]]))
# # Calculate minimum frequencies
# min_frequencies = min(flatten([frequency_peaks_flattened_on_bridge,frequency_peaks_flattened_off_bridge,[bridge_frequency],vehicle_frequencies[2:]]))
# # Calculate bins
# bins_ = np.arange(min_frequencies,max_frequencies+0.5,0.5)
# # plot histogram of vehicle runs in subplots
# fig,axs = plt.subplots(2,sharex=True)
# # Add shared title:
# fig.suptitle(r'$g_d = 4.0e^{-6}m^3$')
# # On bridge:
# # axs[0].hist(frequency_peaks_flattened_on_bridge,bins=math.ceil(max(frequency_peaks_flattened_on_bridge)-min(frequency_peaks_flattened_on_bridge)/frequency_bin_width))
# axs[0].hist(frequency_peaks_flattened_on_bridge,bins=bins_)
# # axs[0].vlines(vehicle_frequencies[0],0,max(vehicle_frequencies),'k','--',label='vehicle bounce frequency')
# # axs[0].vlines(vehicle_frequencies[1],0,max(vehicle_frequencies),'r',':',label='vehicle pitch frequency')
# axs[0].vlines(vehicle_frequencies[2],0,10,'k','-',label='rear axke bounce frequency')
# axs[0].vlines(vehicle_frequencies[3],0,10,'r','-',label='front axle bounce frequency')
# axs[0].vlines(bridge_frequency,0,10,'r','--',label='bridge frequency')
# axs[0].legend()
# axs[0].set_title('On Bridge')
# axs[0].set_xticks(np.arange(min_frequencies,max_frequencies+0.5,0.5))
# axs[0].set_ylabel('Count')
# # Off bridge
# # axs[1].hist(frequency_peaks_flattened_off_bridge,bins=math.ceil(max(frequency_peaks_flattened_off_bridge)-min(frequency_peaks_flattened_off_bridge)/frequency_bin_width))
# axs[1].hist(frequency_peaks_flattened_off_bridge,bins=bins_)
# # axs[1].vlines(vehicle_frequencies[0],0,max(vehicle_frequencies),'k','--',label='vehicle bounce frequency')
# # axs[1].vlines(vehicle_frequencies[1],0,max(vehicle_frequencies),'r',':',label='vehicle pitch frequency')
# axs[1].vlines(vehicle_frequencies[2],0,10,'k','-',label='rear axke bounce frequency')
# axs[1].vlines(vehicle_frequencies[3],0,10,'r','-',label='front axle bounce frequency')
# axs[1].vlines(bridge_frequency,0,10,'r','--',label='bridge frequency')
# axs[1].legend()
# axs[1].set_title('Off Bridge')
# axs[1].set_ylabel('Count')
# axs[1].set_xticks(np.arange(min_frequencies,max_frequencies+0.5,0.5))
# axs[1].set_xlabel('Frequency (Hz)'); 
# # fig.savefig('passenger_FDD_histogram')
# print(frequency_peaks_flattened_off_bridge)
# print(frequency_peaks_flattened_on_bridge)
# plt.show()