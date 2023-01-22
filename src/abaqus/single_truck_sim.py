from tkinter import Y
from input_file_commands import*
from python_functions import*
from vehicle_library import*
from computation_functions import*
import numpy as np
import os
import os.path
import time
import matplotlib.pyplot as plt

def truck_sim(brdg_prop,veh_prop,file_name):
    # Given:
    # Bridge:
    E = brdg_prop['E'] 
    PRatio = brdg_prop['poisson']
    BeamMass = brdg_prop['beam_mass'] 
    Height = brdg_prop['beam_height']    
    Width = brdg_prop['beam_width']  
    Lapproach = brdg_prop['approach_slab_length']
    BridgeLength = brdg_prop['bridge_length']  
    Lexit = brdg_prop['exit_slab_length']
    NumberOfElements = brdg_prop['number_elements']
    V = veh_prop['velocity']

    # Signal inputs:
    time_step = veh_prop['dt'] # time step in analysis in sec

    # Damage inputs
    DamageLocation = []
    CrackRatio = []
    surface_roughness_bool = 1 # boolean to include surface (0 = No and 1 = Yes)
    road_class = 'A' # ISO road class (A,B,C)

    BeamGeometry = [Width,Height,BridgeLength,Lapproach,Lexit]
    FEInput = [NumberOfElements,DamageLocation,CrackRatio]

    # Vehicle:
    axle_label = ['Rear_Axle','Center_Axle','Front_Axle']
    
    # Name of parts:
    PartNames = ['bridge_part','TRUCKPART']
    MatName = ['BridgeMat','TruckPart']
    ######################################################################################
    #-----------------------------------HEADER--------------------------------------------
    v = [0,0,0,0]
    HeaderLines = input_header('JobTest',PartNames[0],v)
    FinalLines = HeaderLines

    ######################################################################################
    #-----------------------------------PARTS---------------------------------------------
    # ----- Vehicle part:
    LinesVehicle = truck_vehicle_part(PartNames[1],-Lapproach)
    FinalLines += ['** PARTS VEHICLE'] + LinesVehicle['Lines']
    # Vehicle properties:
    length_rear,length_center,length_front = LinesVehicle['beam_lengths'][0], LinesVehicle['beam_lengths'][1], LinesVehicle['beam_lengths'][2]
    radius_rear, radius_center, radius_front = LinesVehicle['radius'][0], LinesVehicle['radius'][1], LinesVehicle['radius'][2]
    mass_rear,mass_center,mass_front = LinesVehicle['beam_masses'][0],LinesVehicle['beam_masses'][1],LinesVehicle['beam_masses'][2]
    mass_rear_axle,mass_center_axle,mass_front_axle = LinesVehicle['axle_masses'][0],LinesVehicle['axle_masses'][1],LinesVehicle['axle_masses'][2]
    k_unsprung = LinesVehicle['unsprung_stiffness']
    c_sprung = LinesVehicle['sprung_damping']
    Lv = LinesVehicle['vehicle_length']
    mass_vehicle = sum([mass_rear,mass_center,mass_front])
    axle_position = LinesVehicle['axle_position']
    cm_ids = LinesVehicle['CM_IDs']
    
    ######################################################################################
    #-----------------------------------CALCULATION---------------------------------------
    total_bridge_length = Lapproach + BridgeLength + Lexit
    time_period = (total_bridge_length-Lv)/V
    time_vehicle_driving = np.round(crange(0,time_period,time_step),3)
    vehicle_position_path = pd.DataFrame(np.vstack((crange(axle_position[0],total_bridge_length-Lv,V*time_step),
                                                    crange(axle_position[1],total_bridge_length-(Lv-axle_position[1]),V*time_step),
                                                    crange(axle_position[2],total_bridge_length,V*time_step))).T,
                                                    columns=axle_label)
    
    # ----- Bridge part:
    bridge_part_lines = bridge_part(PartNames[0],E,PRatio,BeamGeometry,FEInput, MatName[0], 'No')
    FinalLines += ['** PARTS BRIDGE'] + bridge_part_lines['Lines']
    bridge_nodes_of_interest = bridge_part_lines['bridge_nodes_int']
    ######################################################################################
    #-----------------------------------ASSEMBLY------------------------------------------
    # ----- Assemble parts:
    FinalLines += assemble_parts(PartNames)

    # ----- Assign rigid body definition to sprung mass part:
    # FinalLines += rigid_body (PartNames[1],'Set',['',],'Center')
    FinalLines += rigid_body (PartNames[1],'EntireBeam',['Tie','BeamTie'],'BeamCM') # vehicle body
    FinalLines += rigid_body (PartNames[1],'RearAxle',['',''],'RearAxleCM') # rigid body rear axle
    FinalLines += rigid_body (PartNames[1],'CenterAxle',['',''],'CenterAxleCM') # rigid body center axle
    FinalLines += rigid_body (PartNames[1],'FrontAxle',['',''],'FrontAxleCM') # rigid body front axle

    # ----- Define Node sets for results of interest:
    FinalLines += nset('Results',PartNames[1]+'.VehicleCMs,'+PartNames[0]+'.NodesOfInterest',0)

    # ----- End Assembly:
    FinalLines += ['*End assembly']

    ######################################################################################
    #-----------------------------------MATERIAL DEFINITION-------------------------------
    # ----- Sprung mass definition:
    # FinalLines += mat_definition(MatName[1],20.0e20,0.2,M/(np.product(DimensionMass)),[])
    density_rear = compute_density('Circle',[radius_rear,length_rear],mass_rear)
    density_center = compute_density('Circle',[radius_center,length_center],mass_center)
    density_front = compute_density('Circle',[radius_front,length_front],mass_front)
    FinalLines += mat_definition(PartNames[1]+'RearBeam',1.0e10,0.2,density_rear,[])
    FinalLines += mat_definition(PartNames[1]+'CenterBeam',1.0e10,0.2,density_center,[])
    FinalLines += mat_definition(PartNames[1]+'FrontBeam',1.0e10,0.2,density_front,[])
    FinalLines += mat_definition(PartNames[1]+'RearAxle',1.0e10,0.2,mass_rear_axle/(0.1**3),[])
    FinalLines += mat_definition(PartNames[1]+'CenterAxle',1.0e10,0.2,mass_center_axle/(0.1**3),[])
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
    # ----- Define static step to allow vehicle to reach equilibrium position:
    FinalLines += step('Static',5.0,1.0,0.0,'Staticstep1',0)
    # ----- Remove dashpot element (if present):
    if c_sprung:
        FinalLines += model_change('REMOVE',PartNames[1]+'.RearSprungDashpot') # Remove rear sprung dashpot 
        FinalLines += model_change('REMOVE',PartNames[1]+'.FrontSprungDashpot') # Remove front sprung dashpot

    # ----- Define concentrated load due to gravity
    FinalLines += gravity_acceleration(PartNames[1],'EntireVehicle')

    # ----- Output requests
    # Field output:
    FinalLines += output_request(0,'Field', 1,'Results','U, V, A')
    # History output:
    FinalLines += output_request(0,'History', 1,'Results','U, V, A')
    # End of step
    FinalLines += ['*End step']

    if c_sprung:
        # ----- Define static step to re-add dashpot if removed:
        FinalLines += step('Static',5.0,1.0,0.0,'Staticstep2',0)
        # ----- Remove dashpot element (if present):
        if c_sprung:
            FinalLines += model_change('ADD',PartNames[1]+'.RearSprungDashpot') # Remove rear sprung dashpot 
            FinalLines += model_change('ADD',PartNames[1]+'.FrontSprungDashpot') # Remove front sprung dashpot
        
        # ----- Output requests
        # Field output:
        FinalLines += output_request(0,'Field', 1,'Results','U, V, A')
        # History output:
        FinalLines += output_request(0,'History', 1,'Results','U, V, A')
        # End of step
        FinalLines += ['*End step']

    # ----- Define implicit dynamic step:
    FinalLines += step('Dynamic',time_period,time_step,-0.41421,'DynamicStep',0)
    # ----- Surface definition:
    surface_roughness_df = surface_roughness(surface_roughness_bool,road_class,total_bridge_length)
    # Rear axle:
    amp_r_surf_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,axle_label[0],[k_unsprung[0],0.0,0.0])
    # Center axle:
    amp_c_surf_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,axle_label[1],[k_unsprung[1],0.0,0.0])
    # Front axle:
    amp_f_surf_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,axle_label[2],[k_unsprung[2],0.0,0.0])
    # ----- Amplitude definition:
    # Save amplitude csv files for rear side:
    amp_r_surf_df['amplitude_contact'].to_csv('amplitude_rear.inp',columns=['Time_sec_','Rear_Axle'],header=False,index=False)
    # Save amplitude csv files for center side:
    amp_c_surf_df['amplitude_contact'].to_csv('amplitude_center.inp',columns=['Time_sec_','Center_Axle'],header=False,index=False)
    # Save amplitude csv files for front side:
    amp_f_surf_df['amplitude_contact'].to_csv('amplitude_front.inp',columns=['Time_sec_','Front_Axle'],header=False,index=False)
    # Include amplitude in input lines:
    # Rear:
    FinalLines += amplitude('RearAmp','amplitude_rear.inp')
    # Center:
    FinalLines += amplitude('CenterAmp','amplitude_center.inp')
    # Front:
    FinalLines += amplitude('FrontAmp','amplitude_front.inp')
    # ----- Define concentrated force to simulate surface roughness:
    # Rear:
    FinalLines += concentrated_load('RearAxleCF','RearAmp') # Defining concentrated force on rear axle
    FinalLines += [PartNames[1]+'.RearAxleCM,2,1'] 
    FinalLines += concentrated_load('RearContactCF','RearAmp') # Defining concentrated force on rear contact
    FinalLines += [PartNames[1]+'.RearSurface,2,-1']
    # Center:
    FinalLines += concentrated_load('CenterAxleCF','CenterAmp') # Defining concentrated force on rear axle
    FinalLines += [PartNames[1]+'.CenterAxleCM,2,1']
    FinalLines += concentrated_load('CenterContactCF','CenterAmp') # Defining concentrated force on rear contact
    FinalLines += [PartNames[1]+'.CenterSurface,2,-1']
    # Front:
    FinalLines += concentrated_load('FrontAxleCF','FrontAmp') # Defining concentrated force on rear axle
    FinalLines += [PartNames[1]+'.FrontAxleCM,2,1']
    FinalLines += concentrated_load('FrontContactCF','FrontAmp') # Defining concentrated force on rear contact
    FinalLines += [PartNames[1]+'.FrontSurface,2,-1']
    # ----- Define horizontal displacement of vehicle:
    # Passenger Vehicle:
    FinalLines += boundary_condition(PartNames[1],'VehicleCMs','Applied',[total_bridge_length-Lv],[1],'NEW') # Beam CM 
    ######
    # Assign Pin:
    FinalLines += boundary_condition(PartNames[0],'VerticalSupport','Support',[],[1,2],'NEW')

    # Assign Roller:
    FinalLines += boundary_condition(PartNames[0],'AllNodes','Support',[],[1],'NEW')

    FinalLines += boundary_condition(PartNames[1],'VehicleCMsForRotation','Support',[],[6],'NEW')
    # ----- Output requests
    # Field output:
    FinalLines += output_request(0,'Field', 1,'Results','U, V, A')
    # History output:
    FinalLines += output_request(0,'History', 1,'Results','U, V, A')
    # End of step
    FinalLines += ['*End step']

    # ----- Save input file in separate text file:
    np.savetxt(file_name+'.inp',FinalLines,fmt="%s")

    ######################################################################################
    #-----------------------------------RUN ABAQUS----------------------------------------
    os.system('abaqus job='+file_name)
    time.sleep(5)
    start_time = time.time()
    while os.path.exists(file_name+'.lck'):
        pass
    end_time = time.time()
    duration = end_time - start_time
    print(duration)
    #-----------------------------------GENERATE ABAQUS REPORT----------------------------
    os.system('abaqus ODBreport job='+file_name+' history')
    time.sleep(5)
    res_df = {
            'veh_cm_nodes':cm_ids,
            'bridge_nodes_inter':bridge_nodes_of_interest
            }
    return res_df
