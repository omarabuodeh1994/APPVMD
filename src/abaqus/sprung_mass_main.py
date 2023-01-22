import numpy as np
from python_functions import *
from input_file_commands import *
from vehicle_library import *
import time
# Given:
# Bridge:
E =  27.5e09 # 200.0e9
PRatio = 0.2
BeamMass = 4800 # 5600.0
Height =  (12.0*0.12/2.0)**0.5 # 1.9282
Width = 2.0/Height # 0.1409 
Lapproach = 0.0
BridgeLength = 25.0
Lexit = 0.0
NumberOfElements = 2500
V = 10.0 

# Damage inputs
DamageLocation = []
CrackRatio = []
time_step = 0.001 # time step in analysis in sec
surface_roughness_bool = 0 # boolean to include surface (0 = No and 1 = Yes)
road_class = 'A' # ISO road class (A,B,C)

BeamGeometry = [Width,Height,BridgeLength,Lapproach,Lexit]
FEInput = [NumberOfElements,DamageLocation,CrackRatio]

# Vehicle:
DimensionMass = [0.1,0.1,0.1] # Dimensions of mass block
M = 1200.0 # Mass of sprung mass in kg
K = 500e3 # Spring stiffness N/m
C = 0.0e03 # Damping of suspension Ns/m

# Position of center of mass node: 
XPosition = -Lapproach
YPosition = 1.0

# Name of parts:
PartNames = ['BridgePart','VehiclePart']
MatName = ['BridgeMat','VehicleMat']

######################################################################################
#-----------------------------------HEADER--------------------------------------------
v = [0,0,0,0]
HeaderLines = input_header('JobTest',PartNames[0],v)
FinalLines = HeaderLines
#-----------------------------------PARTS---------------------------------------------
# ----- Vehicle part:
# FinalLines += ['** PARTS VEHICLE'] + SprungMassPart(XPosition,YPosition,DimensionMass,PartNames[1],K,C,MatName[1],DimensionMass[2])
LinesVehicle = sprung_mass(XPosition,YPosition,PartNames[1],K,C, 'SprungMass', 0.1)
FinalLines += ['** PARTS VEHICLE'] + LinesVehicle
######################################################################################
#-----------------------------------CALCULATION---------------------------------------
total_bridge_length = Lapproach + BridgeLength + Lexit
time_period = (total_bridge_length)/V
time_vehicle_driving = np.round(crange(0,total_bridge_length/V,time_step),3)
vehicle_position_path = pd.DataFrame({'sprung_mass':crange(0.0,total_bridge_length,V*time_step)})

# ----- Bridge part:
# birdge_part_lines = bridge_part(PartNames[0],E,PRatio,BeamGeometry,FEInput, MatName[0], 'No')
bridge_part_lines = suspended_bridge_part(PartNames[0],E,PRatio,BeamGeometry,FEInput, MatName[0], 'No')
FinalLines += ['** PARTS BRIDGE'] + bridge_part_lines['Lines']
nodes_of_interest = bridge_part_lines['node_ID_of_interest']
bridge_node_ids = bridge_part_lines['node_ID']
######################################################################################
#-----------------------------------ASSEMBLY------------------------------------------
# ----- Assemble parts:
FinalLines += assemble_parts(PartNames)

# ----- Assign rigid body definition to sprung mass part:
FinalLines += rigid_body(PartNames[1],'Set',['',''],'CM') # vehicle body

# ----- Define Node sets for results of interest:
# FinalLines += Nset('Results',PartNames[1]+'.Center,'+PartNames[1]+'.Base,'+PartNames[0]+'.NodesOfInterest',0)
FinalLines += nset('Results',PartNames[1]+'.CM,'+PartNames[1]+'.TireNode,'+PartNames[0]+'.NodesOfInterest,'+PartNames[0]+'.AllNodes',0)

# ----- End Assembly:
FinalLines += ['*End assembly']
#-----------------------------------MATERIAL DEFINITION------------------------------------------
# Vehicle
density_sprung_mass = compute_density('Rectangle',[0.1,0.1,0.1],M)
FinalLines += mat_definition('SprungMass',1.0e10,0.2,density_sprung_mass,[])
# Bridge
FinalLines += mat_definition(MatName[0],E,PRatio,BeamMass/(Width*Height),[])

######################################################################################
#-----------------------------------SURFACE DEFINITION--------------------------------
FinalLines += surface_interaction('Surface',PartNames[1]+'.Surf',PartNames[0]+'.BeamSurf')

######################################################################################
#-----------------------------------SUPPORT BOUNDARY CONDITIONS-----------------------
# ----- Apply BC on bridge:
# Assign Pin:
FinalLines += boundary_condition(PartNames[0],'VerticalSupport','Support',[],[1,2],'MOD')

# Assign Roller:
FinalLines += boundary_condition(PartNames[0],'AllNodes','Support',[],[1],'MOD')

######################################################################################
#-----------------------------------STEP DEFINITION-----------------------------------
# ----- Define implicit dynamic step:
FinalLines += step('Dynamic',(Lapproach+BridgeLength+Lexit)/V,time_step,-0.41421,'DynamicStep',0)

# ----- Surface definition:
# surface_roughness_df = amplitude_of_surface(surface_roughness_bool,road_class,total_bridge_length,vehicle_position_path,time_vehicle_driving,['sprung_mass'],[K,C])
phase_angles = pd.read_csv('PhaseAngles5.csv', sep=',',header=None)
surface_roughness_df = surface_roughness(surface_roughness_bool,'A', total_bridge_length, time_step)
amp_contact_df = amplitude_of_surface(surface_roughness_df,vehicle_position_path,time_vehicle_driving,'sprung_mass',[K,C,M])
amp_surface_mass_df = amp_contact_df['amplitude_mass']
amp_surface_contact_df = amp_contact_df['amplitude_contact']

# ----- Amplitude definition:
# Elevation:
amp_surface_mass_df.to_csv('amplitude_mass.inp',columns=['Time_sec_','sprung_mass'],header=False,index=False) # sprung mass elevation
amp_surface_contact_df.to_csv('amplitude_contact.inp',columns=['Time_sec_','sprung_mass'],header=False,index=False) # unsprung mass elevation
FinalLines += amplitude('MassAmp','amplitude_mass.inp')
FinalLines += amplitude('ContactAmp','amplitude_contact.inp')
# ----- Define concentrated force to simulate surface roughness:
FinalLines += concentrated_load('SprungMassCF','MassAmp') # Defining concentrated force on rear axle
# FinalLines += [PartNames[1]+'.CM,2,'+str(K)]; FinalLines += [PartNames[1]+'.TireNode,2,'+str(-K)]
FinalLines += [PartNames[1]+'.CM,2,1']
FinalLines += concentrated_load('ContactCF','ContactAmp') # Defining concentrated force on rear axle
FinalLines += [PartNames[1]+'.TireNode,2,-1']

# ----- Define horizontal displacement of vehicle:
# @ Center of mass:
FinalLines += boundary_condition(PartNames[1],'VehicleCMs','Applied',[Lapproach+BridgeLength+Lexit],[1],'MOD')
# ----- Output requests
# Field output:
FinalLines += output_request(1,'Field', 1,'Results','U, V, A, CF')
# Contact output:
FinalLines += ['*CONTACT OUTPUT,NSET='+PartNames[1]+'.TireNode']
FinalLines += ['CSTRESS']
# History output:
FinalLines += output_request(0,'History', 1,'Results','U, V, A, CF')
# End of step
FinalLines += ['*End step']

np.savetxt('yang_sprung_mass.inp',FinalLines,"%s")

######################################################################################
#-----------------------------------RUN ABAQUS----------------------------------------
import os
os.system("abaqus job=yang_sprung_mass")
time.sleep(10)
while True:
    with open('yang_sprung_mass.sta','r') as f:
        if f.readlines()[-1] == ' THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n':
            break
    # print(f.readlines()[-1])