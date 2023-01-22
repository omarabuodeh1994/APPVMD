# This script is developed to compute the eigenfrequencies of a passenger vehicle:
# An input file will be generated and submitted to the Abaqus solver
from input_file_commands import*
from vehicle_library import*
from computation_functions import*
import numpy as np
import os

######################################################################################
#-----------------------------------HEADER--------------------------------------------
v = [0,0,0,0]
HeaderLines = input_header('JobTest','VEHICLE_FREQUENCY',v)
FinalLines = HeaderLines

#######################################################################################
#-----------------------------------PARTS---------------------------------------------
# ----- Vehicle part:
# FinalLines += ['** PARTS VEHICLE'] + SprungMassPart(XPosition,YPosition,DimensionMass,PartNames[1],K,C,MatName[1],DimensionMass[2])
LinesVehicle = passenger_vehicle_part('VEHICLE_FREQUENCY',0.0) # passenger vehicle model
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
#-----------------------------------ASSEMBLY------------------------------------------
# ----- Assemble parts:
FinalLines += assemble_parts(['VEHICLE_FREQUENCY'])
# ----- Assign rigid body definition to sprung mass part:
FinalLines += rigid_body ('VEHICLE_FREQUENCY','EntireBeam',['',''],'BeamCM') # vehicle body
FinalLines += rigid_body ('VEHICLE_FREQUENCY','RearAxle',['',''],'RearAxleCM') # rigid body rear axle
FinalLines += rigid_body ('VEHICLE_FREQUENCY','FrontAxle',['',''],'FrontAxleCM') # rigid body front axle

# ----- End Assembly:
FinalLines += ['*End assembly']

######################################################################################
#-----------------------------------MATERIAL DEFINITION-------------------------------
# ----- Sprung mass definition:
density_rear = compute_density('Circle',[radius_rear,length_rear],mass_rear)
density_center = compute_density('Circle',[radius_center,length_center],mass_center)
density_front = compute_density('Circle',[radius_front,length_front],mass_front)
FinalLines += mat_definition('VEHICLE_FREQUENCY'+'RearBeam',1.0e10,0.2,density_rear,[])
FinalLines += mat_definition('VEHICLE_FREQUENCY'+'CenterBeam',1.0e10,0.2,density_center,[])
FinalLines += mat_definition('VEHICLE_FREQUENCY'+'FrontBeam',1.0e10,0.2,density_front,[])
FinalLines += mat_definition('VEHICLE_FREQUENCY'+'RearAxle',1.0e10,0.2,mass_rear_axle/(0.1**3),[])
FinalLines += mat_definition('VEHICLE_FREQUENCY'+'FrontAxle',1.0e10,0.2,mass_front_axle/(0.1**3),[])
######################################################################################
#-----------------------------------SUPPORT BOUNDARY CONDITIONS-----------------------
FinalLines += boundary_condition('VEHICLE_FREQUENCY','VehicleCMs','Support',[],[1],'MOD')
FinalLines += boundary_condition('VEHICLE_FREQUENCY','VehicleCMsForRotation','Support',[],[6],'MOD')
FinalLines += boundary_condition('VEHICLE_FREQUENCY','RearSurface','Support',[],[2],'MOD')
FinalLines += boundary_condition('VEHICLE_FREQUENCY','FrontSurface','Support',[],[2],'MOD')

######################################################################################
#-----------------------------------step DEFINITION-----------------------------------
FinalLines += step('Frequency',0,0,0.0,'Freqstep',5)
# ----- Output requests
FinalLines += ['*Output, field, variable=PRESELECT']
# End of step
FinalLines += ['*End step']

# ----- Save input file in separate text file:
np.savetxt('vehicle_frequency.inp',FinalLines,fmt="%s")
######################################################################################
#-----------------------------------RUN ABAQUS----------------------------------------
import os
os.system("abaqus job=vehicle_frequency cpus=2")
######################################################################################
