# This script is developed to model a simple sprung mass example with mass M and spring stiffness K
# An input file will be generated and submitted to the Abaqus solver
from input_file_commands import *

# Given:
# Bridge:
E = 27.5e9 # 200.0e09
PRatio = 0.2
BeamMass = 4800.0 # 5600.0
Height = (12.0*0.12/2.0)**0.5 # 1.9282 # 
Width = 2.0/Height # 0.1409 # 
Lapproach = 6.0
BridgeLength = 25.0
Lexit = 6.0
NumberOfElements = 50

# Damage inputs
DamageLocation = []
CrackRatio = []

BeamGeometry = [Width,Height,BridgeLength,Lapproach,Lexit]
FEInput = [NumberOfElements,DamageLocation,CrackRatio]

bridge_frequency(E,BeamMass,PRatio,BeamGeometry,FEInput)

######################################################################################
#-----------------------------------RUN ABAQUS----------------------------------------
import os
os.system("abaqus job=bridge_frequencyuency cpus=2")


