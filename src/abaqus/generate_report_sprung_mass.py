# This script is used to generate an abaqus report file and take in results
######################################################################################
#-----------------------------------GENERATE ABAQUS REPORT----------------------------
import os
# os.system("abaqus ODBreport job=yang_sprung_mass history")

# Import functions for extracting results:
from python_functions import *
from post_processing_functions import *
import math
# Read report Abaqus report:
reportLines = read_inp_file('yang_sprung_mass.rep')

# Inputs:
Lapproach = 0.0
BridgeLength = 25.0 
Lexit = 0.0
Lv = 0.0
V = 10.0
dT = 0.001 
sprung_mass_freq = 2.08
bridge_freq =  [3.83,15.32,34.46] 
NumberOfElements = 2500

# Calculation:
TotalLengthTraveled = Lapproach+BridgeLength+Lexit-Lv
TimePeriod = round(TotalLengthTraveled/V,2)


# Extract indices for the following:
step_name = "Step name 'DynamicStep'\n"
ind_Dynamicstep = reportLines.index(step_name)

######################################################################################
#------------------------------EXTRACTING RESULTS-------------------------------------
# importing the required module
import matplotlib.pyplot as plt

#------------------------------ On Bridge Data
TimeStartBridge = (Lapproach-Lv)/V
TimeEndBridge = (Lapproach+BridgeLength)/V
# ------------------ Sprung Mass:
# extracted_res_df = pd.DataFrame(extracted_res).T
# extracted_res_df.to_csv('extracted_results.csv')
sprung_mass_cm = "'Node VEHICLEPART.5'"
# ind_Staticresults = reportLines[ind_Staticstep:].index('  History Region '+FrontAxleCM+'\n')
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+sprung_mass_cm+'\n')
# Displacement:
sprung_mass_displacement = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'U2'\n",TimePeriod)
sprung_mass_displacement_truncated = time_history(sprung_mass_displacement,TimeStartBridge,TimeEndBridge,1)
plt.figure()
plt.plot(sprung_mass_displacement_truncated['Frame'],sprung_mass_displacement_truncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Displacement (m)'],'Sprung Mass Displacement Plot')
# Displacement:
sprung_mass_velocity = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'V2'\n",TimePeriod)
sprung_mass_velocity_truncated = time_history(sprung_mass_velocity,TimeStartBridge,TimeEndBridge,1)
plt.figure()
plt.plot(sprung_mass_velocity_truncated['Frame'],sprung_mass_velocity_truncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Velocity (m/s)'],'Sprung Mass Velocity Plot')
# Acceleration Data:
sprung_mass_acceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
sprung_mass_acceleration_truncated = time_history(sprung_mass_acceleration,TimeStartBridge,TimeEndBridge,0)
filtered_sprung_mass_acceleration = butter_lowpass_filter(sprung_mass_acceleration_truncated['Response'], 50, 1000, 8)
# Save results in csv:
sprung_mass_results = np.vstack((sprung_mass_displacement_truncated['Frame'].values,sprung_mass_displacement_truncated['Response'].values,sprung_mass_velocity_truncated['Response'].values,sprung_mass_acceleration_truncated['Response'].values)).T
sprung_mass_results_df = pd.DataFrame(sprung_mass_results,columns=['time','displacement','velocity','acceleration'])
sprung_mass_results_df.to_csv('sprung_unsprung_mass.csv',index=False)
# Plot:
plt.figure()
plt.plot(sprung_mass_acceleration_truncated['Frame'], filtered_sprung_mass_acceleration)
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Sprung Mass Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_sprung_mass = get_psd_welch(filtered_sprung_mass_acceleration,0.25,1000)
PSD_sprung_mass_peaks = peak_picking(PSD_sprung_mass,0.5)
# Plot:
plt.figure()
plt.plot(PSD_sprung_mass['Frequency'],PSD_sprung_mass['PSD'])
plt.scatter(x = PSD_sprung_mass_peaks['Frequency'], y = PSD_sprung_mass_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Sprung Mass',[sprung_mass_freq,bridge_freq[0],bridge_freq[1],bridge_freq[2]],[0,40])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
#------------------------------Bridge
all_bridge_nodes = [int(elem) for elem in np.linspace(1,NumberOfElements+1,NumberOfElements+1)]
extracted_bridge_acc = rep_file_extraction(all_bridge_nodes,'BRIDGEPART','A2','DynamicStep',reportLines,TimePeriod)
extracted_bridge_acc.to_csv('extracted_results.csv',sep=',',index=False)
bridge_response = np.diag(extracted_bridge_acc[extracted_bridge_acc.columns[1:]])
np.savetxt(bridge_response,'bridge_results.csv',delimiter=',')
ind_CN = reportLines[ind_Dynamicstep:].index('  History Region '+"'Node BRIDGEPART.51'"+'\n') 
# Displacement Results:
Bridgedisplacement = extract_results(reportLines[ind_CN:],"    History Output 'U2'\n",TimePeriod)
BridgedisplacementTruncated = time_history(Bridgedisplacement,TimeStartBridge,TimeEndBridge,1)
filtered_bridge_displacement = butter_lowpass_filter(BridgedisplacementTruncated['Response'], 50, 1000, 8)
# Plot:
plt.figure()
plt.plot(BridgedisplacementTruncated['Frame'],filtered_bridge_displacement)
time_history_plot_format(16,plt,['Time (sec)','Displacement (m)'],'Bridge Midspan Displacement Plot')
# Velocity Results:
Bridgevelocity = extract_results(reportLines[ind_CN:],"    History Output 'V2'\n",TimePeriod)
BridgevelocityTruncated = time_history(Bridgevelocity,TimeStartBridge,TimeEndBridge,1)
filtered_bridge_velocity = butter_lowpass_filter(BridgevelocityTruncated['Response'], 50, 1000, 8)
# Plot:
plt.figure()
plt.plot(BridgevelocityTruncated['Frame'],filtered_bridge_velocity)
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s)'],'Bridge Acceleration Plot')
# Acceleration Results:
BridgeAcceleration = extract_results(reportLines[ind_CN:],"    History Output 'A2'\n",TimePeriod)
BridgeAccelerationTruncated = time_history(BridgeAcceleration,TimeStartBridge,TimeEndBridge,1)
filtered_bridge_acceleration = butter_lowpass_filter(BridgeAccelerationTruncated['Response'], 50, 1000, 8)
# Plot:
plt.figure()
plt.plot(BridgeAccelerationTruncated['Frame'],filtered_bridge_acceleration)
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Bridge Acceleration Plot')
# plt.savefig('bridge_CN_Acc.pdf')
# plt.show()
# FFT:
PSD_bridge = get_psd_welch(filtered_bridge_acceleration,0.25,1000)
PSD_bridge_peaks = peak_picking(PSD_bridge,0.05)
# Plots
plt.figure()
plt.plot(PSD_bridge['Frequency'],PSD_bridge['PSD'])
plt.scatter(x = PSD_bridge_peaks['Frequency'], y = PSD_bridge_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Bridge CN',[sprung_mass_freq,bridge_freq[0],bridge_freq[1],bridge_freq[2]],[0,100])
# plt.savefig('bridge_CN_freq.pdf')
# plt.show()

# ========================================================================
# FDD:
# import seaborn as sns
# import PyOMA as oma
# data = np.vstack((VehicleAccelerationTruncated['Response'],FrontAxleAccelerationTruncated['Response'].to_numpy(),RearAxleAccelerationTruncated['Response'].to_numpy())).T
# FDD = FDDsvp(data, 1000,0.5)
# plt.show()