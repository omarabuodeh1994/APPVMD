# This script is used to generate an abaqus report file and take in results
######################################################################################
#-----------------------------------GENERATE ABAQUS REPORT----------------------------
import os
os.system("abaqus ODBreport job=passenger_robs_bridge_surf_1 history")

# Import functions for extracting results:
from python_functions import *
from post_processing_functions import *
from pyoma import *
# Read report Abaqus report:
reportLines = read_inp_file('passenger_robs_bridge_surf_1.rep')
# Inputs:
Lapproach = 7.0
BridgeLength = 21.3 
Lexit = 7.0
Lv = 6.0
V = 10.0
dT = 0.001 
vehicle_freq = 2.1
front_axle_freq = 10.18  # 3.25
rear_axle_freq = 15.4 
bridge_freq =  6.0 #2.08 
# Calculation:
TotalLengthTraveled = Lapproach+BridgeLength+Lexit-Lv
TimePeriod = round(TotalLengthTraveled/V,2)


# Extract indices for the following:
step_name = "Step name 'Dynamicstep'\n"
ind_Dynamicstep = reportLines.index(step_name)

######################################################################################
#------------------------------EXTRACTING RESULTS-------------------------------------
# importing the required module
import matplotlib.pyplot as plt

#------------------------------ On Bridge Data
TimeStartBridge = (Lapproach-Lv)/V
TimeEndBridge = (Lapproach+BridgeLength)/V
# ------------------ Vehicle Body:
VehicleCM = "'Node VEHICLEPART.4'"
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+VehicleCM+'\n')
# Acceleration Data:
VehicleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
VehicleAccelerationTruncated = time_history(VehicleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Save results in csv:
VehicleAccelerationTruncated.to_csv('body_surf2.csv',header=False,index=False)
# Plot:
plt.figure()
plt.plot(VehicleAccelerationTruncated['Frame'], VehicleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Vehicle Body Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_vehicle = get_psd_welch(VehicleAccelerationTruncated['Response'],0.5,1000)
PSD_vehicle_peaks = peak_picking(PSD_vehicle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_vehicle['Frequency'],PSD_vehicle['PSD'])
plt.scatter(x = PSD_vehicle_peaks['Frequency'], y = PSD_vehicle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Vehicle Body',[vehicle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
# ------------------ Front Axle:
FrontAxleCM = "'Node VEHICLEPART.19'"
# ind_Staticresults = reportLines[ind_Staticstep:].index('  History Region '+FrontAxleCM+'\n')
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+FrontAxleCM+'\n')
# Acceleration Data:
FrontAxleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
FrontAxleAccelerationTruncated = time_history(FrontAxleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Save results in csv:
FrontAxleAccelerationTruncated.to_csv('front_axle_surf2.csv',header=False,index=False)
# Plot:
plt.figure()
plt.plot(FrontAxleAccelerationTruncated['Frame'], FrontAxleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Front Axle Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_front_axle = get_psd_welch(FrontAxleAccelerationTruncated['Response'],0.5,1000)
PSD_vehicle_peaks = peak_picking(PSD_front_axle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_front_axle['Frequency'],PSD_front_axle['PSD'])
plt.scatter(x = PSD_vehicle_peaks['Frequency'], y = PSD_vehicle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Front Axle',[front_axle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
# ------------------ Rear Axle:
RearAxleCM = "'Node VEHICLEPART.14'"
# ind_Staticresults = reportLines[ind_Staticstep:].index('  History Region '+FrontAxleCM+'\n')
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+RearAxleCM+'\n')
# Acceleration Data:
RearAxleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
RearAxleAccelerationTruncated = time_history(RearAxleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Save results in csv:
RearAxleAccelerationTruncated.to_csv('rear_axle_surf2.csv',header=False,index=False)
# Plot:
plt.figure()
plt.plot(RearAxleAccelerationTruncated['Frame'], RearAxleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Rear Axle Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_rear_axle = get_psd_welch(RearAxleAccelerationTruncated['Response'],0.5,1000)
PSD_rear_axle_peaks = peak_picking(PSD_rear_axle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_rear_axle['Frequency'],PSD_rear_axle['PSD'])
plt.scatter(x = PSD_rear_axle_peaks['Frequency'], y = PSD_rear_axle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Rear Axle',[rear_axle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
#------------------------------Bridge
ind_CN = reportLines[ind_Dynamicstep:].index('  History Region '+"'Node BRIDGE_PART.52'"+'\n') 
# Acceleration Results:
BridgeAcceleration = extract_results(reportLines[ind_CN:],"    History Output 'A2'\n",TimePeriod)
BridgeAccelerationTruncated = time_history(BridgeAcceleration,TimeStartBridge,TimeEndBridge,1)
# Plot:
plt.figure()
plt.plot(BridgeAccelerationTruncated['Frame'],BridgeAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Bridge Acceleration Plot')
# plt.savefig('bridge_CN_Acc.pdf')
plt.show()
# FFT:
PSD_bridge = get_psd_welch(BridgeAccelerationTruncated['Response'],0.5,1000)
PSD_bridge_peaks = peak_picking(PSD_bridge,0.05)
# Plots
plt.figure()
plt.plot(PSD_bridge['Frequency'],PSD_bridge['PSD'])
plt.scatter(x = PSD_bridge_peaks['Frequency'], y = PSD_bridge_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Bridge CN',[front_axle_freq,bridge_freq],[0,100])
# plt.savefig('bridge_CN_freq.pdf')
# plt.show()

# ========================================================================
# FDD:
import seaborn as sns
import PyOMA as oma
data = np.vstack((VehicleAccelerationTruncated['Response'],FrontAxleAccelerationTruncated['Response'].to_numpy(),RearAxleAccelerationTruncated['Response'].to_numpy())).T
FDD = FDDsvp(data, 1000,0.5)
plt.show()

#------------------------------ Off Bridge Data
TimeStartBridge = 0.0
TimeEndBridge = (Lapproach-Lv)/V
# ------------------ Vehicle Body:
VehicleCM = "'Node VEHICLEPART.4'"
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+VehicleCM+'\n')
# Acceleration Data:
VehicleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
VehicleAccelerationTruncated = time_history(VehicleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Plot:
plt.figure()
plt.plot(VehicleAccelerationTruncated['Frame'], VehicleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Vehicle Body Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_vehicle = get_psd_welch(VehicleAccelerationTruncated['Response'],0.5,1000)
PSD_vehicle_peaks = peak_picking(PSD_vehicle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_vehicle['Frequency'],PSD_vehicle['PSD'])
plt.scatter(x = PSD_vehicle_peaks['Frequency'], y = PSD_vehicle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Vehicle Body',[vehicle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
# ------------------ Front Axle:
FrontAxleCM = "'Node VEHICLEPART.19'"
# ind_Staticresults = reportLines[ind_Staticstep:].index('  History Region '+FrontAxleCM+'\n')
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+FrontAxleCM+'\n')
# Acceleration Data:
FrontAxleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
FrontAxleAccelerationTruncated = time_history(FrontAxleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Plot:
plt.figure()
plt.plot(FrontAxleAccelerationTruncated['Frame'], FrontAxleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Front Axle Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_front_axle = get_psd_welch(FrontAxleAccelerationTruncated['Response'],0.5,1000)
PSD_vehicle_peaks = peak_picking(PSD_front_axle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_front_axle['Frequency'],PSD_front_axle['PSD'])
plt.scatter(x = PSD_vehicle_peaks['Frequency'], y = PSD_vehicle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Front Axle',[front_axle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
# ------------------ Rear Axle:
RearAxleCM = "'Node VEHICLEPART.14'"
# ind_Staticresults = reportLines[ind_Staticstep:].index('  History Region '+FrontAxleCM+'\n')
ind_Dynamicresults = reportLines[ind_Dynamicstep:].index('  History Region '+RearAxleCM+'\n')
# Acceleration Data:
RearAxleAcceleration = extract_results(reportLines[ind_Dynamicresults:],"    History Output 'A2'\n",TimePeriod)
RearAxleAccelerationTruncated = time_history(RearAxleAcceleration,TimeStartBridge,TimeEndBridge,0)
# Plot:
plt.figure()
plt.plot(RearAxleAccelerationTruncated['Frame'], RearAxleAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Rear Axle Acceleration Plot')
# plt.savefig('center_axle_Acc.pdf')
# plt.show()
# FFT:
PSD_rear_axle = get_psd_welch(RearAxleAccelerationTruncated['Response'],0.5,1000)
PSD_rear_axle_peaks = peak_picking(PSD_rear_axle,0.25)
# Plot:
plt.figure()
plt.plot(PSD_rear_axle['Frequency'],PSD_rear_axle['PSD'])
plt.scatter(x = PSD_rear_axle_peaks['Frequency'], y = PSD_rear_axle_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Rear Axle',[rear_axle_freq,bridge_freq],[0,100])
# plt.savefig('center_axle_freq.pdf')
# plt.show()
#------------------------------Bridge
ind_CN = reportLines[ind_Dynamicstep:].index('  History Region '+"'Node BRIDGE_PART.27'"+'\n') 
# Acceleration Results:
BridgeAcceleration = extract_results(reportLines[ind_CN:],"    History Output 'A2'\n",TimePeriod)
BridgeAccelerationTruncated = time_history(BridgeAcceleration,TimeStartBridge,TimeEndBridge,1)
# Plot:
plt.figure()
plt.plot(BridgeAccelerationTruncated['Frame'],BridgeAccelerationTruncated['Response'])
time_history_plot_format(16,plt,['Time (sec)','Acceleration (m/s^2)'],'Bridge Acceleration Plot')
# plt.savefig('bridge_CN_Acc.pdf')
# plt.show()
# FFT:
PSD_bridge = get_psd_welch(BridgeAccelerationTruncated['Response'],0.5,1000)
PSD_bridge_peaks = peak_picking(PSD_bridge,0.05)
# Plots
plt.figure()
plt.plot(PSD_bridge['Frequency'],PSD_bridge['PSD'])
plt.scatter(x = PSD_bridge_peaks['Frequency'], y = PSD_bridge_peaks['PSD'], marker = 'x', c='r')
frequency_plot_format(16,plt,['Frequency (Hz)','PSD'],'FFT of Bridge CN',[front_axle_freq,bridge_freq],[0,100])
# plt.savefig('bridge_CN_freq.pdf')
# plt.show()

# ========================================================================
# FDD:
import seaborn as sns
import PyOMA as oma
data = np.vstack((VehicleAccelerationTruncated['Response'],FrontAxleAccelerationTruncated['Response'].to_numpy(),RearAxleAccelerationTruncated['Response'].to_numpy())).T
FDD = FDDsvp(data, 1000,0.5)
plt.show()