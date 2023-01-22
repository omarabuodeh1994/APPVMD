# Description

This project includes python scripts that analyze signals measured by the acceleration response of vehicles passing across bridges. *Finite element* (FE) models were used to carry out *vehicle-bridge interaction* (VBI) simulations and autonomously extract the bridge frequencies from the vehicle signals. The proposed algorithm is called *Autonomous Peak Picking Variational Mode Decomposition* (APPVMD), which uses an ensemble of literature-established signal processing techniques to de-noise signals for more robust bridge frequency extractions. It is worth mentioning that the data used herein was taken from a separate FE simulation that is not part of this project.

## Numerical Setup
We used FE models to simulate VBI simulations where four different vehicle models, compiled from the literature, and 45 bridge models that were designed by the authors from realistic bridges. Each bridge is crossed with a single vehicle model *n* number of times to obtain a large sample for a more confidence detection. These runs are categorized into datasets; on and off bridge data. On-bridge data is taken as the time period in which the front axle enters the bridge and the rear axle exits the bridge, whereas the off-bridge data is taken as the time period where the vehicle is approaching the bridge before the front axle enters the bridge. This helps the algorithm cross out any spurious modes caused by road surface roughness effects.  

## Objective
* Verify the application of APPVMD to successfully detect bridge frequencies from a broad range of vehicle and bridge models.
* Compare the success rates of bridge frequency extractions between each vehicle model and assess the factors that influence them.
* Investigate the effect of the bridge's stiffness, mass, and span length on the success rate of bridge frequency extractions.
* Conduct a sensitivity study to investigate the effect of increasing speed of sedan vehicle classes on the algorithm's detection accuracy/confidence.
* Conduct a sensitivity study to investigate the effect of decreasing tire stiffness (i.e., reducing surface roughness effect) in the truck model on the algorithm's detection accuracy/confidence.

## How to employ APPVMD
The user should have acceleration readings taken from accelerometers that are attached to two opposing parts, longitudinally, of the vehicle (i.e., front and rear parts). In this study, we used the front and rear axles as the points to measure vehicle acceleration signals as the axles are known to attenuate noise coming from the sprung body (i.e., high frequency noise from the engine). The data should be organized into csv files containing the on and off bridge portion data for each pass (e.g., off_bridge_data_1.csv and on_bridge_data_1.csv for data of first bridge crossing) where each column contains the acceleration reading from each sensor (e.g., front_axle and rear_axle). An example of a typical csv file is shown below.

| time | front_axle  | rear_axle |
| :-------------: | :-------------: | :-------------: |
| $T_{0}$  | $Acc_{front,0}$  | $Acc_{rear,0}$  |
| $T_{1}$  | $Acc_{front,1}$  | $Acc_{rear,1}$  |
|  .  |  .  |  .  |
|  .  |  .  |  .  |
|  .  |  .  |  .  |
|  $T_{n-1}$  |  $Acc_{front,n-1}$  |  $Acc_{rear,n-1}$  |

### Inputs
To employ the APPVMD functions, input variables associated with the type of study must be given. These are defined in the input_params.py and dir_projects.py. Each file is designated for a specific purpose. The input_params.py is responsible for storing the variables like vehicle properties, bridge properties, and signal processing parameters and the dir_projects.py is responsible for defining the paths of the files that are going to be analyzed.
