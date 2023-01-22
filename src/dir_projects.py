from input_params import *
class DirectoryProjects:
    def __init__(self,case_study,type_of_study):
        """This class object is used to define directory to extract vehicle signals from.

        Args:
            case_study (str): string describing the study (Case_0, Case_1, Case_2, Case_3) -- if a different naming convention is to be used then this script needs to follow it consistently.
            type_of_study (str): string describing the study (single_veh,mult_veh,velocity,tire_stiffness)
        """
    # directory paths for different cases and types:
        # regular parametric study according to our paper:
        self.dir = f'../parametric_studies/{type_of_study}/case_studies/{case_study}/'

    # directory path for files within self.dir
        if (type_of_study == 'single_veh') | (type_of_study == 'mult_veh') | (type_of_study == 'tire_stiffness'):
            self.off_bridge_dir = ['/off_bridge'] # off-bridge 
            self.on_bridge_dir = ['/on_bridge'] # on-bridge
            self.vmd_sigs_dir = ['/vmd_sigs'] # vmd signals
        else:
            self.off_bridge_dir = [f'/off_bridge_V_{vel_i}' for vel_i in InputParams(case_study,type_of_study).vel] # off-bridge
            self.on_bridge_dir = [f'/on_bridge_V_{vel_i}' for vel_i in InputParams(case_study,type_of_study).vel] # on-bridge
            self.vmd_sigs_dir = [f'/vmd_sigs_V_{vel_i}' for vel_i in InputParams(case_study,type_of_study).vel] # vmd signals


if __name__ == '__main__':
    import pdb
    x = DirectoryProjects('Case_1','velocity')
    pdb.set_trace()