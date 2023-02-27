class InputParams:
    """This class is used to input parameters for the APPVMD functions during the parametric studies"""
    
    def __init__(self,case_study,type_of_study):
        """This class object is used to define the input parameters used in the APPVMD algorithm.

        Args:
            case_study (str): string describing the study (Case_0, Case_1, Case_2, Case_3) -- if a different naming convention is to be used then this script needs to follow it consistently.
            type_of_study (str): string describing the study (single_veh,mult_veh,velocity,tire_stiffness)

        Returns:
            instances: instances outputting different input parameters that the appvmd algorithm requires.
        """
    # store object inputs:
        self.case_study = case_study
    # bridge input parameters:
        # bridge case studies to specify bridge span lengths in m:
        if case_study == 'Case_0':
            self.bridge_spans = [21.] # arbitrary bridge length of a single bridge class
        elif case_study == 'Case_1': # bridge class from "Using drive-by health monitoring to detect bridge damage considering environmental and operational effects"
            self.bridge_spans = [16.,30.,40.,16.,16.,16.,16.,16.,16.,20.,38.,56.,20.,24.,28.] 
        elif case_study == 'Case_2': # bridge class from "Fatigue analysis and life prediction of composite highway bridge decks under traffic loading"
            self.bridge_spans =  [20.,30.,40.,40.,40.,40.,40.,40.,40.,40.,44.,46.,15.,27.,42.]
        else: # bridge class from "Drive-by health monitoring of highway bridges usingBayesian estimation technique for damage classification"
            self.bridge_spans =  [21.3,30.,40.,21.3,21.3,21.3,21.3,21.3,21.3,20.,38.,56.,13.,28.,44.]
        
        # bridge case studies to specify bridge masses in kg/m:
        if case_study == 'Case_0':
            self.bridge_masses = [5600] # arbitrary bridge length of a single bridge class
        elif case_study == 'Case_1': # bridge class from "Using drive-by health monitoring to detect bridge damage considering environmental and operational effects"
            self.bridge_masses = [11600,11600,11600,11600,11700,11750,9040,11600,15500,9040,11600,15500,11600,11700,11750] 
        elif case_study == 'Case_2': # bridge class from "Fatigue analysis and life prediction of composite highway bridge decks under traffic loading"
            self.bridge_masses = [10400,10400,10400,10302,10400,10600,10400,12265,14080,10400,12265,14080,10302,10400,10600] 
        else: # bridge class from "Drive-by health monitoring of highway bridges usingBayesian estimation technique for damage classification"
            self.bridge_masses = [5600,5600,5600,5600,5790,5840,5600,6700,9800,5600,6700,9800,5600,5790,5840] 
        
        # bridge case studies to specify bridge moment of inertia values in m^4:
        if case_study == 'Case_0':
            self.bridge_Is = [0.0837] # arbitrary bridge length of a single bridge class
        elif case_study == 'Case_1': # bridge class from "Using drive-by health monitoring to detect bridge damage considering environmental and operational effects"
            self.bridge_Is = [0.0621,0.0621,0.0621,0.0621,0.0744,0.0863,0.0324,0.0621,0.0915,0.0324,0.0621,0.0915,0.0621,0.0744,0.0863] 
        elif case_study == 'Case_2': # bridge class from "Fatigue analysis and life prediction of composite highway bridge decks under traffic loading"
            self.bridge_Is = [0.451,0.451,0.451,0.253,0.451,0.714,0.451,0.753,0.997,0.451,0.753,0.997,0.253,0.451,0.714] 
        else: # bridge class from "Drive-by health monitoring of highway bridges usingBayesian estimation technique for damage classification"
            self.bridge_Is = [0.0837,0.0837,0.0837,0.0837,0.177,0.284,0.0837,0.159,0.235,0.0837,0.159,0.235,0.0837,0.177,0.284] 
        
        # bridge case studies to specify bridge elastic modulus values in GPa:
        if case_study == 'Case_0':
            self.bridge_Es = [205] # arbitrary bridge length of a single bridge class
        elif case_study == 'Case_1': # bridge class from "Using drive-by health monitoring to detect bridge damage considering environmental and operational effects"
            self.bridge_Es = [211,211,211,211,211,211,211,211,211,211,211,211,211,211,211] 
        elif case_study == 'Case_2': # bridge class from "Fatigue analysis and life prediction of composite highway bridge decks under traffic loading"
            self.bridge_Es = [205,205,205,205,205,205,205,205,205,205,205,205,205,205,205] 
        else: # bridge class from "Drive-by health monitoring of highway bridges usingBayesian estimation technique for damage classification"
            self.bridge_Es = [200,200,200,200,200,200,200,200,200,200,200,200,200,200,200]

        # bridge case studies to define bridge names:
        if (case_study == 'Case_1') | (case_study == 'Case_2') | (case_study == 'Case_3'): # bridge class from "Using drive-by health monitoring to detect bridge damage considering environmental and operational effects"
            self.bridge_names = ['short_span','med_span','long_span','small_thick','med_thick','large_thick','low_num','med_num',
                    'high_num','low_num_constant_flexure','med_num_constant_flexure',
                    'high_num_constant_flexure','thin_constant_mass',
                    'med_thick_constant_mass','large_thick_constant_mass']
        else:
            self.bridge_names = ['med_thick'] # arbitrary bridge name for a single bridge class
        
    # bridge boundary condition:
        self.boundary_condition = 'pp' # ff = fixed-fixed; pp = pinned-pinned; fp = fixed-pinned
    # span type boundaries:
        self.bridge_bounds = [20,30]

    # parallelization parameters:
        self.parll_bool = False # True to invoke parallelization False to not.
        self.num_jobs_appvmd = 10 # number of jobs to parallelize in the appvmd algorithm
        self.num_jobs_cases = 3 # number of jobs to parallelize for different case studies
    
    # signal processing input parameters:
        self.fs = 1000 # sampling frequency in Hz
        self.cut_off_freq = 50 # cut-off frequency in Hz
        self.freq_res = 0.1 # frequency resolution in Hz
        self.freq_cfr_bounds = [0.4,0.7] # confidence region bounds based on probability of successful frequency extraction
        
    # peak picking input parameters:
        self.bin_width = 0.5 # frequency bin width in Hz
        self.num_peaks = 10 # number of peaks to pick 
        self.num_modes = 7 # number of modes to pick from
        
    # vehicle input parameters:
        self.num_passes = 10 # number of time vehicle drives across bridge
    
    # Velocity:
        # type of study the researcher is interested in. In our study we performed three types of studies (regular, varying velocity, varying tire stiffness)
        if (type_of_study == 'single_veh') | (type_of_study == 'mult_veh') |(type_of_study == 'tire_stiffness'):
            self.vel = [10] # velocity of vehicle in m/s
        else:
            self.vel = [9,13.4,17.9,22.4] # velocity of vehicle in m/s
    
    # Mass:
        # masses are dependent on type of study conducted:
        if type_of_study == 'single_veh':
            self.veh_masses = [4800] # truck
        elif type_of_study == 'mult_veh':
            self.veh_masses = [940,1707,2450,4800] # hatcback, sedan, SUV, and truck
        elif type_of_study == 'velocity':
            self.veh_masses = [1707] # four different velocities for the same vehicle
        else:
            self.veh_masses = [4800] # tire stiffness study was conducted for the truck
        
        # vehicle class names:
        if type_of_study == 'single_veh':
            self.veh_classes = ['truck'] 
        elif type_of_study == 'mult_veh':
            self.veh_classes = ['hatchback','sedan','SUV','truck']
        elif type_of_study == 'velocity':
            self.veh_classes = ['sedan']
        else:
            self.veh_classes = ['truck']
        
        # number of sensors to analyze:
        self.num_sensors = 2
        
        # sensor names:
        self.sensor_names = ['front_axle','rear_axle']
    
    
    # seaborn plot format definition:
        self.plot_format = { 
            'font.size': 18.0,
            'font.family':'Times New Roman',
            'axes.labelsize': 24,
            'axes.titlesize': 16,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'axes.linewidth': 1.5,
            'axes.grid':True,
            'grid.linewidth': 0.8,
            'grid.linestyle':'--',
            'grid.color':'k',
            'lines.linewidth': 2,
            'lines.markersize': 8.0,
            'patch.linewidth': 1.0,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'xtick.major.size': 5.5,
            'ytick.major.size': 5.5,
            'xtick.minor.size': 2.0,
            'ytick.minor.size': 2.0,
            'legend.title_fontsize': None
        }
    
if __name__ == '__main__':
    import pdb
    input_params = InputParams('Case_1','velocity')
    pdb.set_trace()