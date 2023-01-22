import numpy as np
from python_functions import*
from computation_functions import*
import pandas as pd
import pdb
# ----------------------------------
# Abaqus Part Construction Function:
# ----------------------------------
def input_header(JobName,ModelName,PreprintOptions):
    """This function is used to call the header portion of an Abaqus input file
    It takes three inputs: the string of the job name 'JobName', a string for model name 
    'ModelName' and a vector of booleans for preprinting options (echo,model,history,contact)
    'PreprintOptions'

    Args:
        JobName (str): name of job
        ModelName (str): name of modeling
        PreprintOptions ([bool]): a list of booleans for preprinting options (echo,model,history,contact)

    Returns:
        list[str]: header syntax
    """
    InputLine = ['** --------------- HEADER SECTION'] # Define empty list to store input lines
    echo = 'NO' if PreprintOptions[0] == 0 else 'Yes'
    model = 'NO' if PreprintOptions[1] == 0 else 'Yes'
    history = 'NO' if PreprintOptions[2] == 0 else 'Yes'
    contact = 'NO' if PreprintOptions[3] == 0 else 'Yes'
    InputLine.append('*Heading')
    InputLine.append('** Job name: ' + JobName + ' Model name: ' + ModelName)
    InputLine.append('*Preprint, echo='+echo+', model='+model+', history='+history+
    ', contact='+contact)
    return InputLine

def rectangle_corners(node_id,x,y):
    """This function is used to create corners for a rectangle.

    Args:
        node_id (int): first node ID for rectangle.
        x (float): position in the x axis.
        y (float): position in the y axis.

    Returns:
        str: abaqus input command of corner in a rectangle.
    """
    return str(node_id)+','+str(x)+','+str(y)

def plane_stress_node(start_node_ID,CM_x,CM_y):
    """This function is used to create a plane stress nodes.

    Args:
        start_node_ID (int): first node ID for this nodes.
        CM_x (float): center of mass of element in the x direction.
        CM_y (float): center of mass of element in the y direction. 

    Returns:
        dict: dictionary comprising of lines, corner node IDs and center of mass node {'lines','corner_node_ID','node_CM'}.
    """
    Lines = [rectangle_corners(start_node_ID,CM_x-0.05,CM_y-0.05)] # bottom left corner 
    Lines += [rectangle_corners(start_node_ID+1,CM_x+0.05,CM_y-0.05)] # bottom right corner 
    Lines += [rectangle_corners(start_node_ID+2,CM_x+0.05,CM_y+0.05)] # top right corner 
    Lines += [rectangle_corners(start_node_ID+3,CM_x-0.05,CM_y+0.05)] # top left corner 
    Lines += [rectangle_corners(start_node_ID+4,CM_x,CM_y)] # center of rectangle 
    corner_node_id = []
    for i in range(start_node_ID,start_node_ID+4):
        corner_node_id.append(i)
    CM_ID = start_node_ID+4
    results_dict_ = {'Lines': Lines, 'corner_node_ID':corner_node_id,'node_CM':CM_ID}
    return results_dict_ # Lines, corner_node_id, CM_ID

def plane_stress_element(StartElemID,corner_node_id):
    """This function is used to create a plane stress element using abaqus input commands.

    Args:
        StartElemID (int): first element ID.
        corner_node_id ([int]): list of corner node IDs.

    Returns:
        dict: dictionary comprising of a list of abaqus input commands to create plane stress elements and list of corner node IDs of plane stress element {'Lines','element_IDs'}.
    """
    Lines = []
    ElemID = []
    for i in corner_node_id:
        Lines.append(str(StartElemID)+','+str(i)[1:-1])
        ElemID.append(StartElemID)
        StartElemID += 1
    results_dict_ = {'Lines':Lines,'element_IDs':ElemID}
    return results_dict_ # Lines,ElemID

def nset(Name,NodeID,GenerateBool):
    """This function is used to define node sets in abaqus input file.

    Args:
        Name (string): name of node set defined.
        NodeID ([int]): node ID defined.
        GenerateBool (int): checks whether the generate command should be invoked.

    Returns:
        string: node set in abaqus input syntax.
    """
    Lines = ['*nset, nset='+Name] if GenerateBool == 0 else ['*nset, nset='+Name+', generate']
    if isinstance(NodeID,list):
        NodeIDStr = ''
        for i in str(NodeID)[1:-1]:
            NodeIDStr += i
        Lines+=[NodeIDStr] if GenerateBool == 0 else [NodeIDStr+',1']
    else:
        Lines.append(NodeID)
    return Lines

def elset(Name, ElementID, GenerateBool):
    """This function is used to define an element set in abaqus input.

    Args:
        Name (string): name of element set defined.
        ElementID ([int]): element ID defined previously.
        GenerateBool (int): checks whether the generate command should be invoked.

    Returns:
        string: element set in abaqus input syntax.
    """
    Lines = ['*elset, elset='+Name] if GenerateBool == 0 else ['*elset, elset='+Name+', generate']
    if isinstance(ElementID,list):
        ElementIDStr = ''
        for i in str(ElementID)[1:-1]:
            ElementIDStr += i
        Lines+= [ElementIDStr] if GenerateBool == 0 else [ElementIDStr+', 1']
    else:
        Lines.append(ElementID)
    return Lines

def surf(Name,Type,ID):
    """This function is used to define surface for parts of models.

    Args:
        Name (string): name of the surface definition.
        Type (string): type of surface (e.g. node or element).
        ID ([int]): node or element ID previously defined.

    Returns:
        string: surface definition in abaqus syntax.
    """
    Lines = ['*Surface, name ='+ Name+',type = Node'] if Type == 'Node' else ['*Surface, name = '+ Name+',type = Element']
    if Type == 'Node':
        for i in ID:
            Lines += [str(i)+',0.001']
    else:
        Lines += [ID+',SPOS'] 
    
    return Lines

def release_rotational_dof(ElementID,ElementEnd):
    """This function is used to define a rotation DOF release_rotational_dof in abaqus input syntax.

    Args:
        ElementID (string): element ID of interest to be release_rotational_dofd.
        ElementEnd (string): element end (e.g. s1 = beginning and s2 = end).

    Returns:
        string: release_rotational_dof definition in abaqus input syntax.
    """
    Lines = ['*Release']
    Lines.append(ElementID+','+ElementEnd+',M1')
    return Lines

def beam_section(BeamSet,MatDef,PRatio,Width,Height,radius,LumpMass,section):
    """This function is used to define a rectangular beam section.

    Args:
        BeamSet (string): beam set consisting of elements of interest.
        MatDef (string): material definition name.
        PRatio (float): Poisson Ratio.
        Width (float): width of beam in m.
        Height (float): height of beam in m. 
        LumpMass (string): string that includes the lump mass definition or not.

    Returns:
        string: beam section definition in abaqus input syntax.
    """
    if LumpMass == 'No':
        Lines = ['*Beam Section, elset='+BeamSet+', material='+MatDef+', poisson = '+str(PRatio)
        +',lumped=NO, temperature=GRADIENTS, section='+section]
    else:
        Lines = ['*Beam Section, elset='+BeamSet+', material='+MatDef+', poisson = '+str(PRatio)+',lumped=YES, temperature=GRADIENTS, section='+section]
    Lines += [str(Width)+','+str(Height)] if section == 'RECT' else [str(radius)]
    Lines += ['0.,0.,-1.']
    return Lines

def spring_dashpot_element(ElementID,Name,K,C,PartNameNode,DOF):
    """This function is used to define a spring/dashpot element(s) at the assembly level. 

    Args:
        ElementID (string): element ID (user should be aware of predefined elements)
        Name (string): name of spring/dashpot element definition.
        K (float): stiffness of spring N/m.
        C (float): damping of spring in N/s.
        PartNameNode ([string]): nodes that are connected by spring/dashpot.
        DOF (string): degree of freedom that spring acts in.

    Returns:
        dict: dictionary comprising of a list of spring/dashpot commands in abaqus input syntax and its element ID {'Lines','element_ID'}.
    """
    Lines = ['** ----- SPRING/DASHPOT NAME='+Name+'Spring']
    # Define element for spring definition:
    Lines.append('*Element, type = SPRING2, elset = '+Name+'Spring')
    Lines.append(str(ElementID)+','+PartNameNode[0]+','+PartNameNode[1])
    
    # Define spring element:
    Lines.append('*Spring,elset = '+Name+'Spring')
    Lines.append(DOF)
    Lines.append(str(K))
    
    # --------------------- DASHPOT DEFINITION
    if C > 0:
        # Define element for spring definition:
        Lines.append('*Element, type = DASHPOT2, elset = '+Name+'Dashpot')
        ElementID += 1
        Lines.append(str(ElementID)+','+PartNameNode[0]+','+PartNameNode[1])
        
        # Define dashpot element:
        Lines.append('*Dashpot,elset = '+Name+'Dashpot')
        Lines.append(DOF)
        Lines.append(str(C))
    
    results_dict_ = {'Lines':Lines, 'element_ID':ElementID}
    return results_dict_ # Lines,ElementID

def lines_element(StartElemID,NodeIDArr):
    """This function is used to create a line element using node IDs.

    Args:
        StartElemID (int): first element ID in part.
        NodeIDArr ([int]): list of node IDs of vehicle body.

    Returns:
        string: list of abaqus input commands to create a beam element.
        [int]: list of element IDs in this definition.
    """
    Lines = []
    ElemID = []
    for i in NodeIDArr[0:-1]:
        Lines.append(str(StartElemID)+','+str(i)+','+str(i+1))
        ElemID.append(StartElemID)
        StartElemID += 1
    results_dict_ = {'Lines':Lines,'element_IDs': ElemID}
    return results_dict_ # Lines,ElemID

def vehicle_line_node(X,Y,CM,start_node_ID,beam_portion_lengths,axle_position):
    """This function is used to create vehicle line comprised of nodes.

    Args:
        X ([float]): list of nodal position in the x axis.
        Y ([float]): list of nodal positions in the y axis.
        CM (float): position of center of mass.
        start_node_ID (int): first node ID.
        beam_portion_lengths ([float]): length of the rear, center, and front parts of the beam.
        axle_position ([float]): position of axles for the rear and front axles, respectively.

    Returns:
        [string] : list of abaqus input commands to create nodes for the vehicle body.
        [int]: list of node IDs defined in this node definition.
        [int]: list of node IDs for axle positions on the beam.
        int: node ID for CM node.
    """
    Lines = []
    FinalID = []
    # Insert CM node:
    if CM not in X:
        ind_ = find_closest_to(X,[CM])
        X[ind_] = CM
    # Insert axle position nodes
    for i in axle_position:
        if i not in X:
            X = sorted_insert(X,i)
    # Insert beam portion nodes:
    for i in beam_portion_lengths[:-1]:
        if i not in X:
            X = sorted_insert(X,i)
    
    ind_unique_elements = find_closest_to(X,beam_portion_lengths)
    SelectedIndUniqueElements = [0] + [i for i in ind_unique_elements]
    SelectedElementID = []
    count = 0
    for i in SelectedIndUniqueElements[1:]:
        SelectedElementID.append(np.arange(SelectedIndUniqueElements[count]+1,i+1))
        count += 1
    # Generate abaqus input commands:
    for i in X:
        Lines.append(str(start_node_ID)+','+str(i)+','+str(Y))
        FinalID.append(start_node_ID)
        start_node_ID += 1
    indAxle = find_closest_to(X,axle_position)
    indCM = find_closest_to(X,[CM])
    NodeAxleID = [FinalID[i] for i in indAxle]
    results_dict_ = {'Lines':Lines, 'node_IDs':FinalID,'selected_element_ID':SelectedElementID,'node_axle_IDs':NodeAxleID,'node_CM': FinalID[indCM[0]]}
    return results_dict_ # Lines,FinalID,SelectedElementID,NodeAxleID,FinalID[indCM[0]]

def node_def(NodeID,Position):
    """This function is used to define a node in abaqus input syntax.

    Args:
        NodeID (string): identification of the node to be created (user should be aware of the number).
        Position ([float]): list of coordinates for the node to be created in m. 

    Returns:
        dict: dictionary comprising of list of node definition in abaqus input syntax and node ID {'Lines','node_ID'}. 
    """
    Lines = []
    Lines.append(str(NodeID)+','+str(Position[0])+','+str(Position[1]))
    results_dict_ = {'Lines':Lines,'node_ID':NodeID}
    return results_dict_ # Lines,NodeID

def solid_section(element_set,material_name,thickness):
    """This function is used to create a solid section definition for a plane stress element.

    Args:
        element_set (string): element set name of interest
        material_name (string): material name.
        thickness (float): thickness of plane stress element in m.

    Returns:
        [string]: returns a list of commands in Abaqus input syntax for definition a solid section for a single plane stress element
    """
    Lines = ['*Solid Section, elset='+element_set+', material='+material_name]
    Lines += [str(thickness)]
    return Lines
# -----------------
# Assebmly Portion:
# -----------------
def assemble_parts(PartNames):
    """This function is used to assemble all the parts that were defined.

    Args:
        PartNames ([string]): list of part names previously defined. 

    Returns:
        string: abaqus input command of assembling all the parts. 
    """
    Lines = ['** --------------- ASSEMBLY SECTION']
    Lines.append('*Assembly, name=Assembly')
    for i in PartNames:
        Lines.append('*Instance, name='+i+', part='+i)
        Lines.append('*End instance')
    
    return Lines

def model_change(process,element):
    """This function is used to add or remove elements.

    Args:
        process (str): command to add or remove an element ('ADD' or 'REMOVE').
        element (str): name of element to be removed.

    Returns:
        [str]: list of abaqus input commands for adding/removing elements.
    """
    Lines = ['*MODEL CHANGE,TYPE=ELEMENT,'+process]
    Lines += [element]
    return Lines

def rigid_body(Part,Set,Pin_Tie,CM):
    """This function is used to assign a rigid body definition to a specific part of the model.

    Args:
        Part (string): part of the model of interest.
        Set (string): element set of interest.
        Pin_Tie ([string]): list of string that consists of whether pin/tie should be used and which nodes should be used as pin/tie.
        CM (string): center of mass name.

    Returns:
        string: rigid body definition of model part in abaqus input syntax.
    """
    if Pin_Tie[0] == 'Pin':
        return ['*Rigid body, ref node=' + Part + '.' + CM + ', elset=' + Part + '.' + Set + ', Pin =' + Part + '.' + Pin_Tie[1] + ', position=CENTER OF MASS']

    elif Pin_Tie[0] == 'Tie':
        return ['*Rigid body, ref node=' + Part + '.' + CM + ', elset=' + Part + '.' + Set + ', Tie =' + Part + '.' + Pin_Tie[1] + ', position=CENTER OF MASS']

    else:
        return ['*Rigid body, ref node=' + Part + '.' + CM + ', elset=' + Part + '.' + Set + ', position=CENTER OF MASS']
# ----------------------------
# Material Definition Portion:
# ----------------------------
def mat_definition(Name,ElasticModulus,PoissonRatio,Density,Damping):
    """This function is used to define a material in abaqus input syntax.

    Args:
        Name (string): name of the material definition. 
        ElasticModulus (float): elastic modulus of material in N/m^2. 
        PoissonRatio (float): Poisson's Ratio. 
        Density (float): density of material in kg/m^3. 
        Damping ([float]): list of Rayleigh damping parameters (if present). 

    Returns:
        string: returns material definition command in abaqus input syntax.
    """
    Lines = ['** --------------- MATERIAL DEFINITION NAME = ' + Name, '*Material, name=' + Name, '*Damping']
    if Damping:
        Lines.append(f'{str(Damping[0])},{str(Damping[1])}')
    Lines.append('*Density')
    Lines.extend((str(Density), '*Elastic'))
    Lines.append(f'{str(ElasticModulus)},{str(PoissonRatio)}')
    return Lines
# -------------------------------
# Interaction Definition Portion:
# -------------------------------
def surface_interaction(Name,Slave,Master):
    """This function is used to define a surface interaction between the two bodies using a node-to-surface interaction. 

    Args:
        Name (string): name of surface interactiong being defined. 
        Slave (string): model part and set to be defined as a slave node. 
        Master (master): model part and set to be defined as a master surface.

    Returns:
        string: surface interaction definition in Abaqus input syntax.
    """
    Lines = ['** --------------- SURFACE INTERACTION NAME ='+Name]
    Lines += ['*Surface Interaction, name='+Name]
    Lines += ['1.']
    Lines += ['*Surface Behavior, NO SEPARATION, pressure-overclosure=HARD']
    Lines += ['*Contact Pair, interaction='+Name+', type=NODE TO SURFACE']
    Lines += [Slave+','+Master]
    return Lines
# ------------------------
# Step Definition Portion:
# ------------------------
def concentrated_load(Name,amplitude_name):
    """This function is used to defines a concentrated force command. 

    Args:
        Magnitude (float): magnitude of foat in N.
        set_name (str): amplitude name; leave empty if no amplitude is to be defined. 

    Returns:
        string: definition of concentrated force in abaqus input syntax. 
    """
    Lines = ['** --------- Concentrated Load due to '+Name]
    Lines += ['*Cload,Amplitude='+amplitude_name] if amplitude_name else ['*Cload']
    return Lines

def gravity_acceleration(Part,ModelSet):
    """This function is used to define gravity in step.

    Args:
        Part (str): Name of part.
        ModelSet (str): name of set.

    Returns:
        [string]: list of abaqus input commands that defines gravity
    """
    Lines = ['*Dload']
    Lines+=[Part+'.'+ModelSet+',GRAV,9.81,0.0,-1.']
    return Lines

def tire_model(node_ID,start_position,tire_base_properties,tire_type):
    number_nodes = tire_base_properties[0]
    tire_base = tire_base_properties[1]
    Lines = ['*Node']
    if tire_type == 'point':
        Lines += node_def(node_ID,start_position)['Lines']
    elif tire_type == 'uniform pressure':
        start_point = start_position[0] - tire_base/2.0
        tire_nodes = np.linspace(start_point,start_point+tire_base,number_nodes)
        print(tire_nodes)

def output_request(RestartWrite,OutputType,Frequency,ModelSets,StatesOfInterest):
    """This function is used to request outputs during analysis. It takes 5 inputs:

    Args:
        RestartWrite (Bool): For adding the *Restart command 'RestartWrite', the output request type
        OutputType (string): Output type (e.g. Field or History) that specifies
        Frequency (int): integer that specifies the increment rate.
        ModelSets (string): a string of the set of interest
        StatesOfInterest (string): states that are to be obtained (e.g. displacement - U, velocity - V, acceleration - A).

    Returns:
        string: output request input command.
    """
    Lines = ['** ---------- OUTPUT REQUEST SECTION']
    if RestartWrite  == 1:
        Lines.append('*Restart,write,frequency=0')
    else:
        Lines = []
    
    if OutputType == 'Field':
        Lines.append('*Output,field,Frequency='+str(Frequency))
        Lines.append('*Node Output, nset='+ModelSets)
    else:
        Lines.append('*Output,history,Frequency='+str(Frequency))
        Lines.append('*Node Output, nset='+ModelSets)
    
    Lines.append(StatesOfInterest)
    return Lines

def element_output_request(RestartWrite,OutputType,Frequency,StatesOfInterest='NFORC'):
    """This function is used to request element outputs during analysis. It takes 5 inputs:

    Args:
        RestartWrite (Bool): For adding the *Restart command 'RestartWrite', the output request type
        OutputType (string): Output type (e.g. Field or History) that specifies
        Frequency (int): integer that specifies the increment rate.
        ModelSets (string): a string of the set of interest
        StatesOfInterest (string): states that are to be obtained (e.g. internal forces - NFORC (Default)).

    Returns:
        string: output request input command.
    """
    Lines = ['** ---------- OUTPUT REQUEST SECTION']
    if RestartWrite  == 1:
        Lines.append('*Restart,write,frequency=0')
    else:
        Lines = []
    
    if OutputType == 'Field':
        Lines.append('*Output,field,Frequency='+str(Frequency))
        Lines.append('*Element Output')
    else:
        Lines.append('*Output,history,Frequency='+str(Frequency))
        Lines.append('*Element Output')
    
    Lines.append(StatesOfInterest)
    return Lines

def boundary_condition(PartName,element_set,Type,Magnitude,Direction,OP):
    """This function is used to apply a boundary condition to a specified part of the model.

    Args:
        PartName (string): name of the part containing the elements of interest.
        element_set (string): the element of interest. 
        Type (string): type of boundary condition to be applied; restraining the element from moving (support) or displacing the element (applied).
        Magnitude (float): magnitude of displacement (if user input type = 'applied'). 
        Direction (int): direction of boundary condition to be applied (1 is x direction and 2 is y direction).
        OP (str): string to deativate/activate boundary condition.

    Returns:
        string: returns boundary command in abaqus input syntax. 
    """
    Lines = ['** --------------- BOUNDARY SECTION']
    Lines += ['*Boundary, OP=NEW'] if OP == 'NEW' else ['*Boundary'] 
    ind = 0
    for i in Direction:
        if Type == 'Support':
            Lines.append(PartName+'.'+element_set+','+str(i)+','+str(i))
        else:
            Lines.append(PartName+'.'+element_set+','+str(i)+','+str(i)+','+str(Magnitude[ind]))
        ind += 1
    
    return Lines

def step(stepType,Period,dT,ALPHA,stepName,NumOfEigVal):
    """This function is used to define a step (static, dynamic, or frequency). 

    Args:
        stepType (string): type of step ('Static', 'Dynamic', or 'Frequency').
        Period (float): time period of analysis in seconds.
        dT (float): time step of analysis in seconds.
        ALPHA (float): numerical damping.
        stepName (string): name of step being defined.
        NumOfEigVal (int): number of eigenvalues to be retained in an eigenfrequency analysis.

    Returns:
        string: step definition command in abaqus input syntax.
    """
    Lines = ['** --------------- step SECTION, TYPE = '+stepType+', NAME = '+stepName]
    if stepType == 'Static':
        Lines.append('*Step, name='+stepName+', nlgeom=NO, inc=1000000')
        Lines.append('*Static')
        Lines.append(str(dT)+','+str(Period)+','+str(dT/10.0)+','+str(dT))
    elif stepType == 'Dynamic':
        BETA = 0.25*(1-ALPHA)**2; GAMMA = 0.5-ALPHA
        Lines.append('*Step, name='+stepName+', nlgeom=NO, inc=1000000')
        Lines.append('*Dynamic,ALPHA='+str(ALPHA)+',BETA='+str(BETA)+',GAMMA='+str(GAMMA))
        Lines.append(str(dT)+','+str(Period)+','+str(dT/10.0)+','+str(dT))
    else:
        Lines.append('*Step, name=Freq, nlgeom=NO, perturbation')
        Lines.append('*Frequency, eigensolver=Lanczos, sim, acoustic coupling=on, normalization=mass')
        Lines.append(str(NumOfEigVal)+', , , , ,')
    
    return Lines

def amplitude(name,file):
    """This function is used to define an amplitude with a name and file to be read.

    Args:
        name (str): name of amplitude definition.
        file (str): name of file to be read.

    Returns:
        str: an abaqus command input line that defines an amplitude.
    """
    return ['*Amplitude, name='+name+', Input='+file]
# -----------------------
# Bridge Parts Functions:
# -----------------------
def bridge_part(BridgeName,E,PRatio,BeamGeometry,FEInput, MatDef, LumpMass):
    """This function is used to construct a bridge part comprising of an approach slab, a suspended bridge, and an exit slab. 

    Args:
        BridgeName (string): name of bridge part to be constructed. 
        E (float): elastic modulus of beam elements in N/m^2. 
        PRatio (float): Poisson's Ratio.
        BeamGeometry ([float]): list of bridge geometry [Width, Height, BridgeLength, Lapproach, Lexit].
        FEInput ([float/int]): list of FE related inputs [number of elements, damage location, and crack ratio].
        MatDef (string): name of material to be defined.
        LumpMass (string): whether to include a lump mass definition in the beam section definition ('Yes' or 'No').

    Returns:
        dict: returns a dictionary composed of input commands to construct a suspended bridge part in abaqus input syntax, node IDs of interest, and bridge nodes. {'Lines','node_ID_of_interest','node_ID'}
    """
    Width = BeamGeometry[0]; Height = BeamGeometry[1]
    BridgeLength = BeamGeometry[2]; Lapproach = BeamGeometry[3]; Lexit = BeamGeometry[4]
    
    NumberOfElements = FEInput[0]; DamageLocation = FEInput[1]; CrackRatio = FEInput[2]
    ######################################################################################
    #------------------------------CALCULATIONS-------------------------------------------
    # ----------- Discretization of nodes:
    NodeID = [int(elem) for elem in np.linspace(1,NumberOfElements+3,NumberOfElements+3)]
    # --- Suspended bridge nodes:
    BridgeNodes = np.linspace(0,BridgeLength,NumberOfElements+1)

    # --- Bridge nodes of interest for result extraction:
    BridgeNodesOfInterest = np.arange(0.0,1,0.1)*BridgeLength

    # Find the indices the nodes of interest or closest value:
    IndicesOfInterest = find_closest_to(BridgeNodes,BridgeNodesOfInterest)

    # Substitute the nodes of interest to the main nodes to prevent adding extra elements:
    BridgeNodes[IndicesOfInterest] = BridgeNodesOfInterest

    # --- Damaged Nodes:
    DamageIndices_Save = [] # Empty list to save indices of damage locations for release_rotational_dof commands
    FirstNodeRotSpring = [] # Empty list to save indices of first node of rotational spring definitions
    SecondNodeRotSpring = [] # Empty list to save indices of second node of rotational spring definitions
    for i in DamageLocation:
        DamageIndices = find_closest_to(BridgeNodes,[i])
        DamageIndices_Save.append(DamageIndices[0])
        if i not in BridgeNodes:
            BridgeNodes[DamageIndices] = i
            BridgeNodes[DamageIndices[0]-1] = i*0.999
            BridgeNodes[DamageIndices[0]+1] = i*1.001
        else:
            BridgeNodes[DamageIndices[0]-1] = i*0.999
            BridgeNodes[DamageIndices[0]+1] = i*1.001
        FirstNodeRotSpring.append(DamageIndices[0]-1)
        SecondNodeRotSpring.append(DamageIndices[0]+1)
        
    # ----------- Concatenate into global coordinate system:
    # Concatenate approach and exit slab nodes to suspended bridge nodes in a coordinate system:
    NodesXPos = np.concatenate((np.array([-Lapproach]),BridgeNodes,np.array([BridgeLength+Lexit])),axis=0)
    NodesYPos = np.zeros(NumberOfElements+3)
    # create df to store bridge nodes and their x position:
    brg_nds_df = pd.DataFrame({
                                'node_ID':NodeID,
                                'x_pos':NodesXPos,
                                'y_pos':NodesYPos
    })
    # Store all nodes into list of strings:
    NodesString = []
    ind_ = 0 
    for i in NodeID:
        NodesString.append(str(i)+','+str(NodesXPos[ind_])+','+str(NodesYPos[ind_]))
        ind_+=1

    # ----------- Discretization of beam element:
    BeamID = [int(elem) for elem in np.linspace(1,NumberOfElements+2,NumberOfElements+2)]
    # First node of element:
    Beam1stElem = BeamID
    # Second node of element:
    Beam2ndElem = [elem+1 for elem in BeamID]

    # Append numbers into a list of strings
    BeamString = []
    ind_ = 0
    for i in BeamID:
        BeamString.append(str(i)+','+str(Beam1stElem[ind_])+','+str(Beam2ndElem[ind_]))
        ind_+=1

    ######################################################################################
    #------------------------------INPUT FILE SYNTAX--------------------------------------
    # ----------- Define beam part:
    Lines = ['** --------------- BRIDGE SECTION']
    Lines += ['*Part, name = '+BridgeName]
    # Node definition:
    Lines.append('*Node')
    Lines += NodesString
    # Beam definition:
    Lines.append('*Element, type = B23')
    Lines += BeamString

    # ----------- Define Sets:
    # Define nodesets for nodes of interest (every 0.1L)
    NodeIDofInterest = [elem+2 for elem in IndicesOfInterest]
    Lines += nset('NodesOfInterest',NodeIDofInterest[0:6],0)

    # Define node set for removing horizontal dof:
    Lines += nset('AllNodes', [1,NumberOfElements+3],1)
    
    # Define element set for beam elements altogether:
    Lines += elset('Set',[1,NumberOfElements+2],1)

    # Define surface on entire beam
    Lines += surf('BeamSurf','Element','Set')   

    # Define node sets for supports
    Lines += nset('VerticalSupport','1,2,'+str(NumberOfElements+2)+','+str(NumberOfElements+3),0) # Pin support
    
    # ----------- release_rotational_dof Elements Rotational DOF:
    # release_rotational_dof rotational dof for approach and exit slab to simulate a simply supported 
    # suspended bridge:
    # Approach Slab 
    Lines += release_rotational_dof('1','S1')
    Lines += release_rotational_dof('1','S2')
    Lines += release_rotational_dof('2','S1')

    # Exit Slab
    Lines += release_rotational_dof(str(NumberOfElements+1),'S2')
    Lines += release_rotational_dof(str(NumberOfElements+2),'S1')
    Lines += release_rotational_dof(str(NumberOfElements+2),'S2')

    # ---- Crack modeling
    ind_ = 0
    for i in DamageLocation:
        Damagerelease_rotational_dofLines1st = release_rotational_dof(str(DamageIndices_Save[ind_]+1),'S2')
        Damagerelease_rotational_dofLines2nd = release_rotational_dof(str(DamageIndices_Save[ind_]+2),'S1')
        Lines = Lines + Damagerelease_rotational_dofLines1st + Damagerelease_rotational_dofLines2nd
        ind_+=1
    
    ElementCount = 3
    ind_ = 0
    for i in DamageLocation:
        rotational_spring_stiffnessCrack = rotational_spring_stiffness(CrackRatio[ind_],E,Width,Height)
        ConnectedNodes = [str(FirstNodeRotSpring[ind_]+2),str(SecondNodeRotSpring[ind_]+2)]
        DamageRotSpringLines = spring_dashpot_element(NumberOfElements+ElementCount,'Crack'+str(ElementCount-2),rotational_spring_stiffnessCrack,0,ConnectedNodes,'6,6')
        Lines += DamageRotSpringLines['Lines']
        ind_ += 1; ElementCount += 1
    
    # --- Section Definition:
    Lines = Lines + beam_section('Set',MatDef,PRatio,Width,Height,0.0,LumpMass,'RECT')
    Lines = Lines + ['*End part']
    results_dict_ = {'Lines': Lines,'bridge_nodes_int':NodeIDofInterest[0:6], 'bridge_nodes': brg_nds_df}
    return results_dict_ 

def suspended_bridge_part(BridgeName,E,PRatio,BeamGeometry,FEInput, MatDef, LumpMass):
    """This function is used to construct only the suspended bridge part. 

    Args:
        BridgeName (string): name of bridge part to be constructed. 
        E (float): elastic modulus of beam elements in N/m^2. 
        PRatio (float): Poisson's Ratio.
        BeamGeometry ([float]): list of bridge geometry [Width, Height, BridgeLength, Lapproach, Lexit].
        FEInput ([float/int]): list of FE related inputs [number of elements, damage location, and crack ratio].
        MatDef (string): name of material to be defined.
        LumpMass (string): whether to include a lump mass definition in the beam section definition ('Yes' or 'No').

    Returns:
        dict: returns a dictionary composed of input commands to construct a suspended bridge part in abaqus input syntax, node IDs of interest, and bridge nodes. {'Lines','node_ID_of_interest','node_ID'}
    """
    Width = BeamGeometry[0]; Height = BeamGeometry[1]
    BridgeLength = BeamGeometry[2]
    
    NumberOfElements = FEInput[0]; DamageLocation = FEInput[1]; CrackRatio = FEInput[2]
    ######################################################################################
    #------------------------------CALCULATIONS-------------------------------------------
    # ----------- Discretization of nodes:
    NodeID = [int(elem) for elem in np.linspace(1,NumberOfElements+1,NumberOfElements+1)]
    # --- Suspended bridge nodes:
    BridgeNodes = np.linspace(0,BridgeLength,NumberOfElements+1)

    # --- Bridge nodes of interest for result extraction:
    BridgeNodesOfInterest = np.arange(0.1,1,0.1)*BridgeLength

    # Find the indices the nodes of interest or closest value:
    IndicesOfInterest = find_closest_to(BridgeNodes,BridgeNodesOfInterest)

    # Substitute the nodes of interest to the main nodes to prevent adding extra elements:
    BridgeNodes[IndicesOfInterest] = BridgeNodesOfInterest

    # --- Damaged Nodes:
    DamageIndices_Save = [] # Empty list to save indices of damage locations for release_rotational_dof commands
    FirstNodeRotSpring = [] # Empty list to save indices of first node of rotational spring definitions
    SecondNodeRotSpring = [] # Empty list to save indices of second node of rotational spring definitions
    for i in DamageLocation:
        DamageIndices = find_closest_to(BridgeNodes,[i])
        DamageIndices_Save.append(DamageIndices[0])
        if i not in BridgeNodes:
            BridgeNodes[DamageIndices] = i
            BridgeNodes[DamageIndices[0]-1] = i*0.999
            BridgeNodes[DamageIndices[0]+1] = i*1.001
        else:
            BridgeNodes[DamageIndices[0]-1] = i*0.999
            BridgeNodes[DamageIndices[0]+1] = i*1.001
        FirstNodeRotSpring.append(DamageIndices[0]-1)
        SecondNodeRotSpring.append(DamageIndices[0]+1)
        
    # ----------- Concatenate into global coordinate system:
    # Concatenate approach and exit slab nodes to suspended bridge nodes in a coordinate system:
    NodesXPos = BridgeNodes
    NodesYPos = np.zeros(NumberOfElements+1)
    # Store all nodes into list of strings:
    NodesString = []
    ind_ = 0 
    for i in NodeID:
        NodesString.append(str(i)+','+str(NodesXPos[ind_])+','+str(NodesYPos[ind_]))
        ind_+=1
    # ----------- Discretization of beam element:
    BeamID = [int(elem) for elem in np.linspace(1,NumberOfElements,NumberOfElements)]
    # First node of element:
    Beam1stElem = BeamID
    # Second node of element:
    Beam2ndElem = [elem+1 for elem in BeamID]
    # Append numbers into a list of strings
    BeamString = []
    ind_ = 0
    for i in BeamID:
        BeamString.append(str(i)+','+str(Beam1stElem[ind_])+','+str(Beam2ndElem[ind_]))
        ind_+=1
    ######################################################################################
    #------------------------------INPUT FILE SYNTAX--------------------------------------
    # ----------- Define beam part:
    Lines = ['** --------------- BRIDGE SECTION']
    Lines += ['*Part, name = '+BridgeName]
    # Node definition:
    Lines.append('*Node')
    Lines += NodesString
    # Beam definition:
    Lines.append('*Element, type = B23')
    Lines = Lines + BeamString

    # ----------- Define Sets:
    # Define nodesets for nodes of interest (every 0.1L)
    NodeIDofInterest = [elem+1 for elem in IndicesOfInterest]
    NodeIDofInterest_Str = str(NodeIDofInterest[0:5])[1:-1]
    Lines += nset('NodesOfInterest',NodeIDofInterest_Str,0)

    # Define node set for removing horizontal dof:
    Lines += nset('AllNodes', '1,'+str(NumberOfElements+1)+',1',1)
    
    # Define element set for beam elements altogether:
    Lines += elset('Set','1,'+str(NumberOfElements)+',1',1)

    # Define surface on entire beam
    Lines += surf('BeamSurf','Element','Set')   

    # Define node sets for supports
    Lines += nset('VerticalSupport','1,'+str(NumberOfElements+1),0) # Pin support
    
    # ----------- release_rotational_dof Elements Rotational DOF:
    # ---- Crack modeling
    ind_ = 0
    for i in DamageLocation:
        Lines += release_rotational_dof(str(DamageIndices_Save[ind_]),'S2') + release_rotational_dof(str(DamageIndices_Save[ind_]+1),'S1')
        ind_+=1
    
    ElementCount = 1
    ind_ = 0
    for i in DamageLocation:
        rotational_spring_stiffnessCrack = rotational_spring_stiffness(CrackRatio[ind_],E,Width,Height)
        ConnectedNodes = [str(FirstNodeRotSpring[ind_]+1),str(SecondNodeRotSpring[ind_]+1)]
        RotSpring = spring_dashpot_element(NumberOfElements+ElementCount,'Crack'+str(ElementCount),rotational_spring_stiffnessCrack,0,ConnectedNodes,'6,6')
        Lines += RotSpring['Lines']
        ind_ += 1; ElementCount += 1
    
    # --- Section Definition:
    Lines += beam_section('Set',MatDef,PRatio,Width,Height,0.0,LumpMass,'RECT')
    Lines += ['*End part']
    results_dict_ = {'Lines': Lines,'node_ID_of_interest':NodeIDofInterest_Str, 'node_ID':NodeID}
    return results_dict_

def bridge_frequency(E,BeamMass,PRatio,BeamGeometry,FEInput):
    """This function is used to model a suspended bridge and perform an eigenfreq analysis for the bridge.

    Args:
        E (float): elastic modulus of beam element in N/m^2.
        BeamMass (float): distributed mass across beam element in kg/m. 
        PRatio (float): Poisson's Ratio. 
        BeamGeometry ([float]): list of beam cross section [Width, Height].
        FEInput ([float/int]): list of FE related inputs [number of elements, damage location, and crack ratio].
    
    
    """
    # Inputs:
    Width = BeamGeometry[0]; Height = BeamGeometry[1]

    # Name of parts:
    PartNames = ['bridge_part']
    MatName = ['BridgeMat']
    ######################################################################################
    #-----------------------------------HEADER--------------------------------------------
    v = [0,0,0,0]
    HeaderLines = input_header('JobTest',PartNames[0],v)
    FinalLines = HeaderLines

    ######################################################################################
    #-----------------------------------PARTS---------------------------------------------
    # ----- Bridge part:
    FinalLines += ['** PARTS'] + suspended_bridge_part(PartNames[0],E,PRatio,BeamGeometry,FEInput, MatName[0], 'No')

    ######################################################################################
    #-----------------------------------ASSEMBLY------------------------------------------
    # ----- Assemble parts:
    FinalLines += assemble_parts(PartNames)

    # ----- End Assembly:
    FinalLines += ['*End assembly']

    ######################################################################################
    #-----------------------------------MATERIAL DEFINITION-------------------------------
    # ----- Bridge definition:
    FinalLines += mat_definition(MatName[0],E,PRatio,BeamMass/(Width*Height),[])

    ######################################################################################
    #-----------------------------------SUPPORT BOUNDARY CONDITIONS-----------------------
    # ----- Apply BC on bridge:
    # Assign Pin:
    FinalLines += boundary_condition(PartNames[0],'VerticalSupport','Support',[],[1,2],'MOD')

    # Assign Roller:
    FinalLines += boundary_condition(PartNames[0],'AllNodes','Support',[],[1],'MOD')

    ######################################################################################
    #-----------------------------------step DEFINITION-----------------------------------
    # ----- Define implicit dynamic step:
    FinalLines += step('Frequency',0,0,0.0,'Freqstep',5)
    
    # ----- Output requests
    FinalLines += ['*Output, field, variable=PRESELECT']
    # End of step
    FinalLines += ['*End step']

    # ----- Save input file in separate text file:
    np.savetxt('bridge_frequencyuency.inp',FinalLines,fmt="%s")

if __name__ == '__main__':
    BeamMass = 11600 
    Height = 1.3140   
    Width = 0.3285 
    Lapproach = 40.0
    BridgeLength = 16.0  
    Lexit = 8.0
    
    # Signal inputs:
    time_step = 0.001 # time step in analysis in sec
    frequency_bin_width = 0.5 # frequency bin width
    frequency_cut_off = 50

    # Damage inputs
    DamageLocation = []
    CrackRatio = []
    surface_roughness_bool = 1 # boolean to include surface (0 = No and 1 = Yes)
    road_class = 'A' # ISO road class (A,B,C)
    NumberOfElements = 100
    
    BeamGeometry = [Width,Height,BridgeLength,Lapproach,Lexit]
    FEInput = [NumberOfElements,DamageLocation,CrackRatio]
    
    bridge_part_lines = bridge_part('BridgePart',2e11,0.3,BeamGeometry,FEInput, 'bridge_mat', 'No')
    pdb.set_trace()
    True