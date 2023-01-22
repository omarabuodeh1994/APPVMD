from python_functions import*
from input_file_commands import*



def passenger_vehicle_part(vehicle_name,initial_position):
    """This function is used to create a vehicle part with two axles.

    Args:
        vehicle_name (string): name of vehicle part.
        initial_position (float): initial position of vehicle from the rear.

    Returns:
        [string]: list of Abaqus input commands to create a vehicle part.
    """
    # Vehicle properties:
    length_rear,length_center,length_front = 3.23768130e+00, 2.71543854e-02, 2.73516432e+00
    radius_rear, radius_center, radius_front = 1.56985325,8.12115995,2.42906296e-01
    mass_rear,mass_center,mass_front = 2.18791973e+03,9.88665491e+03,3.29425362e+02
    mass_rear_axle,mass_front_axle = 725.4,725.4
    k_sprung = [1969034.0,727812.0] # sprung stiffness [rear,front]
    c_sprung = [7182.0,2190.0] # sprung damping [rear,front]
    k_unsprung = [4735000.0,1972900.0] # unsprung stiffness [rear,front]
    axle_position = [0.0,6.0] # axle initial position [rear,front]
    vehicle_CM_x,vehicle_CM_y = 3.0,2.0
    # Calculation:
    Lv = sum([length_rear,length_center,length_front])
    XMinimizationCummulative = cumulative([length_rear,length_center,length_front])
    x_minimization_cummulative_transformed = [i+initial_position for i in XMinimizationCummulative]
    XNode = np.round(np.linspace(initial_position,Lv+initial_position,7),2)
    ####################### Nodes ####################### 
    Lines = ['*Part, name = '+vehicle_name]
    Lines += ['*Node']
    # Vehicle Body:
    axle_position_transformed = [axle_position[0]+initial_position,round(axle_position[1]+initial_position,2)]
    LinesBeamNode = vehicle_line_node(XNode,vehicle_CM_y,vehicle_CM_x+initial_position,1,x_minimization_cummulative_transformed,axle_position_transformed)
    Lines += LinesBeamNode['Lines']
    BeamCM = LinesBeamNode['node_CM']
    SelectedElementID = LinesBeamNode['selected_element_ID']
    # Rear Axle:
    LinesRearAxleNodes = plane_stress_node(LinesBeamNode['node_IDs'][-1]+1,0.0+initial_position,1.0)
    Lines += LinesRearAxleNodes['Lines']
    RearAxleCM = LinesRearAxleNodes['node_CM']
    # Front Axle
    LinesFrontAxleNodes = plane_stress_node(LinesRearAxleNodes['node_CM']+1,6.0+initial_position,1.0)
    Lines += LinesFrontAxleNodes['Lines']
    FrontAxleCM = LinesFrontAxleNodes['node_CM']
    # Rear Surface: 
    LinesRearSurfaceNode = node_def(LinesFrontAxleNodes['node_CM']+1,[0.0+initial_position,0.0])
    Lines += LinesRearSurfaceNode['Lines']
    # Front Surface: 
    LinesFrontSurfaceNode = node_def(LinesRearSurfaceNode['node_ID']+1,[6.0+initial_position,0.0])
    Lines += LinesFrontSurfaceNode['Lines']
    ####################### Element #######################
    # Beam
    Lines += ['*Element, type = B21']
    LinesBodyElem = lines_element(1,LinesBeamNode['node_IDs'])
    Lines += LinesBodyElem['Lines']
    # Plane stress
    Lines += ['*Element, type = CPS4R']
    LinesRearAxleElement = plane_stress_element(LinesBodyElem['element_IDs'][-1]+1,[LinesRearAxleNodes['corner_node_ID'],LinesFrontAxleNodes['corner_node_ID']])
    Lines += LinesRearAxleElement['Lines']
    ####################### Node set #######################
    Lines += nset('RearAxleCM',[LinesRearAxleNodes['node_CM']],0) # Rear axle CM
    Lines += nset('FrontAxleCM',[LinesFrontAxleNodes['node_CM']],0) # Front axle CM
    Lines += nset('BeamTie', LinesBeamNode['node_axle_IDs'], 0) # Beam tie nodes for rigid body definition
    Lines += nset('BeamCM', [BeamCM],0) # Beam CM
    Lines += nset('RearSurface', [LinesRearSurfaceNode['node_ID']],0) # rear surface point
    Lines += nset('FrontSurface', [LinesFrontSurfaceNode['node_ID']],0) # front surface point
    Lines += nset('VehicleCMs', 'BeamCM,RearAxleCM,FrontAxleCM,RearSurface,FrontSurface',0) # CMs for all vehicle node sets into one
    Lines += nset('VehicleCMsForRotation', 'RearAxleCM,FrontAxleCM,RearSurface,FrontSurface',0) # CM to restrain rotational DOF
    ####################### Element set #######################
    Lines += elset('RearBeam',list(SelectedElementID[0]),0)
    Lines += elset('CenterBeam',list(SelectedElementID[1]),0)
    Lines += elset('FrontBeam',list(SelectedElementID[2]),0)
    Lines += elset('EntireBeam',[LinesBodyElem['element_IDs'][0],LinesBodyElem['element_IDs'][-1]],1)
    Lines += elset('RearAxle',[LinesRearAxleElement['element_IDs'][0]],0)
    Lines += elset('FrontAxle',[LinesRearAxleElement['element_IDs'][1]],0)
    Lines += elset('EntireVehicle','EntireBeam,RearAxle,FrontAxle',0)
    ####################### Surface definition #######################
    Lines += surf('tire_surface','Node',[LinesRearSurfaceNode['node_ID'],LinesFrontSurfaceNode['node_ID']])
    ####################### Spring definition #######################
    # Sprung
    lines_rear_sprunp_element = spring_dashpot_element(LinesRearAxleElement['element_IDs'][-1]+1,'RearSprung',k_sprung[0],c_sprung[0],[str(LinesBeamNode['node_axle_IDs'][0]),str(LinesRearAxleNodes['node_CM'])],'2,2')
    Lines += lines_rear_sprunp_element['Lines']
    lines_front_sprunp_element = spring_dashpot_element(lines_rear_sprunp_element['element_ID']+1,'FrontSprung',k_sprung[1],c_sprung[1],[str(LinesBeamNode['node_axle_IDs'][1]),str(LinesFrontAxleNodes['node_CM'])],'2,2')
    Lines += lines_front_sprunp_element['Lines']
    # Unsprung
    lines_rear_unsprunp_element = spring_dashpot_element(lines_front_sprunp_element['element_ID']+1,'RearUnsprung',k_unsprung[0],0.0,[str(LinesRearAxleNodes['node_CM']),str(LinesRearSurfaceNode['node_ID'])],'2,2')
    Lines += lines_rear_unsprunp_element['Lines']
    lines_front_unsprunp_element = spring_dashpot_element(lines_rear_unsprunp_element['element_ID']+1,'FrontUnprung',k_unsprung[1],0.0,[str(LinesFrontAxleNodes['node_CM']),str(LinesFrontSurfaceNode['node_ID'])],'2,2')
    Lines += lines_front_unsprunp_element['Lines']
    ####################### Section Definition #######################
    Lines += beam_section('RearBeam',vehicle_name+'RearBeam',0.2,0,0,radius_rear,'Yes','CIRC') # Rear element
    Lines += beam_section('CenterBeam',vehicle_name+'CenterBeam',0.2,0,0,radius_center,'Yes','CIRC') # Rear element
    Lines += beam_section('FrontBeam',vehicle_name+'FrontBeam',0.2,0,0,radius_front,'Yes','CIRC') # Rear element
    Lines += solid_section('RearAxle',vehicle_name+'RearAxle',0.1)
    Lines += solid_section('FrontAxle',vehicle_name+'FrontAxle',0.1)
    Lines += ['*End part']
    
    beam_lengths = [length_rear,length_center,length_front]
    radius_sizes = [radius_rear, radius_center, radius_front]
    masses_beam = [mass_rear,mass_center,mass_front]
    mass_axles = [mass_rear_axle,mass_front_axle]
    CM_IDs = [BeamCM,RearAxleCM,FrontAxleCM]
    results_dict_ = {'Lines':Lines,'beam_lengths':beam_lengths,'radius':radius_sizes,'beam_masses':masses_beam,'axle_masses':mass_axles,'unsprung_stiffness':k_unsprung,
                    'sprung_damping':c_sprung,'vehicle_length':Lv,'CM_IDs':CM_IDs,'axle_position':axle_position}
    return results_dict_ # Lines,beam_lengths,radius_sizes,masses_beam,mass_axles,k_unsprung,c_sprung,Lv,CM_IDs,axle_position

def truck_vehicle_part(vehicle_name,initial_position):
    """This function is used to create a vehicle part with three axles.

    Args:
        vehicle_name (string): name of vehicle part.
        initial_position (float): initial position of vehicle from the rear.

    Returns:
        [string]: list of Abaqus input commands to create a vehicle part.
    """
    # Vehicle properties:
    length_rear,length_center,length_front = 3.84496667e+00, 2.54558684e-01, 3.70047465e+00
    radius_rear, radius_center, radius_front = 1.75220610e-01,5.21386062e+00,1.00000458e-02
    mass_rear,mass_center,mass_front = 9.82620663e+03,1.72603802e+04,1.34131245e+01
    mass_rear_axle,mass_center_axle, mass_front_axle = 1100.0,1100.0,700.0
    k_sprung = [1.0e6,1.0e6,4e05]#[rear,center,front]
    c_sprung = [20e03,20e03,10e03]#[rear,center,front]
    k_unsprung = [3.5e06,3.5e06,1.75e06] #[rear,center,front]
    axle_position = [0.0,1.8,7.8]
    vehicle_CM_x,vehicle_CM_y = 3.23,2.0
    # Calculation:
    Lv = sum([length_rear,length_center,length_front])
    X_minimization_cummulative = cumulative([length_rear,length_center,length_front])
    X_minimization_cummulative_transformed = [i+initial_position for i in X_minimization_cummulative]
    XNode = np.round(np.linspace(initial_position,initial_position+Lv,7),2)
    ####################### Nodes ####################### 
    Lines = ['*Part, name = '+vehicle_name]
    Lines += ['*Node']
    #---- Vehicle Body:
    # Define transformed axle position based on initial position of vehicle
    axle_position_transformed = [axle_position[0]+initial_position,axle_position[1]+initial_position,round(axle_position[2]+initial_position,2)] 
    lines_beam_nodes = vehicle_line_node(XNode,vehicle_CM_y,vehicle_CM_x+initial_position,1,X_minimization_cummulative_transformed,axle_position_transformed)
    Lines += lines_beam_nodes['Lines']
    BeamCM = lines_beam_nodes['node_CM']
    SelectedElementID = lines_beam_nodes['selected_element_ID']
    #---- Axles:
    # Rear Axle:
    lines_rear_axle_nodes = plane_stress_node(lines_beam_nodes['node_IDs'][-1]+1,axle_position_transformed[0],1.0)
    Lines += lines_rear_axle_nodes['Lines']
    RearAxleCM = lines_rear_axle_nodes['node_CM']
    # Center Axle:
    lines_center_axle_nodes = plane_stress_node(lines_rear_axle_nodes['node_CM']+1,axle_position_transformed[1],1.0)
    Lines += lines_center_axle_nodes['Lines']
    CenterAxleCM = lines_center_axle_nodes['node_CM']
    # Front Axle
    lines_front_axle_nodes = plane_stress_node(lines_center_axle_nodes['node_CM']+1,axle_position_transformed[2],1.0)
    Lines += lines_front_axle_nodes['Lines']
    FrontAxleCM = lines_front_axle_nodes['node_CM']
    #---- Surface Nodes:
    # Rear Surface: 
    lines_rear_surface_node = node_def(lines_front_axle_nodes['node_CM']+1,[axle_position_transformed[0],0.0])
    Lines += lines_rear_surface_node['Lines']
    # Center Surface: 
    lines_center_surface_node = node_def(lines_rear_surface_node['node_ID']+1,[axle_position_transformed[1],0.0])
    Lines += lines_center_surface_node['Lines']
    # Front Surface: 
    lines_front_surface_node = node_def(lines_center_surface_node['node_ID']+1,[axle_position_transformed[2],0.0])
    Lines += lines_front_surface_node['Lines']
    ####################### Element #######################
    # Beam
    Lines += ['*Element, type = B21']
    LinesBodyElem = lines_element(1,lines_beam_nodes['node_IDs'])
    Lines += LinesBodyElem['Lines']
    # Plane stress
    Lines += ['*Element, type = CPS4R']
    corner_nodes = [lines_rear_axle_nodes['corner_node_ID'],lines_center_axle_nodes['corner_node_ID'],lines_front_axle_nodes['corner_node_ID']]
    lines_axle_elements = plane_stress_element(LinesBodyElem['element_IDs'][-1]+1,corner_nodes)
    Lines += lines_axle_elements['Lines']
    ####################### Node set #######################
    Lines += nset('RearAxleCM',[lines_rear_axle_nodes['node_CM']],0) # Rear axle CM
    Lines += nset('CenterAxleCM',[lines_center_axle_nodes['node_CM']],0) # Center axle CM
    Lines += nset('FrontAxleCM',[lines_front_axle_nodes['node_CM']],0) # Front axle CM
    Lines += nset('BeamTie', lines_beam_nodes['node_axle_IDs'], 0) # Beam tie nodes for rigid body definition
    Lines += nset('BeamCM', [lines_beam_nodes['node_CM']],0) # Beam CM
    Lines += nset('RearSurface', [lines_rear_surface_node['node_ID']],0) # rear surface point
    Lines += nset('CenterSurface', [lines_center_surface_node['node_ID']],0) # center surface point
    Lines += nset('FrontSurface', [lines_front_surface_node['node_ID']],0) # front surface point
    Lines += nset('VehicleCMs','BeamCM,RearAxleCM,CenterAxleCM,FrontAxleCM,RearSurface,CenterSurface,FrontSurface',0) # CMs for all vehicle node sets into one
    Lines += nset('VehicleCMsForRotation','RearAxleCM,CenterAxleCM,FrontAxleCM,RearSurface,CenterSurface,FrontSurface',0) # CM to restrain rotational DOF
    ####################### Element set #######################
    Lines += elset('RearBeam',list(SelectedElementID[0]),0)
    Lines += elset('CenterBeam',list(SelectedElementID[1]),0)
    Lines += elset('FrontBeam',list(SelectedElementID[2]),0)
    Lines += elset('EntireBeam',[LinesBodyElem['element_IDs'][0],LinesBodyElem['element_IDs'][-1]],1)
    Lines += elset('RearAxle',[lines_axle_elements['element_IDs'][0]],0)
    Lines += elset('CenterAxle',[lines_axle_elements['element_IDs'][1]],0)
    Lines += elset('FrontAxle',[lines_axle_elements['element_IDs'][2]],0)
    Lines += elset('EntireVehicle','EntireBeam,RearAxle,CenterAxle,FrontAxle',0)
    ####################### Surface definition #######################
    Lines += surf('tire_surface','Node',[lines_rear_surface_node['node_ID'],lines_center_surface_node['node_ID'],lines_front_surface_node['node_ID']])
    ####################### Spring definition #######################
    # Sprung
    lines_rear_sprunp_element = spring_dashpot_element(lines_axle_elements['element_IDs'][-1]+1,'RearSprung',k_sprung[0],c_sprung[0],[str(lines_beam_nodes['node_axle_IDs'][0]),str(lines_rear_axle_nodes['node_CM'])],'2,2')
    Lines += lines_rear_sprunp_element['Lines']
    lines_center_sprunp_element = spring_dashpot_element(lines_rear_sprunp_element['element_ID']+1,'CenterSprung',k_sprung[1],c_sprung[1],[str(lines_beam_nodes['node_axle_IDs'][1]),str(lines_center_axle_nodes['node_CM'])],'2,2')
    Lines += lines_center_sprunp_element['Lines']
    lines_front_sprunp_element = spring_dashpot_element(lines_center_sprunp_element['element_ID']+1,'FrontSprung',k_sprung[2],c_sprung[2],[str(lines_beam_nodes['node_axle_IDs'][2]),str(lines_front_axle_nodes['node_CM'])],'2,2')
    Lines += lines_front_sprunp_element['Lines']
    # Unsprung
    lines_rear_unsprunp_element = spring_dashpot_element(lines_front_sprunp_element['element_ID']+1,'RearUnsprung',k_unsprung[0],0.0,[str(lines_rear_axle_nodes['node_CM']),str(lines_rear_surface_node['node_ID'])],'2,2')
    Lines += lines_rear_unsprunp_element['Lines']
    lines_center_unsprunp_element = spring_dashpot_element(lines_rear_unsprunp_element['element_ID']+1,'CenterUnsprung',k_unsprung[1],0.0,[str(lines_center_axle_nodes['node_CM']),str(lines_center_surface_node['node_ID'])],'2,2')
    Lines += lines_center_unsprunp_element['Lines']
    lines_front_unsprunp_element = spring_dashpot_element(lines_center_unsprunp_element['element_ID']+1,'FrontUnprung',k_unsprung[2],0.0,[str(lines_front_axle_nodes['node_CM']),str(lines_front_surface_node['node_ID'])],'2,2')
    Lines += lines_front_unsprunp_element['Lines']
    ####################### Section Definition #######################
    Lines += beam_section('RearBeam',vehicle_name+'RearBeam',0.2,0,0,radius_rear,'Yes','CIRC') # Rear element
    Lines += beam_section('CenterBeam',vehicle_name+'CenterBeam',0.2,0,0,radius_center,'Yes','CIRC') # Rear element
    Lines += beam_section('FrontBeam',vehicle_name+'FrontBeam',0.2,0,0,radius_front,'Yes','CIRC') # Rear element
    Lines += solid_section('RearAxle',vehicle_name+'RearAxle',0.1)
    Lines += solid_section('CenterAxle',vehicle_name+'CenterAxle',0.1)
    Lines += solid_section('FrontAxle',vehicle_name+'FrontAxle',0.1)
    Lines += ['*End part']
    
    beam_lengths = [length_rear,length_center,length_front]
    radius_sizes = [radius_rear, radius_center, radius_front]
    masses_beam = [mass_rear,mass_center,mass_front]
    mass_axles = [mass_rear_axle,mass_center_axle,mass_front_axle]
    CM_IDs = [BeamCM,RearAxleCM,CenterAxleCM,FrontAxleCM]
    results_dict_ = {'Lines':Lines,'beam_lengths':beam_lengths,'radius':radius_sizes,'beam_masses':masses_beam,'axle_masses':mass_axles,'unsprung_stiffness':k_unsprung,
                    'sprung_damping':c_sprung,'vehicle_length':Lv,'CM_IDs':CM_IDs,'axle_position':axle_position}
    return results_dict_ # Lines,beam_lengths,radius_sizes,masses_beam,mass_axles,k_unsprung,c_sprung,Lv,CM_IDs,axle_position

def sprung_mass(XPos,YPos,PartName,K,C, MatDef, Thickness):
    """This function is used to construct a sprung mass part within an input file.

    Args:
        XPos (float): positions of sprung mass center of mass in x-direction.
        YPos (float): positions of sprung mass center of mass in y-direction.
        PartName (string): name of the part being defined. 
        K (float): spring stiffness in N/m.
        C (float): dashpot damping in N/s.
        MatDef (string): name of material of sprung mass.
        Thickness (float): thickness of sprung mass.

    Returns:
        string: sprung mass part command in abaqus input syntax.
    """
    Lines = ['** --------------- SPRUNG MASS SECTION']
    Lines.append('*Part, name = '+PartName)
    # ------------------------------ Sprung Mass Part:
    Lines.append('*Node')
    # Rectangular Nodes:
    LinesSprungMass = plane_stress_node(1,XPos,YPos)
    Lines += LinesSprungMass['Lines']
    # Surface Nodes 
    LinesSurfaceNode = node_def(LinesSprungMass['node_CM']+1,[XPos,0.0])
    Lines += LinesSurfaceNode['Lines']
    
    # Element Definition:
    Lines += ['*Element, type = CPS4R']
    LinesRearAxleElement = plane_stress_element(1,[[1,2,3,4]])
    Lines += LinesRearAxleElement['Lines']
    
    # Node Set Definition For:
    # (1) Tire Node:
    Lines += nset('TireNode',[LinesSurfaceNode['node_ID']],0)
    # (2) Center of mass:
    Lines += nset('CM',[LinesSprungMass['node_CM']],0)
    
    # (3) CMs of vehicle for BC application:
    Lines += nset('VehicleCMs','CM,TireNode',0)
        
    # Element Set Definition:
    
    Lines += elset('Set',[1],0)
    
    # Surface definition
    Lines += surf('surf','Node',['TireNode'])

    # --------------------- SPRING DEFINITION
    # Define spring element:
    LinesSpring = spring_dashpot_element(2,'Sprung',K,C,[str(LinesSprungMass['node_CM']),str(LinesSurfaceNode['node_ID'])],'2,2')
    Lines += LinesSpring['Lines']
    
    # Define solid section
    Lines.append('*Solid Section, elset=Set, material='+MatDef)
    Lines.append(str(Thickness))
    Lines.append('*End part')
    return Lines

def sprung_unsprung_mass(XPos,YPos,PartName,K,C, MatDef, Thickness):
    """This function is used to construct a sprung mass part within an input file.

    Args:
        XPos (float): center of sprung and unsprung masses, respectively, in x-direction; both masses should have the same center in the x component.
        YPos ([float]): centers of sprung and unsprung masses, respectively, center in y-direction.
        PartName (string): name of the part being defined. 
        K ([float]): spring stiffness for sprung and unsprung masses, respectively, in N/m [sprung_stiffness,unsprung_stiffness].
        C ([float]): dashpot damping for sprung and unsprung masses, respectively, in N/s [sprung_dashpot,unsprung_dashpot].
        MatDef ([str]): names of material definitions for sprung and unsprung mass, repectively.
        Thickness (float): thickness of sprung mass.

    Returns:
        string: sprung mass part command in abaqus input syntax.
    """
    Lines = ['** --------------- SPRUNG MASS SECTION']
    Lines.append('*Part, name = '+PartName)
    # ------------------------------ Sprung Mass Part:
    Lines.append('*Node')
    # Sprung Mass Nodes:
    LinesSprungMass = plane_stress_node(1,XPos,YPos[0])
    Lines += LinesSprungMass['Lines']
    # Unsprung Mass Nodes:
    lines_unsprung_mass = plane_stress_node(LinesSprungMass['node_CM']+1,XPos,YPos[1])
    Lines += lines_unsprung_mass['Lines']
    # Surface Nodes 
    LinesSurfaceNode = node_def(lines_unsprung_mass['node_CM']+1,[XPos,0.0])
    Lines += LinesSurfaceNode['Lines']
    
    # Element Definition:
    Lines += ['*Element, type = CPS4R']
    # Sprung mass:
    lines_mass_elements = plane_stress_element(1,[LinesSprungMass['corner_node_ID'],lines_unsprung_mass['corner_node_ID']])
    Lines += lines_mass_elements['Lines']
    
    # Node Set Definition For:
    # (1) Tire Node:
    Lines += nset('TireNode',[LinesSurfaceNode['node_ID']],0)
    # (2) Center of mass of sprung mass:
    Lines += nset('SprungMassCM',[LinesSprungMass['node_CM']],0)
    # (3) Center of mass of unsprung mass:
    Lines += nset('UnsprungMassCM',[lines_unsprung_mass['node_CM']],0)
    
    # (4) CMs of vehicle for BC application:
    Lines += nset('VehicleCMs','SprungMassCM,UnsprungMassCM,TireNode',0)
        
    # Element Set Definition:
    # Sprung mass element set:
    Lines += elset('SprungMassSet',[1],0)
    # Unsprung mass element set:
    Lines += elset('UnsprungMassSet',[2],0)
    # Vehicle bodies:
    Lines += elset('Bodies',[1,2],0)
    # Surface definition
    Lines += surf('surf','Node',['TireNode'])

    # --------------------- SPRING DEFINITION
    # Define sprung spring element:
    lines_sprung_spring = spring_dashpot_element(lines_mass_elements['element_IDs'][-1]+1,'Sprung',K[0],C[0],[str(LinesSprungMass['node_CM']),str(lines_unsprung_mass['node_CM'])],'2,2')
    Lines += lines_sprung_spring['Lines']
    # Define unsprung spring element:
    lines_unsprung_spring = spring_dashpot_element(lines_sprung_spring['element_ID']+1,'Unsprung',K[1],C[1],[str(lines_unsprung_mass['node_CM']),str(LinesSurfaceNode['node_ID'])],'2,2')
    Lines += lines_unsprung_spring['Lines']
    
    # Define solid section
    Lines += ['*Solid Section, elset=SprungMassSet, material='+MatDef[0]]
    Lines += [str(Thickness)]
    Lines += ['*Solid Section, elset=UnsprungMassSet, material='+MatDef[1]]
    Lines += [str(Thickness)]
    Lines.append('*End part')
    return Lines

if __name__ == '__main__':

    Lines = sprung_unsprung_mass(0.0,[2.0,1.0],'sprung_unsprung_part',[500.0,1000.0],[10.0,0.0], ['SprungMat','UnsprungMat'], 0.1)

    import numpy as np 

    np.savetxt('vehicle.inp',Lines,"%s")