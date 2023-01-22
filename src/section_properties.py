import matplotlib.pyplot as plt
from math import sqrt, pi, ceil
import numpy as np
import pandas as pd
def rectangle_area(section_width,section_height):
    """This function is used to compute the area of a rectangular section.

    Args:
        section_width (float): width of rectangle.
        section_height (float): height of rectangle.

    Returns:
        float: area of rectangle
    """
    return section_width * section_height

def modular_ratio(E_interest,E_other):
    """This function is used to compute the modular ratio of a composite section.

    Args:
        E_interest (float): Young's Modulus of part to be transformed.
        E_other (float): Young's Modulus of other part.

    Returns:
        float: modular ratio of two materials.
    """
    return E_interest/E_other

def compute_rectangle_section_MOI(section_geometry):
    """This function is used to compute the moment of inertia of a rectangular section.

    Args:
        section_geometry ([float]): list of height and width [width,height].

    Returns:
        float: the moment of inertia of a rectangular section.
    """
    height,width = section_geometry[1],section_geometry[0]
    return 1/12.0*width*height**3.0

def compute_i_section_MOI (bottom_flange,web,top_flange):
    """This function is used to compute the position of the center of gravity from the bottom of an I-section.

    Args:
        bottom_flange ([float]): list of width and thickness of flange, order doesn't matter.
        web ([float]): list of height and thickness of web, order doesn't matter.  
        top_flange ([float]): list of width and thickness of flange, order doesn't matter.

    Returns:
        dict: dictionary comprising of the vertical position of total area, center gravity, and moment of inertia of section {'section_area','y_bar','Iz'}
    """
    # Given:
    # Dimensions of bottom flange:
    bottom_flange_width,bottom_flange_thick = max(bottom_flange),min(bottom_flange)
    # Dimensions of web:
    web_height,web_thick = max(web),min(web)
    # Dimensions of top flange:
    top_flange_width,top_flange_thick = max(top_flange),min(top_flange)

    # Cross sectional area:
    # Bottom flange:
    area_bottom_flange = rectangle_area(bottom_flange_width,bottom_flange_thick)
    # Web:
    area_web = rectangle_area(web_height,web_thick)
    # Top flange:
    area_top_flange = rectangle_area(top_flange_width,top_flange_thick)

    # Distance from center of I-beam part to the datum (lowest point)
    # Bottom flange:
    y_cg_bottom_flange = bottom_flange_thick/2.0
    # Web:
    y_cg_web = bottom_flange_thick+web_height/2.0
    # Top flange:
    y_cg_top_flange = bottom_flange_thick + web_height + top_flange_thick/2.0

    # Product of each area with corresponding center of gravity:
    # Bottom flange:
    area_y_cg_bottom_flange = area_bottom_flange*y_cg_bottom_flange
    # Bottom flange:
    area_y_cg_web = area_web*y_cg_web
    # Bottom flange:
    area_y_cg_top_flange = area_top_flange*y_cg_top_flange

    # Compute center of gravity of entire section:
    total_area = area_bottom_flange + area_web + area_top_flange
    total_area_product_y_cg = area_y_cg_bottom_flange + area_y_cg_web + area_y_cg_top_flange
    y_cg = total_area_product_y_cg/total_area

    # Moment of inertia for each part:
    # Bottom flange:
    bottom_flange_inertia = 1/12.0*bottom_flange_width*bottom_flange_thick**3 + area_bottom_flange*(y_cg - y_cg_bottom_flange)**2.0
    # Web:
    web_inertia = 1/12.0*web_thick*web_height**3 + area_web*(y_cg - y_cg_web)**2.0
    # Top flange:
    top_flange_inertia = 1/12.0*top_flange_width*top_flange_thick**3 + area_top_flange*(y_cg - y_cg_top_flange)**2.0
    # Moment of inertia of section:
    total_moment_inertia = bottom_flange_inertia + web_inertia +top_flange_inertia
    # Create dictionary of results:
    dict_result = {'section_area': total_area, 'y_bar': y_cg, 'Iz':total_moment_inertia}
    return dict_result

def compute_transformed_properties(y_bar,area,Iz):
    # Calculate total area of all parts:
    sigma_A = 0
    for i in area:
        sigma_A += i
    # Calculate total product of area and y_bar:
    sigma_A_y_bar = 0
    for i in range(len(area)):
        sigma_A_y_bar += y_bar[i]*area[i]
    # Calculate vertical center of gravity position:
    y_bar_transformed = sigma_A_y_bar/sigma_A
    # Transformed moment of inertia
    Iz_transformed = 0
    for i in range(len(area)):
        Iz_transformed += Iz[i] + area[i]*(y_bar_transformed-y_bar[i])**2
    return Iz_transformed

def compute_mass_bridge(area_steel,area_concrete,density_steel,density_concrete):
    """This function is used to compute the total linear unit mass of the bridge in kg/m

    Args:
        area_steel (float): total area of steel in bridge.
        area_concrete (float): total area of concrete in bridge.
        density_steel (float): density of steel kg/m3.
        density_concrete (float): density of concrete kg/m3

    Returns:
        float: total linear unit mass of the bridge in kg/m.
    """
    mass_steel = area_steel*density_steel # linear unit mass of steel in kg/m
    mass_concrete = area_concrete*density_concrete # linear unit mass of concrete in kg/m
    total_mass = mass_steel+mass_concrete # total unit mass of bridge in kg/m
    return total_mass*1. # add an addition 15% of mass to include reinforcement, stiffners, bolts, etc.

def compute_first_frequency(E,Iz,beam_span,mass):
    return pi**2*sqrt((E*Iz)/(mass*beam_span**4))/(2*pi)

def read_aisc_sheet(excel_file_name,w_shape_interest):
    # Load aisc table into dataframe:
    aisc_shape_df = pd.read_excel(excel_file_name)
    # Select W-Shape of interest:
    w_shape_selected_df = aisc_shape_df.loc[aisc_shape_df['AISC_Manual_Label']==w_shape_interest]
    return w_shape_selected_df

if __name__ == '__main__':
    # --- Given:
    # material properties:
    # E_deck = 26.4e09 # Pa
    # E_steel = 211.0e09 # Pa
    # density_steel = 7800 # kg/m3
    # density_deck = 2400 # kg/m3
    # # girder properties:
    # n_girders = 3
    # girder_type = 'W24X250'
    # # general properties:
    # span =  16 # m
    # haunch = 0.0762 # m
    # # deck properties:
    # width_deck,thick_deck = 17.3,0.2032 # in m
    # # reinforcement details:
    # diameter_bar = 0.016 # m
    # spacing_bars = 0.3 # m
    # top_cover = 0.05 # m
    # bot_cover = 0.03 # m
    # # barrier properties:
    # width_barrier,height_barrier = 0.61,1 # m
    # # --- Girder calculation:
    # selected_girder = read_aisc_sheet('aisc_shapes_editted.xlsx',girder_type)
    # depth_girder = selected_girder.iloc[0]['d']/1000 # depth of girder in m
    # flange_width_girder = selected_girder.iloc[0]['bf']/1000 # flange width in m
    # y_bar_girder = depth_girder/2.0 # girder c.g in m
    # Iz_girder = selected_girder.iloc[0]['Ix']*10**-6 # Since AISC shapes are stored as mm4/10^6
    # area_girder = selected_girder.iloc[0]['A']*10**-6 # Girder area in m^2
    # mass_girder = selected_girder.iloc[0]['W'] # girder mass in kg/m
    # # Girder grouped:
    # y_bar_girder_group,Iz_girder_group,area_girder_group = np.ones(n_girders)*y_bar_girder,np.ones(n_girders)*Iz_girder,np.ones(n_girders)*area_girder
    # # --- Deck calculation:
    # n = modular_ratio(E_deck,E_steel)
    # Iz_deck = compute_rectangle_section_MOI([n*width_deck,thick_deck])
    # area_deck = rectangle_area(n*width_deck,thick_deck)
    # y_bar_deck = depth_girder + haunch + thick_deck/2.0
    # # --- Concrete barrier calculation:
    # Iz_barrier = 2*compute_rectangle_section_MOI([n*width_barrier,height_barrier])
    # area_barrier = 2*rectangle_area(n*width_barrier,height_barrier)
    # y_bar_barrier =  depth_girder + haunch + thick_deck + height_barrier/2
    # # --- Haunch calculation:
    # Iz_haunch = (n_girders)*compute_rectangle_section_MOI([n*flange_width_girder,haunch])
    # area_haunch = (n_girders)*rectangle_area(n*flange_width_girder,haunch)
    # y_bar_haunch = depth_girder + haunch/2.0
    # # --- Reinforcement calculation:
    # # Bottom:
    # A_bot_bar = pi*diameter_bar**2/4
    # y_bar_bot = depth_girder + haunch + bot_cover + diameter_bar/2
    # n_bot = ceil(width_deck/spacing_bars)
    # As_bot = n_bot*A_bot_bar-ceil(n*width_deck/spacing_bars)*A_bot_bar
    # Iz_bot = 0
    # # Top:
    # A_top_bar = pi*diameter_bar**2/4
    # y_bar_top = depth_girder + haunch + thick_deck - top_cover-diameter_bar/2
    # n_top = ceil(width_deck/spacing_bars)
    # As_top = n_top*A_top_bar-ceil(n*width_deck/spacing_bars)*A_top_bar
    # Iz_top = 0
    # # --- Group all inertial properties of bridge parts:
    # y_bar_group = list(y_bar_girder_group)
    # area_group = list(area_girder_group)
    # Iz_group = list(Iz_girder_group)
    # # y_bar:
    # y_bar_group.append(y_bar_deck) # Deck
    # y_bar_group.append(y_bar_haunch) # Haunch
    # # y_bar_group.append(y_bar_barrier) # Barrier
    # y_bar_group.append(y_bar_bot) # Bottom reinforcement
    # y_bar_group.append(y_bar_top) # Top reinforcement
    # # area:
    # area_group.append(area_deck) # Deck
    # area_group.append(area_haunch) # Haunch
    # # area_group.append(area_barrier) # Barrier
    # area_group.append(As_bot) # Bottom Reinforcement
    # area_group.append(As_top) # Top Reinforcement
    # # Iz:
    # Iz_group.append(Iz_deck) # Deck
    # Iz_group.append(Iz_haunch) # Haunch
    # # Iz_group.append(Iz_barrier) # Barrier
    # Iz_group.append(Iz_bot) # Bottom Reinforcement
    # Iz_group.append(Iz_top) # Top Reinforcement
    # # Mass of bridge
    # area_steel = n_girders*area_girder + n_top*A_top_bar + n_bot*A_bot_bar
    # area_concrete = (area_deck + area_haunch + area_barrier)/n
    # mass_bridge = compute_mass_bridge(area_steel,area_concrete,density_steel,density_deck)
    # # Tranformed Inertial Properties:
    # Iz_bridge = compute_transformed_properties(y_bar_group,area_group,Iz_group)
    # first_frequency = compute_first_frequency(E_steel,Iz_bridge,span,mass_bridge)
    # print('Test:')
    # print('Transformed MOI: {} m^4'.format(Iz_bridge))
    # print('Mass of bridge {} kg/m'.format(mass_bridge))
    # print('First frequency: {} Hz'.format(first_frequency))
    # Fatigue analysis and life prediction of composite highway bridge decks under traffic loading:
    # Material properties:
    # E_deck = 30.5e09 # Pa
    # E_steel = 205.0e09 # Pa
    # density_steel = 7800 # kg/m3
    # density_deck = 2400 # kg/m3
    # # Concrete deck geometry:
    # span = 40
    # deck_width,deck_height = 13.0,0.225 # m
    # haunch = 0.0 # m
    # diameter_bot_bar = 0.016 # m
    # diameter_top_bar = diameter_bot_bar # m
    # reinforcement_spacing_bottom = 0.30 # m
    # reinforcement_spacing_top = 0.30 # m
    # top_cover = 0.05 # m
    # bottom_cover = 0.03 # m
    # # Concrete barrier:
    # width_barrier,height_barrier = 0.35,0.88 # m,m
    # # Girder:
    # # External:
    # bf_external,tf_external = [0.450,0.05],[0.450,0.025]
    # web_external = [1.925,0.0095]
    # n_girder_external = 6
    # # Internal:
    # bf_internal,tf_internal = [0.670,0.050],[0.500,0.025]
    # web_internal = [1.925,0.0095]
    # n_girder_internal = 6
    # # Moment of inertia calculations:
    # n = modular_ratio(E_deck,E_steel)
    # # Deck:
    # Iz_deck = compute_rectangle_section_MOI([n*deck_width,deck_height])
    # area_deck = rectangle_area(n*deck_width,deck_height)
    # y_bar_deck = min(tf_external) + max(web_external) + min(bf_external) + haunch + deck_height/2.0
    # # Haunch:
    # Iz_haunch = (n_girder_external+n_girder_internal)*compute_rectangle_section_MOI([n*tf_external[0],haunch])
    # area_haunch = (n_girder_external+n_girder_internal)*rectangle_area(n*tf_external[0],haunch)
    # y_bar_haunch = min(tf_external) + max(web_external) + min(bf_external) + haunch/2.0
    # # Barrier:
    # # Iz_barrier = 2*compute_rectangle_section_MOI([n*width_barrier,height_barrier])
    # area_barrier = 2*rectangle_area(n*width_barrier,height_barrier)
    # # y_bar_barrier =  min(tf_external) + max(web_external) + min(bf_external) + haunch + height_barrier/2
    # # Steel reinforcement:
    # # Bottom:
    # A_bot_bar = pi*diameter_bot_bar**2/4
    # y_bar_bot = min(tf_external) + max(web_external) + min(bf_external) + haunch + bottom_cover + diameter_bot_bar/2
    # n_bot = ceil(deck_width/reinforcement_spacing_bottom)
    # As_bot = n_bot*A_bot_bar-ceil(n*deck_width/reinforcement_spacing_bottom)*A_bot_bar
    # Iz_bot = 0
    # # Top:
    # A_top_bar = pi*diameter_top_bar**2/4
    # y_bar_top = min(tf_external) + max(web_external) + min(bf_external) + haunch + deck_height - top_cover-diameter_top_bar/2
    # n_top = ceil(deck_width/reinforcement_spacing_top)
    # As_top = n_top*A_top_bar-ceil(n*deck_width/reinforcement_spacing_top)*A_top_bar
    # Iz_top = 0
    # # External Girders:
    # external_gider = compute_i_section_MOI(bf_external,web_external,tf_external)
    # y_bar_girder_external,area_girder_external,Iz_girder_external = external_gider['y_bar'],external_gider['section_area'],external_gider['Iz']
    # y_bar_grouped_external, Iz_group_external,area_group_external = list(np.ones(n_girder_external)*y_bar_girder_external),list(np.ones(n_girder_external)*Iz_girder_external),list(np.ones(n_girder_external)*area_girder_external)
    # # Internal Girder:
    # internal_gider = compute_i_section_MOI(bf_internal,web_internal,tf_internal)
    # y_bar_girder_internal,area_girder_internal,Iz_girder_internal = internal_gider['y_bar'],internal_gider['section_area'],internal_gider['Iz']
    # y_bar_grouped_internal, Iz_group_internal,area_group_internal = list(np.ones(n_girder_internal)*y_bar_girder_internal),list(np.ones(n_girder_internal)*Iz_girder_internal),list(np.ones(n_girder_internal)*area_girder_internal)
    # # Append inertia values into lists:
    # # y_bar:
    # y_bar_grouped = []
    # for i in y_bar_grouped_external:
    #     y_bar_grouped.append(i)
    # for i in y_bar_grouped_internal:
    #     y_bar_grouped.append(i)
    # # area:
    # area_group = []
    # for i in area_group_external:
    #     area_group.append(i)
    # for i in area_group_internal:
    #     area_group.append(i)
    # # Iz:
    # Iz_group = []
    # for i in Iz_group_external:
    #     Iz_group.append(i)
    # for i in Iz_group_internal:
    #     Iz_group.append(i)
    # # y_bar:
    # y_bar_grouped.append(y_bar_deck) # Deck
    # y_bar_grouped.append(y_bar_haunch) # Haunch
    # # y_bar_grouped.append(y_bar_barrier) # Barrier
    # y_bar_grouped.append(y_bar_bot) # Bottom reinforcement
    # y_bar_grouped.append(y_bar_top) # Top reinforcement
    # # area:
    # area_group.append(area_deck) # Deck
    # area_group.append(area_haunch) # Haunch
    # # area_group.append(area_barrier) # Barrier
    # area_group.append(As_bot) # Bottom Reinforcement
    # area_group.append(As_top) # Top Reinforcement
    # # Iz:
    # Iz_group.append(Iz_deck) # Deck
    # Iz_group.append(Iz_haunch) # Haunch
    # # Iz_group.append(Iz_barrier) # Barrier
    # Iz_group.append(Iz_bot) # Bottom Reinforcement
    # Iz_group.append(Iz_top) # Top Reinforcement
    # # Mass of bridge
    # area_steel = n_girder_external*area_girder_external + n_girder_internal*area_girder_internal + n_top*A_top_bar + n_bot*A_bot_bar
    # area_concrete = (area_deck + area_haunch + area_barrier)/n
    # mass_bridge = compute_mass_bridge(area_steel,area_concrete,density_steel,density_deck)
    # # Tranformed Inertial Properties:
    # Iz_bridge = compute_transformed_properties(y_bar_grouped,area_group,Iz_group)
    # first_frequency = compute_first_frequency(E_steel,Iz_bridge,span,mass_bridge)
    # print('Fatigue Analysis Paper:')
    # print('Transformed MOI: {} m^4'.format(Iz_bridge))
    # print('Mass of bridge {} kg/m'.format(mass_bridge))
    # print('First frequency: {} Hz'.format(first_frequency))
    # Rob's thesis:
    # Material Properties:
    E_deck = 24.9e09 # Pa
    density_concrete = 2400 # density of concrete in kg/m3
    E_steel = 200.0e9 # Pa
    density_steel = 7850 # density of steel in kg/m3
    # Concrete deck geometry:
    span = 21.3 # m
    deck_width,deck_height = 15,0.1905 # m
    haunch = 0.0254 # m
    diameter_bot_bar = 0.0127 # m
    diameter_top_bar = diameter_bot_bar # m
    reinforcement_spacing_bottom = 0.450 # m
    reinforcement_spacing_top = 0.450 # m
    top_cover = 0.0762 # m
    bottom_cover = 0.038 # m
    # Concrete barrier:
    width_barrier,height_barrier = 0.356,0.44 # m,m
    # Girder geometry:
    # External:
    bf_external,tf_external = [0.3556,0.01905],[0.2286,0.01905] # m
    web_external = [1.372,0.00953] # m
    n_girders_external = 2
    # Internal:
    bf_internal,tf_internal = [0.3556,0.03175],[0.2286,0.01905] # m
    web_internal = [1.372,0.00953] # m
    n_girders_internal = 7
    # Moment of inertia calculations:
    n = modular_ratio(E_deck,E_steel)
    # Deck:
    Iz_deck = compute_rectangle_section_MOI([n*deck_width,deck_height])
    area_deck = rectangle_area(n*deck_width,deck_height)
    y_bar_deck = min(tf_external) + max(web_external) + min(bf_external) + haunch + deck_height/2.0
    # Haunch:
    Iz_haunch = (n_girders_external+n_girders_internal)*compute_rectangle_section_MOI([n*tf_external[0],haunch])
    area_haunch = (n_girders_external+n_girders_internal)*rectangle_area(n*tf_external[0],haunch)
    y_bar_haunch = min(tf_external) + max(web_external) + min(bf_external) + haunch/2.0 
    # Barrier:
    Iz_barrier = 2*compute_rectangle_section_MOI([n*width_barrier,height_barrier])
    area_barrier = 2*rectangle_area(n*width_barrier,height_barrier)
    y_bar_barrier =  min(tf_external) + max(web_external) + min(bf_external) + haunch + deck_height + height_barrier/2
    # Steel reinforcement:
    # Bottom:
    A_bot_bar = pi*diameter_bot_bar**2/4
    y_bar_bot = min(tf_external) + max(web_external) + min(bf_external) + haunch + bottom_cover + diameter_bot_bar/2
    n_bot = ceil(deck_width/reinforcement_spacing_bottom)
    As_bot = n_bot*A_bot_bar-ceil(n*deck_width/reinforcement_spacing_bottom)*A_bot_bar
    Iz_bot = 0
    # Top:
    A_top_bar = pi*diameter_top_bar**2/4
    y_bar_top = min(tf_external) + max(web_external) + min(bf_external) + haunch + deck_height - top_cover-diameter_top_bar/2
    n_top = ceil(deck_width/reinforcement_spacing_top)
    As_top = n_top*A_top_bar-ceil(n*deck_width/reinforcement_spacing_top)*A_top_bar
    Iz_top = 0
    # Girders
    # External:
    girder_inertial_external = compute_i_section_MOI(bf_external,web_external,tf_external)
    y_bar_girder_external,area_girder_external,Iz_girder_external = girder_inertial_external['y_bar'],girder_inertial_external['section_area'],girder_inertial_external['Iz']
    y_bar_grouped_external, Iz_group_external,area_group_external = list(np.ones(n_girders_external)*y_bar_girder_external),list(np.ones(n_girders_external)*Iz_girder_external),list(np.ones(n_girders_external)*area_girder_external)
    # Internal:
    girder_inertial_internal = compute_i_section_MOI(bf_internal,web_internal,tf_internal)
    y_bar_girder_internal,area_girder_internal,Iz_girder_internal = girder_inertial_internal['y_bar'],girder_inertial_internal['section_area'],girder_inertial_internal['Iz']
    y_bar_grouped_internal, Iz_group_internal,area_group_internal = list(np.ones(n_girders_internal)*y_bar_girder_internal),list(np.ones(n_girders_internal)*Iz_girder_internal),list(np.ones(n_girders_internal)*area_girder_internal)
    # Append inertia values into lists:
    # y_bar:
    y_bar_grouped = []
    for i in y_bar_grouped_external:
        y_bar_grouped.append(i)
    for i in y_bar_grouped_internal:
        y_bar_grouped.append(i)
    # area:
    area_group = []
    for i in area_group_external:
        area_group.append(i)
    for i in area_group_internal:
        area_group.append(i)
    # Iz:
    Iz_group = []
    for i in Iz_group_external:
        Iz_group.append(i)
    for i in Iz_group_internal:
        Iz_group.append(i)
    # y_bar:
    y_bar_grouped.append(y_bar_deck) # Deck
    y_bar_grouped.append(y_bar_haunch) # Haunch
    y_bar_grouped.append(y_bar_barrier) # Barrier
    y_bar_grouped.append(y_bar_bot) # Bottom reinforcement
    y_bar_grouped.append(y_bar_top) # Top reinforcement
    # area:
    area_group.append(area_deck) # Deck
    area_group.append(area_haunch) # Haunch
    area_group.append(area_barrier) # Barrier
    area_group.append(As_bot) # Bottom Reinforcement
    area_group.append(As_top) # Top Reinforcement
    # Iz:
    Iz_group.append(Iz_deck) # Deck
    Iz_group.append(Iz_haunch) # Haunch
    Iz_group.append(Iz_barrier) # Barrier
    Iz_group.append(Iz_bot) # Bottom Reinforcement
    Iz_group.append(Iz_top) # Top Reinforcement
    # Mass of bridge:
    area_steel = n_girders_external*area_girder_external + n_girders_internal*area_girder_internal + n_top*A_top_bar + n_bot*A_bot_bar
    area_concrete = (area_deck + area_haunch + area_barrier)/n
    mass_bridge = compute_mass_bridge(area_steel,area_concrete,density_steel,density_concrete)
    # Tranformed Inertial Properties:
    Iz_bridge = compute_transformed_properties(y_bar_grouped,area_group,Iz_group)
    first_frequency = compute_first_frequency(E_steel,Iz_bridge,span,5600)
    print('Rob thesis:')
    print('Transformed MOI: {} m^4'.format(Iz_bridge))
    print('Mass of bridge: {} kg/m'.format(mass_bridge))
    print('First frequency: {} Hz'.format(first_frequency))