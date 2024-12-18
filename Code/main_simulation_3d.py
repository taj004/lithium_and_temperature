# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:27:22 2024

@author: shin_
"""

# Import the neseccary packages
import numpy as np
import porepy as pp
import pypardiso

import fracture_network_soultz

import os

import equations
import constant_params
import reactive_transport
import update_param
import set_param

# %% Initialize variables related to the domain

# Start with the domain

# Define mdg

scale = 1e3
domain = {
    "xmin": -4.0 * scale, "xmax": 2.0 * scale, 
    "ymin": -3.0 * scale, "ymax": 3.0 * scale,
    "zmin":  0.0 * scale, "zmax": 8.0 * scale
    } 
         
xx = 600 

mesh_args={
    "mesh_size_frac" : xx/4,
    "mesh_size_min"  : xx,
    "mesh_size_bound": xx 
    }
     
mdg = fracture_network_soultz.fracture_network_soultz_test(
    domain, mesh_args, add_extra=True
    )

# Keywords
mass_kw = "mass"
transport_kw = "transport"
flow_kw = "flow"

pressure = "pressure"
temperature = "temperature"
tracer = "passive_tracer"

mortar_pressure = "mortar_pressure"
mortar_temperature_convection = "mortar_temperature_convection"
mortar_temperature_conduction = "mortar_temperature_conduction"
mortar_tracer = "mortar_tracer"

# %% Loop over the mdg, and set initial and default data

ii = 0

for sd, d in mdg.subdomains(return_data=True):

    # Initialize the primary variable dictionaries
    d[pp.PRIMARY_VARIABLES] = {
        pressure: {"cells": 1},
        temperature: {"cells": 1},
        tracer: {"cells": 1}
    }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(sd.num_cells)

    # --------------------------------- #

    # Initial porosity, aperture and permeablity  
    
    aperture = constant_params.open_aperture()  
    
    specific_volume = np.power(aperture, mdg.dim_max() - sd.dim)
    
    if sd.dim==mdg.dim_max():   
        porosity = 0.2 * unity 
        K = 1e-13 * unity
    else :
        porosity = unity
        K = (np.power(aperture, 2) / 12) * unity
    # end if
    
    Kxx = K * specific_volume / constant_params.dynamic_viscosity() 
    
    perm = pp.SecondOrderTensor(kxx = Kxx)

    # Initial guess for Darcy flux    
    init_darcy_flux = np.zeros(sd.num_faces)   
     
    #Densities, heat capacity and conduction    
    solid_density = 2750.0
    fluid_density = update_param.rho()
   
    heat_capacity = (
        porosity * fluid_density * constant_params.specific_heat_capacity_fluid() # fluid part
        + (1-porosity) * solid_density * constant_params.specific_heat_capacity_solid()
        )
    
    conduction = (
        porosity * constant_params.fluid_conduction() # fluid part
        + (1-porosity) * constant_params.solid_conduction() # solid part
        ) 
    
    # --------------------------------- #
        
    # Set the values in dictionaries
    mass_data = {
        "porosity": unity, # porosity.copy(),
        "mass_weight": specific_volume * porosity.copy() ,
        "aperture": aperture * unity,
        "specific_volume": specific_volume * unity,
    }
    
    flow_data = {
        "mass_weight": specific_volume * porosity,
        "permeability": Kxx,
        "second_order_tensor": perm,
        "darcy_flux": init_darcy_flux,
        "normal_permeability": Kxx
    }

    transport_data = {
        "darcy_flux": init_darcy_flux,
    }

    temp_data = {
        "mass_weight": specific_volume * heat_capacity,
        "solid_density": solid_density,
        "darcy_flux": init_darcy_flux,
        "second_order_tensor": pp.SecondOrderTensor(
            specific_volume * conduction * unity
            ),
    }

    passive_tracer_data = {
        "darcy_flux": init_darcy_flux,
        "mass_weight": specific_volume * porosity.copy(),
    }

    # The Initial values also serve as reference values
    reference_data = {
        "porosity": porosity.copy(),
        "permeability": K.copy(),
        "permeability_aperture_scaled": K.copy(),
        "mass_weight": porosity.copy(),
    }

    # --------------------------------- #

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(sd, d, mass_kw, mass_data)

    d[pp.PARAMETERS][flow_kw] = flow_data
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
    d[pp.PARAMETERS][temperature] = temp_data
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    d[pp.PARAMETERS][tracer] = passive_tracer_data
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}

    # The reference values
    #d[pp.PARAMETERS]["reference"] = reference_data

    # Set some data only in the highest dimension, in order to avoid techical issues later on
    if sd.dim == mdg.dim_max():

        d[pp.PARAMETERS][transport_kw].update(
            {
                "time_step": 1 * pp.YEAR,  # s,
                "current_time": 0,
                "final_time": 60 * pp.YEAR,
                "constant_time_step": True
            }
        )
        d[pp.PARAMETERS]["grid_params"] = {}
        d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
        d[pp.PARAMETERS]["previous_newton_iteration"] = {
            "Number_of_Newton_iterations": [],
            "AD_full_flux" : pp.ad.DenseArray(init_darcy_flux)
        }
    # end if

    # --------------------------------- #

    # Set state (initial) values
    
    # Pressure
    pressure_state = 0 * unity 
    
    # Tracer
    tracer_state = unity 
    
    # Temperature
    temp_state = constant_params.ref_temp() * unity
    
    d[pp.STATE].update(
        {
            "dimension": sd.dim * unity,
            "sd_index": ii * unity,
            pressure: pressure_state,
            temperature: temp_state,
            tracer: tracer_state , # 0 * unity,
            pp.ITERATE: {
                pressure: pressure_state,
                temperature: temp_state,
                tracer: tracer_state,
            },
        }
    )
    if sd.dim < 3:
        ii +=1
        
# end sd,d-loop

set_param.set_bc_param_3d(mdg)

#%% The interfaces
for e,d in mdg.interfaces(return_data=True):
  
    # Set state
    pp.set_state(d)
    pp.set_iterate(d)
    
    # Primary variables
    d[pp.PRIMARY_VARIABLES] = {
        mortar_pressure: {"cells": 1},
        mortar_temperature_convection: {"cells": 1},
        mortar_temperature_conduction: {"cells": 1},
        mortar_tracer: {"cells": 1}
        }
    
    unity = np.ones(e.num_cells)
        
    # Initial values
    vals = {
        mortar_pressure: 0.0 * unity,
        mortar_temperature_convection: 0.0 * unity,
        mortar_temperature_conduction: 0.0 * unity,
        mortar_tracer: 0.0 * unity 
        }
    
    # Set to state
    d[pp.STATE].update(vals)
    d[pp.STATE][pp.ITERATE].update(vals)
    
    # Set the parameter dictionary
    flow_params = {"darcy_flux": 0.0 * unity,
                   "normal_diffusivity": 0.0 * unity}
    
    
    temp_params = {"darcy_flux": 0.0 * unity,
                  "normal_diffusivity": 0 * unity}
    
    tracer_params = {"darcy_flux": 0.0 * unity}
    
    pp.initialize_data(
        grid=e, data=d, keyword="flow", specified_parameters=flow_params
        )
          
    d[pp.PARAMETERS][temperature] = temp_params
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    
    d[pp.PARAMETERS][tracer] = tracer_params
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}
    
# end interface-loop

# Set the conductive normal fluxes
update_param.update_interface(mdg)

#%% Aspects for the transport calculations

# Equation system
equation_system = pp.ad.EquationSystem(mdg)

# and the initial equations
eqs = equations.EquationConstruction(
    equation_system=equation_system, 
    well_source=True
    )

rt_eqs = reactive_transport.ReactiveTransport(pde_solver=eqs)

set_param.set_source_rate(mdg)

# %% Solve for pressure and set flow field
eqs.get_incompressible_flow_eqs(
    equation_system, well_source=True
    )

J,r = equation_system.assemble_subsystem(
    equations = eqs.eq_names,
    variables= eqs.ad_vars
    )

pressure_sol = pypardiso.spsolve(J,r) 

equation_system.set_variable_values(
    values=pressure_sol,
    variables=eqs.ad_vars,
    to_iterate=True,
    to_state=True,
    additive=False
    )


update_param.update_darcy(equation_system)
eqs.remove_equations(eqs.eq_names)

 #%% Prepere for exporting

fields = [pressure, "sd_index", 
          tracer, temperature]

if len(mdg.subdomains(dim=2)) == 39:
    folder_name = "3d/without_extra/"
elif len(mdg.subdomains(dim=2)) == 40:
      folder_name = "3d/with_extra/"
# end if

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# end if
export = pp.Exporter(mdg, file_name="vars", folder_name=folder_name)


def save_array(array, kw, time):
    save_name = folder_name + kw + "_at_time_" + str(time)
    np.save(file=save_name, arr=array)
    

#%% Preper for time loop

store_time = np.array(
    [1.0, 1010e4]
    ) * pp.YEAR # the last one is for safty

# The data in the highest dimension
sd = mdg.subdomains(dim=mdg.dim_max())[0]
data = mdg.subdomain_data(sd)

current_time = data[pp.PARAMETERS]["transport"]["current_time"]
time = []
time.append(current_time)

time_step = data[pp.PARAMETERS]["transport"]["time_step"]
final_time = data[pp.PARAMETERS]["transport"]["final_time"]
j=0
step=0

#%% To store stuff fixed in space

sd_out, point_out = equations.production_point(mdg, for_1d_analysis=True)
data_out = mdg.subdomain_data(sd_out)

tracer_time = []
temp_time = []

# Initial values
tracer_var = data_out[pp.STATE][tracer]
tracer_time.append(tracer_var[point_out[0]])

temp_var = data_out[pp.STATE][temperature]
temp_time.append(temp_var[point_out[0]])

#%% Initialise equations

eqs.get_temperature_eqs(
    equation_system, 
    iterate=False, 
    well_source=True
    )
eqs.get_tracer_eqs(
    equation_system, 
    iterate=False, 
    well_source=True
    )

#%% Time loop
while current_time < final_time :

    print(f"Current time {current_time/pp.YEAR}")
    step+=1

    # Get the Jacobian and the rhs
    A, b = equation_system.assemble_subsystem(
        variables=rt_eqs.pde_solver.ad_vars,
        equations=rt_eqs.pde_solver.eq_names
        )

    # Previous solution
    x_prev = equation_system.get_variable_values(
        variables=rt_eqs.pde_solver.ad_vars,
        from_iterate=True
        )

    # Solve in order to advance forward in time.
    x = pypardiso.spsolve(A, b)
    
    # New solution
    x_new = x_prev + x

    # Distribute the solution
    equation_system.set_variable_values(
        values=x_new.copy(),
        variables=rt_eqs.pde_solver.ad_vars,
        to_state=False,
        to_iterate=True,
        additive=False,
    )

    equation_system.set_variable_values(
        values=x_new.copy(),
        variables=eqs.ad_vars,
        to_state=True,
        to_iterate=False,
        additive=False,
    )
    
    # Update time
    current_time += time_step
    time.append(current_time)
    
    # Store
    if np.isclose(current_time,store_time[j]):
        # Save in space
        export.write_vtu(data=fields, time_step=int(current_time/pp.YEAR))      
        j+=1
    # end if

    # Get tracer in production point
    tracer_var = data_out[pp.STATE][tracer]
    tracer_time.append(tracer_var[point_out[0]])
    
    temp_var = data_out[pp.STATE][temperature]
    temp_time.append(temp_var[point_out[0]])
# end time-loop


#%% Final export
print(f"Current time {current_time/pp.YEAR}")

export.write_vtu(data=fields, time_step=int(current_time/pp.YEAR))

# Save in time
save_array(array=np.asarray(tracer_time), 
    kw=tracer, 
    time=int(current_time/pp.YEAR) 
    )
   
save_array(array=np.asarray(temp_time), 
    kw=temperature, 
    time=int(current_time/pp.YEAR) 
    )

