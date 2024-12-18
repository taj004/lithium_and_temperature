# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:58:25 2024

@author: shin_
"""

# Import the neseccary packages
import numpy as np
import porepy as pp
import pypardiso

import os

import equations
import constant_params
import reactive_transport
import update_param
import set_param
        
#%% Setup for the MC        
        
# Generate the fracture network, using the P21 condition
num_fracs = np.array(
    [8, 30, 105] # it is set up to run one density at the time; if you want to run several at one time, that is at your own risk 
    )

mesh_size = np.array([70, 400, 2500])

# For stability purposes with the high-performance computer at UiB,
# the MC-simulation is split in two parts. 
run_nr = 1  

save_realisations = np.array([0,46])

#%% Start MC
for ii in range(num_fracs.size):
    
    num_ignored = 0
    
    if run_nr == 1:
        np.random.seed(4)  
        jj = 0
        num_realisation = 7000
        
    elif run_nr == 2:
        np.random.seed(5)
        jj = 7000       
        num_realisation = 10000  
    else: 
        raise ValueError("Unknown run number")

    while jj < num_realisation :
        print(f"Start density {num_fracs[ii]}, realisation nr {jj}")
        # Initialize variables related to the domain
        # Start with the domain
        
        # Define mdg, which is [0, 3] km x [0,2] km
        mesh_args={"mesh_size_frac" : mesh_size[ii],
                   "mesh_size_min"  : mesh_size[ii], # 190.0,
                   "mesh_size_bound": mesh_size[ii], # 190.0
                   }
        
        scale = 1e3
        domain = {"xmin": 0 * scale, "xmax": 3 * scale, 
                  "ymin": 0 * scale, "ymax": 2 * scale} 
    
        ff=[]
        for kk in range(num_fracs[ii]):
            
            # The centres 
            centre_x = np.random.uniform(0.0 * scale , 3.0 * scale)
            centre_y = np.random.uniform(0.0 * scale , 2.0 * scale)
            
            centre = np.array([centre_x, centre_y])
            
            # The length
            
            # expected
            e_l = 500.0 # m
            s_l = 200.0 #np.sqrt(10000) # m
            
            # underlying values, 
            # see https://stackoverflow.com/questions/48014712/get-lognormal-random-number-given-log10-mean-and-log10-standard-deviation/48016650#48016650
            normal_s = np.sqrt(
                np.log(1 + (s_l/e_l)**2 )
                )
            normal_e = np.log(e_l) - normal_s**2 / 2

            length = np.random.lognormal(normal_e, normal_s)
            
            # The start and end points of the fracture
            start_x = centre_x - length / 2    
            start_y = centre_y - 0*length / 2       
            
            start_pt = np.array([start_x, start_y])
            
            end_x = centre_x + length / 2    
            end_y = centre_y + 0*length / 2    
            
            end_pt = np.array([end_x, end_y])
            
            # Rotation angle
            angle = np.random.uniform(-np.pi/2, np.pi/2)
            
            # The rotation matrix
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
                ])
            
            # Check that the unrotated fractures have the expected length
            length_test = np.isclose(end_x-start_x, length)
            assert length_test
            
            diff_before = np.linalg.norm(end_pt-start_pt) 
            
            # Rotate teh fracture coordinates
            rotate_end_pt   = R.dot( (end_pt - centre) )  + centre
            rotate_start_pt = R.dot( (start_pt - centre) ) + centre
 
            diff_after = np.linalg.norm(rotate_end_pt-rotate_start_pt)
      
            # The Euclidean norm should be the same before and after rotation
            assert np.isclose(diff_after- diff_before, 0.0) # use isclose for numerical reasons
            
            # Stack teh fracture
            frac = np.vstack((rotate_start_pt, rotate_end_pt)).T
            
            ff.append(pp.LineFracture(frac))
        # end kk-loop
        
        #%% Mesh
        
        # Certain networks may cause errors 
        # (zero volumes, not in domain etc.). Skip them
        try:
            network_2d = pp.FractureNetwork2d(
                fractures = ff , 
                domain=pp.Domain( 
                    bounding_box = domain
                    ),
                )       
            mdg = network_2d.mesh(mesh_args)
                        
            # If no error, advance

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
            
            # Loop over the mdg, and set initial and default data
                        
            init_temp = constant_params.ref_temp()
            
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
                if sd.dim == mdg.dim_max():
                    aperture = 1.0
                else: 
                    aperture = constant_params.open_aperture()
                # end if
                        
                specific_volume = np.power(aperture, mdg.dim_max() - sd.dim)
                
                if sd.dim==mdg.dim_max():   
                    porosity = 0.2 * unity
                    K = 1e-13 * unity
                else :
                    porosity = unity
                    K = (np.power(aperture, 2) / 12) * unity
                # end if
                
                Kxx = K * specific_volume  / constant_params.dynamic_viscosity() 
                
                perm = pp.SecondOrderTensor(kxx = Kxx)
            
                # Initial guess for Darcy flux    
                init_darcy_flux = np.zeros(sd.num_faces)   # m/s
                    
                # Densities, heat capacity and conduction
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
                    "porosity": porosity.copy(),
                    "mass_weight": specific_volume * porosity.copy() ,
                    "aperture": aperture * unity,
                    "specific_volume": specific_volume * unity,
                    "specific_volume_scalar" : specific_volume
                }
                
                flow_data = {
                    "mass_weight": specific_volume * porosity,
                    "permeability": Kxx,
                    "second_order_tensor": perm,
                    "darcy_flux": init_darcy_flux,
                    "normal_permeability": Kxx 
                }
            
                transport_data = {        
                    "darcy_flux": init_darcy_flux
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
                            "time_step": 1.5 * pp.YEAR,  # s
                            "current_time": 0.0,
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
                temp_state = init_temp * unity 
                
                d[pp.STATE].update(
                    {
                        "dimension": sd.dim * unity,
                        pressure: pressure_state,
                        temperature: temp_state,
                        tracer: tracer_state ,
                        pp.ITERATE: {
                            pressure: pressure_state,
            
                            temperature: temp_state,
                            tracer: tracer_state,
                        },
                    }
                )
                    
                # end g,d-loop
            
            set_param.set_bc_param_mc(mdg)
            set_param.set_source_rate(mdg)

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
            eqs = equations.EquationConstruction(equation_system=equation_system, 
                                                 well_source=True)
            
            rt_eqs = reactive_transport.ReactiveTransport(pde_solver=eqs)
            
            # %% Solve for pressure and set flow field
            eqs.get_incompressible_flow_eqs(equation_system, well_source=True)
            
            J,r = equation_system.assemble_subsystem(
                equations = eqs.eq_names,
                variables= eqs.ad_vars
                )
            
            pressure_sol =  pypardiso.spsolve(J,r) 
            
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
            
            fields = [pressure, "dimension", tracer, temperature]
                    
            folder_name = (
                "mc/frac_density_"+str(num_fracs[ii])+"/"
                )
            if not os.path.exists(folder_name): 
                os.makedirs(folder_name)
            # end if
            
            # For realisations
            sub_folder = (
                folder_name + "realisations/"
                )
            
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            # end if 
            
            # Save for thing in space
            if jj in save_realisations:

                # Make exporter
                folder_name += "in_space_"+str(jj)+"/"

                export = pp.Exporter(
                    mdg, 
                    file_name=(
                        "Realisation_nr_"+str(jj)+"_vars_at_time" 
                        ), 
                    folder_name=folder_name,
                    )
            # end if
            
            # To store stuff fixed in space
            p = equations.production_point(mdg)
            tracer_time = []
            temp_time = []
               
            # The data in the highest dimension
            sd = mdg.subdomains(dim=mdg.dim_max())[0]
            data = mdg.subdomain_data(sd)
            
            # Get initial values in the production point
            tracer_var = data[pp.STATE][tracer]
            tracer_time.append(tracer_var[p[0][0]])
            
            temp_var = data[pp.STATE][temperature]
            temp_time.append(temp_var[p[0][0]])
            
            #%% To store stuff fixed in space
    
            store_time = np.array([5, 25, 1010e4]) * pp.YEAR # Last one is for safty
            
            current_time = data[pp.PARAMETERS]["transport"]["current_time"]
            time = []
            time.append(time)
            
            time_step = data[pp.PARAMETERS]["transport"]["time_step"]
            final_time = data[pp.PARAMETERS]["transport"]["final_time"]
            j=0
            
            
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
                    additive=False
                    )
                
                # Update time
                current_time += time_step
                time.append(current_time)
                
                if jj in save_realisations : 
                    if np.abs(current_time-store_time[j]) < pp.YEAR:
                        # Save in space
                        export.write_vtu(
                            data=fields, time_step=int(current_time/pp.YEAR)
                            )          
                        j+=1
                    # end if
                
                # Get tracer in production point
                tracer_var = data[pp.STATE][tracer]
                tracer_time.append(tracer_var[p[0][0]])
                
                temp_var = data[pp.STATE][temperature]
                temp_time.append(temp_var[p[0][0]])
            # end time-loop
            
            #%% Final store
           
            if jj in save_realisations :
                # Export
                export.write_vtu(
                    data=fields, time_step=int(current_time/pp.YEAR)
                    )
            # end if
            
            # Save realisations
            def save_array(array, kw):
                save_name = (
                    sub_folder + kw + "_realisation_nr_" + str(jj)
                    )
                np.save(file=save_name, arr=array)
            # end function  
            
            # Save 
            save_array(
                array=np.asarray(tracer_time), 
                kw=tracer
                )
               
            save_array(
                array=np.asarray(temp_time), 
                kw=temperature, 
                )
            
            # Check if solutions are meaningful
            physical_sol_is_ok = True
            
            for sd, d in mdg.subdomains(return_data=True):
                
                temp = d[pp.STATE][temperature]
                temp_sol_ok = np.logical_and(
                    temp <= init_temp      + 2e-1, # The pm 0.2 is to account for floating numbers
                    temp >= init_temp - 50 - 2e-1
                    ).all()
                
                conc = d[pp.STATE][tracer]
                conc_sol_ok = np.logical_and(
                    conc <= 1.0 + 5e-2,
                    conc >= 0.0 - 5e-2 
                    ).all()
                
                # If one solution is unphysical, ignore realisation
                if not temp_sol_ok:
                    physical_sol_is_ok = False
                    break
        
                if not conc_sol_ok:
                    physical_sol_is_ok = False
                    break
                # end ifs
            # end-mdg-loop     
                
            # Sucsessfull realisation if ok solution
            if physical_sol_is_ok:
                print("Solution passed")
                if jj+1 == num_realisation: 
                    print(f"Done realisation nr {jj}.")   
                else:
                    print(f"Done realisation nr {jj}. Start new one")
                # end if              

                jj += 1 # Increment realisation
            
            else:
                num_ignored += 1
                print(f"Realisation nr {jj} did not pass. Try new one")
            # end if-else
                
        # end try-part
        
        # Exceptions
        except AssertionError:
            num_ignored += 1
        except ValueError:
            num_ignored += 1
        except IndexError:
            num_ignored += 1
        # end try-except
        
    # end while
    
    if num_fracs[ii] == num_fracs[-1]:
        print(f"Done density {num_fracs[ii]}")
    else:
        print(f"Done denisty {num_fracs[ii]}. Start new one")
    # end if
    
    print(f"number of ignored realisations {num_ignored}")
# end frac density
