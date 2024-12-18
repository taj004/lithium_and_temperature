#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update various parameters for the computations

@author: uw

TODO: The function needs a cleanup, since there are some redundencies
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import constant_params

# ---- Constitituive laws

def equil_state_to_param(mdg):
    """Functionallity if the equilibrium constants are solved 
    as a part of the overall problem
    
    """
    # Splitt the equilibrium constants for components
    # and precipitation species
    x1 = np.zeros(mdg.num_subdomain_cells() * 3, dtype=int)
    x2 = np.zeros(mdg.num_subdomain_cells() * 2, dtype=int)
    for i in range(mdg.num_subdomain_cells()):
        j = i * 5
        inds1 = np.array([j, j + 1, j + 2])
        inds2 = np.array([j + 3, j + 4])

        x1[3 * i : 3 * i + 3] = inds1
        x2[2 * i : 2 * i + 2] = inds2
    # end i-loop

    equil = np.zeros(5 * mdg.num_subdomain_cells())
    cell_val = 0
    for sd, d in mdg.subdomains(return_data=True):
        if sd.dim == mdg.dim_max():
            data = d
        # end if
        inds = slice(cell_val, cell_val + 5 * sd.num_cells)
        equil[inds] = d[pp.STATE][pp.ITERATE]["equilibrium_constants"]
        cell_val += 5 * sd.num_cells
    # end loop

    data["parameters"]["chemistry"].update(
        {
            "cell_equilibrium_constants_comp": sps.diags(equil[x1]),
            "cell_equilibrium_constants_prec": sps.diags(equil[x2]),
        }
    )

def taylor_app_equil(temp):
    """
    Taylor approximation of the equililibrium constants
    """
    delta_H = constant_params.standard_enthalpy()
    dt = temp - constant_params.ref_temp()
    taylor = (
        1
        + delta_H * dt / constant_params.ref_temp() ** 2
        # + delta_H*(delta_H-2*constant_params.ref_temp()) * np.power(dt,2) / constant_params.ref_temp()**4
    )
    return taylor


def rho(p=0.0, temp=0.0):
    """
    Constitutive law relating density with pressure and temperature

    The choise of 0 as default value is due to the PorePy Ad-framework does not support
    ``None`` as input to functions

    """

    # reference density
    rho_f = 1e3

    # Pressure values
    c = 1e-9 # compresibility [1/Pa]
    p_ref = constant_params.ref_p()  # reference pressure [Pa]

    # Temperature values
    beta = 4e-4
    temp_ref = constant_params.ref_temp()

    # Safty if default input is used
    if isinstance(p, (float, int)) is True:  # p is a scalar
        if p == 0.0:
            p = p_ref
    # end if

    if isinstance(temp, (float, int)) is True:
        if temp == 0.0:
            temp = temp_ref
    # end if

    dt = temp - temp_ref 
    
    density = rho_f * (1 + 0*c * (p - p_ref) - 0*beta * dt)

    return density


def matrix_perm(phi, ref_phi, ref_perm):
    """
    The matrix permeability
    """
    factor1 = np.power((1 - ref_phi), 2) / np.power((1 - phi), 2)
    factor2 = np.power(phi / ref_phi, 3)
    K = ref_perm * factor1 * factor2
    return K

def fracture_perm(aperture):
    """
    Cubic law for the fracture permeability
    """
    K = np.power(aperture, 2) / 12
    return K


# ---- Update params

# ---- use Reaktoro

def get_mineral_volume_from_states(chemical_solver, state):
    """Get the volumes of the individual minerals"""
    props = state.props()
    mineral_volume = []
    for phase in chemical_solver.chem_system.phases():
        if phase.name() != "AqueousPhase" and phase.name() != "GaseousPhase":
            mineral_volume.append(props.phaseProps(phase.name()).volume().val())
        # end if
    # end phase-loop

    return np.asarray(mineral_volume)

def calulate_mineral_volume(chemical_solver, mdg):
    """Calulate a mineral volume, using Reaktoro"""
    # First loop over the mdg

    grid_index = 0  # Global grid index that couple RT and PP
    cell_mineral_volume = np.zeros(
        mdg.num_subdomain_cells()
    )  # Cell-wise mineral volume

    for sd, d in mdg.subdomains(return_data=True):

        # Loop over the individual cell in the subdomain
        for index in range(sd.num_cells):

            mesh_point_index = index + grid_index

            tot_mineral_volumes = get_mineral_volume_from_states(
                chemical_solver, chemical_solver.states[mesh_point_index]
            )

            cell_mineral_volume[mesh_point_index] = np.sum(tot_mineral_volumes)

        # end index-loop
        grid_index += sd.num_cells
    # end mdg-loop

    return cell_mineral_volume


def update_from_reaktoro(chemical_solver, mdg):
    """Calculate grid parameters using reaktoro"""

    # Global vector
    cell_mineral_volume = calulate_mineral_volume(chemical_solver, mdg)
    global_index = 0
    
    for sd, d in mdg.subdomains(return_data=True):

        # Check that porosity and aperture exist as a variable in d
        if "porosity" not in d[pp.STATE][pp.ITERATE]:
            d[pp.STATE][pp.ITERATE].update({"porosity": np.zeros(sd.num_cells)})
        # end if
        if "aperture" not in d[pp.STATE][pp.ITERATE]:
            d[pp.STATE][pp.ITERATE].update(
                {
                    "aperture": np.zeros(sd.num_cells),
                }
            )
        # end if

        for subdomain_index in range(sd.num_cells):
            mesh_point_index = global_index + subdomain_index

            # The porosity
            calulate_porosity(
                cell_mineral_volume,
                d,
                mesh_point_index,
                subdomain_index,
                chemical_solver.states[mesh_point_index].props(),
            )

        # end index-loop
        
        # Calulate apertute
        calulate_aperture(mdg, sd, d)
            
        # Finally the permeability
        calulate_perm(mdg, sd, d)

        global_index += sd.num_cells
    # end mdg-loop

    # Update in parameter-dictionaries, for re-discretisation
    update_grid_params(mdg)

    # re-scale aqueous phase with new porosity
    chemical_solver.scale_aqueous_phases_in_states(mdg)

    chemical_solver.scale_mineral_phases_in_states()


def calulate_porosity(
    cell_mineral_volume, d, mesh_point_index, subdomain_index, chemical_props
):

    d[pp.STATE][pp.ITERATE]["porosity"][subdomain_index] = (
        1
        - cell_mineral_volume[mesh_point_index] / chemical_props.volume().val()
        - 0 / chemical_props.volume().val()  # non-reactive mineral
    )


def calulate_aperture(mdg, sd, d):

    # The aperture
    if sd.dim == mdg.dim_max():
        d[pp.STATE][pp.ITERATE].update(
            {
                "aperture": np.ones(sd.num_cells),
                #"specific_volume": np.ones(sd.num_cells),
            }
        )
    elif sd.dim == mdg.dim_max() - 1:
      
        aperture = (
            constant_params.open_aperture() * d[pp.STATE][pp.ITERATE]["porosity"] 
            )
              
        d[pp.STATE][pp.ITERATE]["aperture"]= aperture

    else:  # sd.dim < mdg.dim_max() -1
        aperture = update_intersection_aperture(mdg, sd)
        d[pp.STATE][pp.ITERATE].update({"aperture": aperture})
    # end if-else

def calulate_perm(mdg, sd, d):

    if sd.dim == mdg.dim_max():
        ref_phi = d[pp.PARAMETERS]["reference"]["porosity"]
        ref_perm = d[pp.PARAMETERS]["reference"]["permeability"]
        phi = d[pp.STATE][pp.ITERATE]["porosity"]
        perm = matrix_perm(phi, ref_phi, ref_perm)
    else:
        aperture = d[pp.STATE][pp.ITERATE]["aperture"]
        perm = fracture_perm(aperture)
    # end if-else

    d[pp.STATE][pp.ITERATE].update({"permeability": perm})

# ---- use PorePy

def update_porosity(mdg):
    """
    Update the mass weights in the transport equations:
    For the soltue transport this is the porosity, while for the 
    temeprature equation, it is the heat capacity
        
    The porosity is updated as "porosity = 1 - sum_m x_m",
    where x_m is the mth volume mineral fraction 
    
    The heat capacity is updated using the new porosity 
    and fluid density
    """
    for sd,d in mdg.subdomains(return_data=True):
        
        # The mineral concentration
        conc_CaCO3 = d[pp.STATE][pp.ITERATE]["CaCO3"]
        conc_CaSO4 = d[pp.STATE][pp.ITERATE]["CaSO4"]
        
        # A more sensible approach is to get the concentration by 
        # conc = d[pp.STATE][pp.ITERATE][mineral], and decompose cell-wise by
        # min_in_cell = conc / g.num_cells
        
        # Number of moles
        mol_CaCO3 = conc_CaCO3 * sd.cell_volumes 
        mol_CaSO4 = conc_CaSO4 * sd.cell_volumes 
        
        # Next convert mol to g
        mass_CaCO3 = mol_CaCO3 * constant_params.molar_mass_CaCO3()
        mass_CaSO4 = mol_CaSO4 * constant_params.molar_mass_CaSO4() 
     
        # the mineral volumes                                            
        mineral_vol_CaCO3 = mass_CaCO3 / constant_params.density_CaCO3() 
        mineral_vol_CaSO4 = mass_CaSO4 / constant_params.density_CaSO4() 
        
        # Porosity
        porosity = 1 - (mineral_vol_CaCO3 + mineral_vol_CaSO4) / sd.cell_volumes
      
        # Include the non-reactive porosity
        if mdg.dim_max() == sd.dim:
            porosity -=  d[pp.PARAMETERS]["reference"]["non_reactive_volume_frac"]
        # end if
        
        d[pp.STATE].update({"porosity": porosity})
        d[pp.STATE][pp.ITERATE].update({"porosity": porosity})
    # end sd, d-loop

def update_aperture(mdg):
    """
    Update the aperture in the fracture (D-2). The (D-2) and (D-3) 
    cases are handled by in update_intersection_aperture
    """
    for sd, d in mdg.subdomains(return_data=True):
        
        # Only update the aperture in the fracture
        if mdg.dim_max()-1 == sd.dim :
            ref_param = d[pp.PARAMETERS]["reference"]
         
            # the reactive surface area
            S = ref_param["surface_area"]    
            
            conc_CaCO3 = d[pp.STATE][pp.ITERATE]["CaCO3"]
            conc_CaSO4 = d[pp.STATE][pp.ITERATE]["CaSO4"]

            # Number of moles
            mol_CaCO3 = conc_CaCO3 * sd.cell_volumes 
            mol_CaSO4 = conc_CaSO4 * sd.cell_volumes 
        
            # Next convert mol to g
            mass_CaCO3 = mol_CaCO3 * constant_params.molar_mass_CaCO3() 
            mass_CaSO4 = mol_CaSO4 * constant_params.molar_mass_CaSO4() 
                                                   
            # the mineral volumes                                            
            mineral_vol_CaCO3 = mass_CaCO3 / constant_params.density_CaCO3() 
            mineral_vol_CaSO4 = mass_CaSO4 / constant_params.density_CaSO4() 
            
            mineral_width_CaCO3 = np.zeros(S.size)
            mineral_width_CaSO4 = mineral_width_CaCO3.copy()
            
            mineral_width_CaCO3 = mineral_vol_CaCO3 / S
            mineral_width_CaSO4 = mineral_vol_CaSO4 / S

            #initial_aperture = d[pp.PARAMETERS]["mass"]["initial_aperture"]
            #open_aperture = d[pp.PARAMETERS]["mass"]["open_aperture"]
            
            aperture = ( 
                constant_params.open_aperture()- mineral_width_CaCO3 - mineral_width_CaSO4
                )
            
            # Clip to avoid exactly zero aperture
            aperture = np.clip(aperture, 
                               a_min=1e-7, 
                               a_max=constant_params.open_aperture())
            
            d[pp.PARAMETERS]["mass"].update({
                "aperture": aperture.copy(),
                "specific_volume": aperture.copy() 
                })
            
            d[pp.STATE][pp.ITERATE].update({"aperture": aperture})
            d[pp.STATE].update({
                "aperture_difference": ( 
                    constant_params.open_aperture() - aperture.copy() 
                    ),
                "aperture": aperture
                })
        # end if
    # end sd,d-loop

def update_grid_perm(mdg):
    """
    Calculate perm
    """
    for sd, d in mdg.subdomains(return_data=True):
        #print(sd.dim)
        if sd.dim == mdg.dim_max():
            ref_phi = d[pp.PARAMETERS]["reference"]["porosity"]
            ref_perm = d[pp.PARAMETERS]["reference"]["permeability"]
            phi = d[pp.STATE][pp.ITERATE]["porosity"]
            perm = matrix_perm(phi, ref_phi, ref_perm)
        else:
            aperture = d[pp.STATE][pp.ITERATE]["aperture"]
            perm = fracture_perm(aperture)
        # end if-else

        d[pp.STATE][pp.ITERATE]["permeability"] = perm
    # end mdg-loop

def update_equil_const(mdg, temp_dependent=False):
    """Update equilibrium constants for temperture-dependent problem"""
    
    if temp_dependent is True: # The problem is temperature-dependent
        
        delta_H = constant_params.standard_enthalpy()     
        vec = np.zeros(mdg.num_subdomain_cells() * delta_H.size)  
        ref_temp = constant_params.ref_temp()
        
        # Idea: Loop over the gb, and fill in cell-wise. Then return as a sparse matrix.

        val = 0
        for sd, d in mdg.subdomains(return_data=True):
            #breakpoint()
            if sd.dim == mdg.dim_max():
                data = d
            # end if
            
            # Cell-wise enthalpy energy for the calculations below
            cell_delta_H = np.tile(delta_H, sd.num_cells)
        
            # The temperature
            t = d[pp.STATE][pp.ITERATE]["temperature"]
                
            # Expand the temperature for the calculation
            temp =  np.repeat(t, repeats=delta_H.shape)
          
            # Temperature factor
            dt = temp - ref_temp
            # np.clip(temp-ref_temp, a_min=-ref_temp/6, a_max=ref_temp/6) 
            
            ref_eq = constant_params.ref_equil() #d[pp.PARAMETERS]["reference"]["equil_consts"] 
            if ref_eq.size != cell_delta_H.size: # safty if shapes (size) are different
                ref_eq = np.tile(ref_eq, sd.num_cells)
            # end
            
            # Taylor approximation for stability purposes
            taylor = 1 + cell_delta_H * dt / ref_temp ** 2
            
            # Calulate the equilibrium constants
            cell_equil_consts = ref_eq * taylor
            
            # Return
            inds = slice(val, val+sd.num_cells*delta_H.size) 
            vec[inds] = cell_equil_consts
            
            # For the next grid
            val += sd.num_cells * delta_H.size
        # end g,d-loop
            
        # Splitt the equilibrium constants for components 
        # and precipitation species
        x1 = np.zeros(mdg.num_subdomain_cells() * 3, dtype=int)
        x2 = np.zeros(mdg.num_subdomain_cells() * 2, dtype=int)
        for i in range(mdg.num_subdomain_cells()):
            j = i * 5
            inds1 = np.array([j, j+1, j+2]) 
            inds2 = np.array([j+3, j+4])
            
            x1[3*i:3*i+3] = inds1
            x2[2*i:2*i+2] = inds2
        # end i-loop
        
        cell_equil_comp = vec[x1]
        cell_equil_prec = vec[x2]
        
        data["parameters"]["chemistry"].update({
            "cell_equilibrium_constants_comp": sps.diags(cell_equil_comp),
            "cell_equilibrium_constants_prec": sps.diags(cell_equil_prec)
            })
            
def update_using_porepy(mdg):
    """Update aperture, porosity, perm in within the PorePy-framework
    """
    
    # Aperture
    for sd,d in mdg.subdomains(return_data=True):
        if sd.dim == mdg.dim_max()-1:
            update_aperture(mdg)
        elif sd.dim < mdg.dim_max() - 1:
            update_intersection_aperture(mdg, sd)
    # end if
    
    # Porosity
    update_porosity(mdg)
    
    # Permeability
    update_grid_perm(mdg)
    
    # For updated disctetisation
    update_grid_params(mdg)
    
    # equilibrium constants
    update_equil_const(mdg, temp_dependent=True)
    
# ---- for both Reaktoro and PorePy

def update_perm(mdg):
    """
    Update the permeability in the matrix and fracture
    """
    
    for sd, d in mdg.subdomains(return_data=True):
            
        specific_volume = specific_vol(mdg, sd)

        ref_perm = d[pp.PARAMETERS]["reference"][
            "permeability"
        ]  # Not scaled by aperture

        K = d[pp.STATE][pp.ITERATE]["permeability"]  # not scaled by aperture

        d[pp.PARAMETERS]["flow"].update(
            {
                "permeability": K
                * specific_volume
                / constant_params.dynamic_viscosity(),
                "second_order_tensor": pp.SecondOrderTensor(
                    K * specific_volume / constant_params.dynamic_viscosity()
                ),
            }
        )
        
        # We are also interested in the current permeability,
        # compared to the initial one
        d[pp.STATE].update({"ratio_perm": K / ref_perm})

    # end sd,d-loop


def update_mass_weight(mdg, iterate=True):
    """
    Update parameters that depend on porosity:

    """
    for sd, d in mdg.subdomains(return_data=True):

        specific_volume = specific_vol(mdg, sd)

        porosity = d[pp.STATE][pp.ITERATE]["porosity"]
        d[pp.PARAMETERS]["mass"].update(
            {
                "porosity": porosity.copy(),
                "mass_weight": specific_volume.copy(),
            }
        )

        d[pp.PARAMETERS]["flow"].update(
            {"mass_weight": porosity.copy() * specific_volume.copy()}
        )

        d[pp.PARAMETERS]["passive_tracer"].update(
            {"mass_weight": porosity.copy() * specific_volume.copy()}
        )

        d[pp.STATE].update({"porosity": porosity})

        # Next, iterate on the heat effects
        d_temp = d[pp.PARAMETERS]["temperature"]
        specific_heat_capacity_fluid = constant_params.specific_heat_capacity_fluid()
        specific_heat_capacity_solid = constant_params.specific_heat_capacity_solid()

        solid_density = d_temp["solid_density"]
        fluid_density = rho(
            d[pp.STATE][pp.ITERATE]["pressure"], d[pp.STATE][pp.ITERATE]["temperature"]
        )

        heat_capacity = (
            # fluid part
            porosity.copy() * fluid_density * specific_heat_capacity_fluid
            # solid poart
            + (1 - porosity.copy())  
            * solid_density
            * specific_heat_capacity_solid  
        )
        conduction = (
            porosity.copy() * constant_params.fluid_conduction()
            + (1 - porosity.copy()) * constant_params.solid_conduction()
        )

        d_temp.update(
            {
                "mass_weight": heat_capacity * specific_volume.copy(),
                "second_order_tensor": pp.SecondOrderTensor(
                    conduction * specific_volume.copy()
                ),
            }
        )
    # end sd, d-loop


def update_interface(mdg):
    """
    Update the interfacial permeability,
    
    """
    #sds = mdg.subdomains(dim=1)[-1] 
    for e, d in mdg.interfaces(return_data=True):

        # Higher- and lower-dimensional between the interface e
        sd_h, sd_l = mdg.interface_to_subdomain_pair(e)

        # Aperture in lower-dimensional
        data_l = mdg.subdomain_data(sd_l)
        aperture = data_l[pp.PARAMETERS]["mass"]["aperture"]
        #print(aperture)
        Vl = specific_vol(mdg, sd_l)
        Vh = specific_vol(mdg, sd_h)

        # Assume the normal and tangential permeability are equal
        # in the fracture
        #ks = data_l[pp.PARAMETERS]["flow"]["second_order_tensor"].values[0][0]
        ks = data_l[pp.PARAMETERS]["flow"]["permeability"]
        tr = np.abs(sd_h.cell_faces)
        Vj = e.primary_to_mortar_int() * tr * Vh
        
        # The normal diffusivity
        # if aperture[0] == 1e-15:
        #     nd = 1e-30 * np.ones(Vj.size)
        # else:
        nd = e.secondary_to_mortar_int() * np.divide(ks, aperture * Vl / 2) * Vj
        q = (
            2
            * constant_params.fluid_conduction()
            / (e.secondary_to_mortar_int() * aperture)
        )
        #print(nd)
        d[pp.PARAMETERS]["flow"].update({"normal_diffusivity": nd})
        d[pp.PARAMETERS]["temperature"].update({"normal_diffusivity": q})
    # end e, d-loop

def specific_vol(mdg, sd):
    return mdg.subdomain_data(sd)[pp.PARAMETERS]["mass"]["specific_volume"]

def update_intersection_aperture(mdg, sd, return_val=True):
    """
    calulcate aperture intersection points (i.e. 0D for 2D matrix, or 1 and 0D for 3D matrix)
    gb : pp.GRID Bucket
    sd : intersection grid

    """

    if sd.dim > mdg.dim_max() - 2:
        raise ValueError("The grid must be an intersection domain")
    # end if

    parent_aperture = []
    num_parent = []

    for sd_h in mdg.neighboring_subdomains(sd):

        # Get the higher dimensional aperture
        dh = mdg.subdomain_data(sd_h)
        a = dh[pp.PARAMETERS]["mass"]["aperture"]
        # ah = np.abs(gh.cell_faces) * a

        # The interface between sd_h and sd
        e = mdg.subdomain_pair_to_interface((sd_h, sd))
        # mg = mdg.interface_data(e)["mortar_grid"]

        # To projection the aperture
        projection = (
            e.mortar_to_secondary_avg()
            * e.primary_to_mortar_avg()
            * np.abs(sd_h.cell_faces)
        )

        # The projected aperture
        al = projection * a

        parent_aperture.append(al)
        num_parent.append(np.sum(e.mortar_to_secondary_int().A, axis=1))

    # end edge-loop

    parent_aperture = np.array(parent_aperture)
    num_parent = np.sum(np.array(num_parent), axis=0)

    aperture = np.sum(parent_aperture, axis=0) / num_parent
    
    if return_val == False:
        d = mdg.subdomain_data(sd)
        d[pp.STATE][pp.ITERATE].update({"aperture": aperture})
    else:
        return aperture

def aperture_state_to_param(mdg):

    for sd, d in mdg.subdomains(return_data=True):
        d_mass = d[pp.PARAMETERS]["mass"]
        if sd.dim < mdg.dim_max():

            aperture = np.clip(
                d[pp.STATE][pp.ITERATE]["aperture"].copy(),
                1e-8,
                constant_params.open_aperture(),
            )
            d_mass.update(
                {
                    "aperture": aperture.copy(),
                    "specific_volume": np.power(aperture, mdg.dim_max() - sd.dim),
                }
            )

            d[pp.STATE].update(
                {
                    "aperture": aperture.copy(),
                    "aperture_difference": constant_params.open_aperture()
                    - aperture.copy(),
                }
            )
        else:

            d_mass.update(
                {
                    "aperture": np.ones(sd.num_cells),
                    "specific_volume": np.ones(sd.num_cells),
                }
            )
            d[pp.STATE].update(
                {
                    "aperture": np.ones(sd.num_cells),
                    "aperture_difference": np.zeros(sd.num_cells),
                }
            )

    # end sd, d-loop


def update_grid_params(mdg):
    """
    Update the aperture, porosity and permeability
    """

    aperture_state_to_param(mdg)

    # Update porosity
    update_mass_weight(mdg)

    # Update permeability
    update_perm(mdg)

    # Update the interface properties
    update_interface(mdg)

def update_darcy(equation_system):
    """
    Update the darcy flux in the parameter dictionaries

    """
    
    mdg = equation_system.mdg

    # Get the Ad-fluxes
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    data = mdg.subdomain_data(sd)
    num_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_full_flux"]

    # Convert to numerical values, if necessary
    if isinstance(num_flux, (pp.ad.AdArray, pp.ad.Operator)):
        num_flux = num_flux.evaluate(equation_system)
        if hasattr(num_flux, "val"):
            num_flux = num_flux.val
        # end if
    # end if

    # Get the signs
    sign_flux = np.sign(num_flux)
    
    # Finally, loop over the gb and return the signs of the darcy fluxes
    val = 0
    for sd, d in mdg.subdomains(return_data=True):

        inds = slice(val, val + sd.num_faces)
        x = sign_flux[inds]

        # For the solute transport we only need the signs
        if "transport" in d[pp.PARAMETERS]:
            d[pp.PARAMETERS]["transport"]["darcy_flux"] = x.copy()
        if "temperature" in d[pp.PARAMETERS]:
            d[pp.PARAMETERS]["temperature"]["darcy_flux"] = x.copy()
        if "passive_tracer" in d[pp.PARAMETERS]:
            d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = x.copy()
            

        # For the flow, we need the absolute value of the signs
        if "flow" in d[pp.PARAMETERS]:
            d[pp.PARAMETERS]["flow"]["darcy_flux"] = np.abs(x.copy())
        # end ifs
        val += sd.num_faces
    # end sd, d-loop

    # Do the same over the interfaces
    if mdg.dim_max() > mdg.dim_min():

        # The flux
        edge_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_edge_flux"]
        num_edge_flux = edge_flux.evaluate(equation_system).val

        val = 0
        for e, d in mdg.interfaces(return_data=True):
            nc = e.num_cells
            inds = slice(val, val + nc)
            y = num_edge_flux[inds]  # sign_edge_flux[inds]

            if "transport" in d[pp.PARAMETERS]:
                d[pp.PARAMETERS]["transport"]["darcy_flux"] = y.copy()
            if "temperature" in d[pp.PARAMETERS]:
                d[pp.PARAMETERS]["temperature"]["darcy_flux"] = y.copy()
            if "flow" in d[pp.PARAMETERS]:
                d[pp.PARAMETERS]["flow"]["darcy_flux"] = y.copy()
            if "passive_tracer" in d[pp.PARAMETERS]:
                d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = y.copy()
            # end ifs
            val += nc
        # end e, d-loop


def update_concentrations(mdg, equation_system):
    """
    Update concenctations

    This function can most likely be deleted. If not,it requires a cleanup wrt.
    returning chemical names an values.

    """
    # dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    # x = dof_manager.assemble_variable(from_iterate=True)

    # First get the concentations

    minerals = equation_system.get_variable_values(["minerals"], from_iterate=False)

    # The components
    log_component = equation_system.get_variable_values(
        ["aq_amount"], from_iterate=False
    )

    # calculate the secondary species
    data_chemistry = mdg.subdomain_data(mdg.subdomains(dim=2)[0])[pp.PARAMETERS][
        "chemistry"
    ]
    S = data_chemistry["cell_stoic_coeff_S"]
    equilibrium_const = data_chemistry["cell_equilibrium_constants_comp"].diagonal()
    secondary_species = equilibrium_const * np.exp(S * log_component)
    # end if
    
    #breakpoint()

    start_pt = 0 # for indexing 

    # Loop over the subdomain and return values
    for sd, d in mdg.subdomains(return_data=True):
        # inds1 = slice(start_pt*2, (start_pt*2+2*sd.num_cells))
        # inds2 = slice(start_pt*4, (start_pt*4+4*sd.num_cells))
        # inds3 = slice(start_pt*3, (start_pt*3+3*sd.num_cells))

        # print(inds1)
        # print(inds2)
        # print(inds3)        
        mineral_in_subdomain = minerals[
            start_pt * 2 : start_pt * 2+ 2 * sd.num_cells
            ]
        components_in_subdomain = np.exp(log_component[
            start_pt * 4 : start_pt * 4 + 4 * sd.num_cells
            ])
        secondary_in_subdomain = secondary_species[
            start_pt * 3 : start_pt * 3 + 3 * sd.num_cells
            ]
        
        x =  {
                "CaCO3": mineral_in_subdomain[0::2],
                "CaSO4": mineral_in_subdomain[1::2],
                # Components
                "Ca2+": components_in_subdomain[0::4],
                "CO3": components_in_subdomain[1::4],
                "SO4": components_in_subdomain[2::4],
                "H+": components_in_subdomain[3::4],
                # Secondary
                "HCO3": secondary_in_subdomain[0::3],
                "HSO4": secondary_in_subdomain[1::3],
                "OH-": secondary_in_subdomain[2::3],
            }
        
        d[pp.STATE].update(x)
        d[pp.STATE][pp.ITERATE].update(x)
        
        start_pt += sd.num_cells
    # end sd, d-loop

# End function
