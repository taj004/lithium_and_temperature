import numpy as np
import porepy as pp
import scipy.sparse as sps

from update_param import rho, update_darcy
import rt_utills as utills

import constant_params


#%% Injection/production well coordinates 
def injection_point(mdg, for_1d_analysis=False):
    "Get injection well coordinate"
    
    # Only well sourcer in the matrix
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    
   
    if sd.dim == 2:
        # Injection and point
        injection_1 = sd.closest_cell(
            p = np.reshape([1000, 1000, 0], (3,1))
            )
    elif sd.dim == 3:
        
        domain = mdg.subdomains(dim=2)[13]

        # Injection at fracture
        if len(mdg.subdomains(dim=2)) == 39:
            cells = domain.cell_centers[:, 74] 
        elif len(mdg.subdomains(dim=2)) == 40:
            cells = domain.cell_centers[:, 62]
        # end if-else
        
        injection_1 = domain.closest_cell(
            p = np.reshape(cells, (3,1))
            )  
        
        if for_1d_analysis:
            return domain, injection_1
        # end if
        
        global_ind = 0 
        
        for sdd in mdg.subdomains(): 
            if domain == sdd: break           
            else: global_ind += sdd.num_cells

        # end loop
        
        injection_1 += global_ind # adjust index for the global vector
        
    # end if
    
    return np.array([
        injection_1
        ])

def production_point(mdg, for_1d_analysis=False):
    "Get production well coordinate"
    
    # Only well sourcer in the matrix
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    
    if sd.dim == 2:
        # Production 
        production_1 = sd.closest_cell(
            p = np.reshape([2000, 1000, 0], (3,1))
            ) 
   
    elif sd.dim == 3:
        
        domain = mdg.subdomains(dim=2)[34]
        
        if len(mdg.subdomains(dim=2)) == 39:
            cells = domain.cell_centers[:, 446] 
        elif len(mdg.subdomains(dim=2)) == 40:
            cells = domain.cell_centers[:, 443]
        # end if
        
        production_1 = domain.closest_cell(
            p = np.reshape(cells, (3,1))
            ) 
        
        if for_1d_analysis:
            return domain, production_1
        # end if
        
        global_ind = 0 
        for sdd in mdg.subdomains():    
            if domain == sdd: break
            else: global_ind += sdd.num_cells
        # end loop
        
        production_1 += global_ind # For the global vector

    # end if
    
    return np.array([
        production_1
        ])

def source_rate(mdg: pp.MixedDimensionalGrid):
    """Set the external well source rate for the different equations"""
    
    source_in = np.zeros(mdg.num_subdomain_cells())
    source_out = np.zeros(mdg.num_subdomain_cells())
    
    # Injection/production flow rate
    if mdg.dim_max()==2:
        rate = 4e-1 * 1e-3 # unit m^3/ s ; unit(L) = 1e-3 unit(m3)
    elif mdg.dim_max()==3:
        rate = 30 * 1e-3 # m3/s
    # end if
    
    if mdg.dim_max() == 2: 

        injection = injection_point(mdg)
        production = production_point(mdg)
        
        source_in[injection] = rate
        source_out[production] = -rate 
          
    elif mdg.dim_max() == 3:
          
        injection = injection_point(mdg)
        production = production_point(mdg)
        
        source_in[injection] = rate 
        source_out[production] = -rate 
      
    # end if   
    
    # Finally, scale the source term by specific volume    
    sd_ind = 0
    for sd, d in mdg.subdomains(return_data=True):
        sv = d[pp.PARAMETERS]["mass"]["specific_volume"]
      
        xx = slice(sd_ind, sd_ind + sd.num_cells)
        source_in[xx] *= sv
        source_out[xx] *= sv    
        sd_ind += sd.num_cells
    # end sd-d loop
    
    return pp.ad.DenseArray(source_in), pp.ad.DenseArray(source_out)

#%% Main class
class EquationConstruction:

    def __init__(
            self, 
            equation_system, 
            well_source=False
            ):
        
        self.equation_system = equation_system
        
        # -- Keywords -- #

        # Parameter keywords
        self.mass_kw = "mass"
        self.flow_kw = "flow"
        self.transport_kw = "transport"
        self.chemistry_kw = "chemistry"
        self.temperature_kw = "temperature"
        self.tracer_kw = "passive_tracer"
        
        # Variable keyword:
        pressure = "pressure"
        tot_var = "total_amount"
        aq_var = "aq_amount"
        temperature = "temperature"
        minerals = "minerals"
        tracer = "passive_tracer"

        mortar_pressure = "mortar_pressure"
        mortar_tot_var = "mortar_transport"
        mortar_temperature_convection = "mortar_temperature_convection"
        mortar_temperature_conduction = "mortar_temperature_conduction"
        mortar_tracer = "mortar_tracer"

        # -- Mixed-dimensional grid related -- #

        # self.equation_system = equation_system
        mdg = equation_system.mdg

        # Get data that are necessary for later use
        d = mdg.subdomains(mdg.dim_max())[0][1]

        d_primary_vars = d["primary_variables"]

        # The number of components of different chemical aspects
        aq_components = d[pp.PARAMETERS][self.transport_kw].get(
            "aqueous_components", np.ones(1)
            )
        num_aq_components = d[pp.PARAMETERS][self.transport_kw].get(
            "num_components", 1
            )
        num_components = d[pp.PARAMETERS][self.mass_kw].get(
            "num_components", 1
            )

        #---- Ad representations of the variables -- #
        
        # Set subdomain variabels and equation names
        
        # Check which equations should be constructed
        
        self.eq_names = []
        self.ad_vars = []
        
        if pressure in d_primary_vars.keys():
            self.ad_var_p = equation_system.create_variables(
                name=pressure, 
                dof_info={"cells":  d_primary_vars[pressure]["cells"]}, 
                subdomains=mdg.subdomains()
                )  
            
            # To fix the `defined outside init'-code analysis warning 
            self.full_flux = "init_flux"
            
            # BC
            self.bound_flux = pp.ad.BoundaryCondition(
            keyword=self.flow_kw, subdomains=mdg.subdomains()
            )            
            
            self.eq_name_fluid_conservation = "fluid_conservation"
            self.eq_name_incompressible_fluid = "incompressible_fluid"
        # end if
        
        if tot_var in d_primary_vars.keys():
            self.ad_var_tot_U = equation_system.create_variables(
                name=tot_var,
                dof_info={"cells": d_primary_vars[tot_var]["cells"]},
                subdomains=mdg.subdomains(),
                )        
                        
            self.eq_name_solute_transport_TC = "transport"
        # end if    
        
        if aq_var in d_primary_vars.keys():
            self.ad_var_aq = equation_system.create_variables(
                name=aq_var,
                dof_info={"cells": d_primary_vars[aq_var]["cells"]},
                subdomains=mdg.subdomains(),
                )   
            
            # BC
            self.bound_transport = pp.ad.BoundaryCondition(
            keyword=self.transport_kw, subdomains=mdg.subdomains()
            )   
            
            self.eq_name_solute_transport_CC = "transport"  
        
            # For scaling
            self.porosity_scale = 1.0
        # end if    

        if minerals in d_primary_vars.keys():
            self.ad_var_precipitate = equation_system.create_variables(
                name=minerals, 
                dof_info={"cells": d_primary_vars[minerals]["cells"] }, 
                subdomains=mdg.subdomains()
                )     
            
            self.eq_name_equilibrium = "equlibrium"
            self.eq_name_dissolution_precipitation = "dissolution_and_precipitation"
        # end if    

        if temperature in d_primary_vars.keys():
            self.ad_var_temp = equation_system.create_variables(
                name=temperature,
                dof_info={"cells": d_primary_vars[temperature]["cells"]},
                subdomains=mdg.subdomains(),
            )      
            
            # BC
            self.bound_temp = pp.ad.BoundaryCondition(
            keyword=temperature, subdomains=mdg.subdomains()
            )   
            self.eq_name_temperature = "temperature"
        # end if    

        if tracer in d_primary_vars.keys():
            self.ad_var_passive_tracer = equation_system.create_variables(
                name=tracer, 
                dof_info={"cells": d_primary_vars[tracer]["cells"]}, 
                subdomains=mdg.subdomains()        
                )    
            
            # BC
            self.bound_tracer = pp.ad.BoundaryCondition(
            keyword=tracer, subdomains=mdg.subdomains()
            )        
            self.eq_name_passive_tracer = "passive_tracer"
        # end if    

        # Fluid density related:
        rho_ad = pp.ad.Function(rho, name="density")
        
        if pressure in d_primary_vars.keys():
            if temperature in d_primary_vars.keys(): # Both pressure and temperature dependent
                self.bound_rho = rho_ad(self.bound_flux, self.bound_temp)
            else: # Only pressure dependent
                self.bound_rho = rho_ad(self.bound_flux)
            # end if
        # end-if
        if (temperature in d_primary_vars.keys() and 
            pressure not in d_primary_vars.keys() ): # depend only on temperature 
            self.bound_rho = rho_ad(0, self.bound_temp) 
            # Use 0 as default, since ad-functions don't accept None,
        # end if
        
        # Do the same for the fluxes over the interfaces
        if len(mdg.interfaces()) > 0:
            
            d_primary_vars = mdg.interface_data(
                mdg.interfaces(dim=mdg.dim_max()-1)[0]
                )[pp.PRIMARY_VARIABLES]
            
            # Flow
            if mortar_pressure in d_primary_vars.keys():
                self.ad_var_v = equation_system.create_variables(
                    name=mortar_pressure,
                    dof_info={
                        "cells": d_primary_vars[mortar_pressure]["cells"]
                              },
                    interfaces=mdg.interfaces(),
                )
                
                self.interface_flux = np.zeros(mdg.num_interface_cells())            
                
                self.eq_name_interface_flux = "interface_flux"
                self.eq_name_interface_incompressible = "interface_flux"
            # end if    

            # Transport
            if mortar_tot_var in d_primary_vars.keys():
                self.ad_var_zeta = equation_system.create_variables(
                    name=mortar_tot_var,
                    dof_info={"cells": d_primary_vars[mortar_tot_var]["cells"]},
                    interfaces=mdg.interfaces(),
                )           
                
                self.eq_name_transport_over_interface = "interface_transport"
            # end if    

            # Temperature
            if mortar_temperature_convection in d_primary_vars.keys():
                self.ad_var_w = equation_system.create_variables(
                    name=mortar_temperature_convection,
                    dof_info={
                        "cells": d_primary_vars[mortar_temperature_convection]["cells"]
                              },
                    interfaces=mdg.interfaces(),
                )
                self.ad_var_q = equation_system.create_variables(
                    name=mortar_temperature_conduction,
                    dof_info={
                        "cells": d_primary_vars[mortar_temperature_conduction]["cells"]
                              },
                    interfaces=mdg.interfaces(),
                )   
                self.eq_name_interface_convection = "interface_convection"
                self.eq_name_interface_conduction = "interface_conduction"
            # end if    

            # Passive tracer
            if mortar_tracer in d_primary_vars.keys():
                self.ad_var_zeta_tracer = equation_system.create_variables(
                    name=mortar_tracer,
                    dof_info={
                        "cells": d_primary_vars[mortar_tracer]["cells"]
                              },
                    interfaces=mdg.interfaces(),
                )

                self.eq_name_tracer_interface_flux = "tracer_over_interface"
            # end if    

        # end if
        
        #---- Constant grid related parameters -- #
        
        self.mortar_projection_single = pp.ad.MortarProjections(
            mdg=mdg,
            subdomains=mdg.subdomains(),
            interfaces=mdg.interfaces(),
            dim=1,
        )
        
        self.mortar_projection_several = pp.ad.MortarProjections(
            mdg=mdg,
            subdomains=mdg.subdomains(),
            interfaces=mdg.interfaces(),
            dim=num_components,
        )
        
        self.trace_single = pp.ad.Trace(subdomains=mdg.subdomains(), dim=1)
        
        self.trace_several = pp.ad.Trace(
            subdomains=mdg.subdomains(), dim=num_components
        )
        
        self.divergence_single = pp.ad.Divergence(
            subdomains=mdg.subdomains(), dim=1
            )
        self.divergence_several = pp.ad.Divergence(
            subdomains=mdg.subdomains(), dim= num_aq_components
        )
        
        # If a problem with well injection sources
        if well_source:
            self.injection_rate, self.production_rate = source_rate(mdg)
        # end if
        
        # Mapping between mobile and immoblie chemical species
        # A sensibility check to verify that chemistry is
        # in the system
        if tot_var in d_primary_vars.keys():
            self.all_2_aquatic = utills.all_2_aquatic_mat(
                aq_components, 
                num_components, 
                num_aq_components,
                mdg.num_subdomain_cells()
            )
        
            if len(mdg.interfaces()):
                self.all_2_aquatic_at_interface = utills.all_2_aquatic_mat(
                    aq_components,
                    num_components,
                    num_aq_components,
                    mdg.num_interface_cells(),
                )
            
            #---- Mapping to alter the number of factors per 
            # grid element, e.g., the flux per faces
                
            self._extend_source_for_solute_transport = utills.enlarge_mat_2(
                mdg.num_subdomain_cells(), num_aq_components * mdg.num_subdomain_cells()  
                )
            
            self.extend_flux = utills.enlarge_mat_2(
                mdg.num_subdomain_faces(), num_aq_components * mdg.num_subdomain_faces()
                )
            
            self._porosity_scale_mat = utills.enlarge_mat_2(
                mdg.num_subdomain_cells(), num_aq_components * mdg.num_subdomain_cells()
                )
        
            if len(mdg.interfaces()):
                self.extend_edge_flux = utills.enlarge_mat_2(
                    mdg.num_interface_cells(), num_aq_components * mdg.num_interface_cells()
                    )
            # end if
        # end if
    
    def injection_value(self, kw="flow"):
        "The well injection rate for the source term"
                
        # The mdg
        mdg = self.equation_system.mdg
        
        # Source term
        S = np.zeros(mdg.num_subdomain_cells()) 

        # Injection point
        injection = injection_point(mdg)
        
        sd = mdg.subdomains(dim=mdg.dim_max())[0] 
        d = mdg.subdomain_data(sd) 
        param = d[pp.PARAMETERS]
        
        injection_kw = "injection_value"
        # For compressible flow, scale by density
        if kw == self.flow_kw:
            
            p_in = param[kw][injection_kw] # [Pa] 
            
            if self.temperature_kw in d[pp.PRIMARY_VARIABLES]:
                temp_in = param[self.temperature_kw][injection_kw]
                density_in = rho(p=p_in, temp=temp_in)
            else:
                density_in = rho(p=p_in)        
            # end if-else
            
            S[injection] = density_in
            
        elif kw == self.temperature_kw :
            temp_in = param[self.temperature_kw][injection_kw]
            S[injection] = temp_in
                
        elif kw == self.tracer_kw :
            
            num_tracer = 1
            injection=injection.reshape(injection.size)
            # expand for several components
            expanded_injection_ponit = pp.fvutils.expand_indices_nd(injection, 
                                                                    injection.size)
            # The injection scale values
            t_in = np.repeat(param[kw][injection_kw], num_tracer)
            
            S = np.tile(S, num_tracer)
            S[expanded_injection_ponit] = t_in
            
        elif kw == self.transport_kw :
                 
            val = param[self.transport_kw][injection_kw]
            injection = injection.reshape(injection.size)
   
            # expand injection points according to number of variables
            expanded_injection_point = pp.fvutils.expand_indices_nd(
                injection, val.size
                )
            
            # expand source value according to number of sources
            t_in = np.repeat(val , injection.size)

            S = np.tile(S, val.size)
            S[expanded_injection_point] = t_in
        
        # end-if
            
        return pp.ad.DenseArray(S)
    
    def get_equilibrium_eqs(self, equation_system):

        data_chemistry = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )[pp.PARAMETERS][self.chemistry_kw]

        # Cell-wise chemical values
        cell_S = data_chemistry["cell_stoic_coeff_S"]
        cell_E = data_chemistry["cell_stoic_coeff_E"]
        cell_equil_comp = data_chemistry["cell_equilibrium_constants_comp"]

        def equilibrium_all_cells(total, log_primary, precipitated_species):
            """
            Residual form of the equilibrium problem,
            posed for all cells together.
            """
            # The secondary species
            secondary_C = cell_equil_comp @ pp.ad.exp(cell_S @ log_primary)

            # Residual equation
            eq = (
                pp.ad.exp(log_primary)
                + cell_S.transpose() @ secondary_C
                + cell_E.transpose() @ precipitated_species
                - total
            )
            return eq

        # Wrap the equilibrium residual into an Ad function.
        equil_ad = pp.ad.Function(equilibrium_all_cells, "equil")

        # Finally, make it into an Expression which can be evaluated.
        equilibrium_eq = equil_ad(
            self.ad_var_tot_U, self.ad_var_aq, self.ad_var_precipitate
        )
        
        # Store information about the equations necessary for splitting methods
        if self.eq_name_equilibrium not in self.eq_names:
            self.eq_names.append(self.eq_name_equilibrium)
            self.ad_vars.append(self.ad_var_aq)
        # end if
        
        equation_system.equations.update({self.eq_name_equilibrium: equilibrium_eq})

    def get_incompressible_flow_eqs(self, equation_system, well_source=False):
        
        # Variable
        p = self.ad_var_p
        
        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )
        
        data_prev_newton = d[pp.PARAMETERS]["previous_newton_iteration"]

        # Ad wrapper of Mpfa discretization
        flow_tpfa = pp.ad.TpfaAd(self.flow_kw, subdomains=equation_system.mdg.subdomains())

        # The interior and boundary fluxes
        interior_flux = flow_tpfa.flux @ p
        boundary_flux = flow_tpfa.bound_flux @ self.bound_flux

        # The Darcy flux (in the higher dimension)
        full_flux = interior_flux + boundary_flux

        # Add the fluxes from the interface to higher dimension,
        # the source accounting for fluxes to the lower dimansions
        # and multiply by the divergence
        if len(equation_system.mdg.interfaces()) > 0:
            
            v = self.ad_var_v
            
            # A projection between subdomains and mortar grids
            mortar_projection = self.mortar_projection_single

            # Include flux from the interface
            full_flux += (
                flow_tpfa.bound_flux @ 
                mortar_projection.mortar_to_primary_int @ 
                v
                )

            # The source term for the flux to \Omega_l
            sources_from_mortar = (
                mortar_projection.mortar_to_secondary_int @ v
                )
            
            # Conservation equation
            conservation = self.divergence_single @ full_flux - sources_from_mortar

        else:
            conservation = self.divergence_single @ full_flux
        # end if

        if len(equation_system.mdg.interfaces()) > 0:
            
            pressure_trace_from_high = (
                mortar_projection.primary_to_mortar_avg @ (
                    flow_tpfa.bound_pressure_cell @ p +  
                    flow_tpfa.bound_pressure_face @ ( 
                        mortar_projection.mortar_to_primary_int @ v + 
                        self.bound_flux
                        )
                    )
                )
            
            pressure_from_low = mortar_projection.secondary_to_mortar_avg @ p

            robin = pp.ad.RobinCouplingAd(
                self.flow_kw, equation_system.mdg.interfaces()
            )

            # The interface flux
            interface_flux = v + robin.mortar_discr @ (
                pressure_trace_from_high - pressure_from_low
            )
        # end if
        
        if well_source:
            # Include possible external (well) sources
            rate_in, rate_out = source_rate(equation_system.mdg)
            
            conservation = conservation - (
                self.injection_rate + self.production_rate
                ) 
        # end if
        
        # Make equations, feed them to the AD-machinery and discretise
        if len(equation_system.mdg.interfaces()) > 0:
            # Conservation equation
            conservation.discretize(equation_system.mdg)

            # Flux over the interface
            interface_flux.discretize(equation_system.mdg)
        else:
            conservation.discretize(equation_system.mdg)
        # end if-else
    
        # Store the flux for transport calculations
        self.full_flux = full_flux
        data_prev_newton["AD_full_flux"] = full_flux

        if len(equation_system.mdg.interfaces()) > 0:
            self.interface_flux = v
            data_prev_newton.update({"AD_edge_flux": v})
        # end if
        
        # Store information about the equations necessary for splitting methods
        if self.eq_name_incompressible_fluid not in self.eq_names:
            self.eq_names.append(self.eq_name_incompressible_fluid)
            self.ad_vars.append(p)
        # end if
        
        # Give to the equation system:
        equation_system.equations.update(
            {self.eq_name_incompressible_fluid: conservation}
        )

        if len(equation_system.mdg.interfaces()) > 0:
            equation_system.equations.update(
                {self.eq_name_interface_incompressible: interface_flux}
            )
            
            if self.eq_name_interface_flux not in self.eq_names:
                self.eq_names.append(self.eq_name_interface_flux)
                self.ad_vars.append(self.ad_var_v)
            # end if
        # end if
        
        # Update the Darcy flux for the transport processes
        update_darcy(equation_system)


    def get_flow_eqs(self, equation_system, iterate=False, well_source=False):

        p = self.ad_var_p  
        if hasattr(self, "ad_var_temp"):
            temp = self.ad_var_temp
        else: 
            temp = pp.ad.Scalar(0.0)
            temp_prev = pp.ad.Scalar(0.0)
        # end it

        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )
        data_transport = d[pp.PARAMETERS][self.transport_kw]
        data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
        data_prev_newton = d[pp.PARAMETERS]["previous_newton_iteration"]

        # Time step
        dt = pp.ad.Scalar( data_transport["time_step"])

        if iterate:  # Newton iterations
            mass_density = pp.ad.MassMatrixAd(
                keyword=self.flow_kw, subdomains=equation_system.mdg.subdomains()
            )

            mass_density_prev = data_prev_time[
                "mass_density_prev"
            ]  # previous time step
            p_prev = data_prev_time["p_prev"]  # previous time step
            temp_prev = data_prev_time["temp_prev"]
        else:
            # The pressure and temperature at the previous time step
            p_prev = p.previous_timestep()
            
            if isinstance(temp, type(p)): # The temperature is included in the simulation
                temp_prev = temp.previous_timestep()
            # end if
            
            # Acculmuation for the pressure equation
            mass_density = pp.ad.MassMatrixAd(
                keyword=self.flow_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_density_prev = pp.ad.MassMatrixAd(
                keyword=self.flow_kw, subdomains=equation_system.mdg.subdomains()
            )

            # Store for the Newton calculations
            data_prev_time["p_prev"] = p_prev
            data_prev_time["temp_prev"] = temp_prev
            data_prev_time["mass_density_prev"] = mass_density_prev
        # end if-else

        # Wrap the constitutive law into ad, 
        # and construct the ad-wrapper of the PDE
        rho_ad = pp.ad.Function(rho, "density")
        rho_ad_main = rho_ad(p, temp_prev) 

        # Build the flux term.
        # Remember that we need to multiply the fluxes by density.
        # The density is a cell-centerd variable,so we need to average it onto the faces
        upwind_weight = pp.ad.UpwindAd(
            keyword=self.flow_kw, subdomains=equation_system.mdg.subdomains()
        )
        
        # The density at the faces
        rho_on_face = (
            upwind_weight.upwind @ rho_ad_main
            - upwind_weight.bound_transport_dir @ self.bound_rho
            - upwind_weight.bound_transport_neu @ self.bound_rho
        )

        # Ad wrapper of TPFA/MPFA discretization
        flow_tpfa = pp.ad.TpfaAd(self.flow_kw, subdomains=equation_system.mdg.subdomains())

        # The interior and boundary fluxes
        interior_flux = flow_tpfa.flux @ p
        boundary_flux = flow_tpfa.bound_flux @ self.bound_flux

        # The Darcy flux (in the higher dimension)
        full_flux = interior_flux + boundary_flux

        # Multiply the flux and the density
        full_density_flux = rho_on_face * full_flux

        # Add the fluxes from the interface to higher dimension,
        # the source accounting for fluxes to the lower dimansions
        # and multiply by the divergence
        if len(equation_system.mdg.interfaces()) > 0:

            # A projection between subdomains and mortar grids
            #mortar_projection = self.mortar_projection_single

            # Include flux from the interface
            full_flux += flow_tpfa.bound_flux @ (
                self.mortar_projection_single.mortar_to_primary_int @ self.ad_var_v
                )

            # Tools to include the density in the source term
            upwind_coupling_weight = pp.ad.UpwindCouplingAd(
                keyword=self.flow_kw, interfaces=equation_system.mdg.interfaces()
            )
            #trace = self.trace_single

            # up_weight_flux = upwind_coupling_weight.flux
            up_weight_primary = upwind_coupling_weight.upwind_primary
            up_weight_secondary = upwind_coupling_weight.upwind_secondary

            # Map the density in the higher- and lower-dimensions
            # onto the interfaces
            high_to_low = (
                up_weight_primary
                @ self.mortar_projection_single.primary_to_mortar_avg
                @ self.trace_single.trace
                @ rho_ad_main
            )

            low_to_high = (
                up_weight_secondary
                @ self.mortar_projection_single.secondary_to_mortar_avg
                @ rho_ad_main
            )

            # The density at the interface
            mortar_density = (high_to_low + low_to_high) * self.ad_var_v

            # # Include the mortar density in the flux towards \Omega_h
            full_density_flux += (
                flow_tpfa.bound_flux
                @ self.mortar_projection_single.mortar_to_primary_int
                @ mortar_density
            )

            # The source term for the flux to \Omega_l
            sources_from_mortar = (
                self.mortar_projection_single.mortar_to_secondary_int @ mortar_density
            )
            
            # Flux and source term
            conservation = self.divergence_single @ full_density_flux - sources_from_mortar

        else:
            # Flux part
            conservation = self.divergence_single @ full_density_flux
        # end if

        # Construct the conservation equation
        density_wrap = (
            (mass_density.mass @ rho_ad_main - 
             mass_density_prev.mass @ rho_ad(p_prev, temp_prev) ) / dt
        ) + conservation 
        
         # Include possible external (well) sources
        if well_source:
            
            injection = self.injection_value(self.flow_kw)
            density_wrap = density_wrap - (
                self.injection_rate * injection + self.production_rate * rho_ad_main
                )
        # end if
        
        if len(equation_system.mdg.interfaces()) > 0:

            pressure_trace_from_high = self.mortar_projection_single.primary_to_mortar_avg @ (
                flow_tpfa.bound_pressure_cell @ p + 
                flow_tpfa.bound_pressure_face @ (
                self.mortar_projection_single.mortar_to_primary_int @ 
                self.ad_var_v + self.bound_flux 
                )
            )

            pressure_from_low = (
                self.mortar_projection_single.secondary_to_mortar_avg @ p
                )

            robin = pp.ad.RobinCouplingAd(
                self.flow_kw, equation_system.mdg.interfaces()
            )

            # The interface flux
            interface_flux = self.ad_var_v + robin.mortar_discr @ (
                pressure_trace_from_high - pressure_from_low
            )
        # end if

        # Make equations, feed them to the AD-machinery and discretize
        if len(equation_system.mdg.interfaces()) > 0:
            # Conservation equation
            density_wrap.discretize(equation_system.mdg)

            # Flux over the interface
            interface_flux.discretize(equation_system.mdg)
        else:
            density_wrap.discretize(equation_system.mdg)
        # end if-else
        
        # Store the flux for transport calculations
        self.full_flux = full_flux 
        
        data_prev_newton["AD_full_flux"] = full_flux

        if len(equation_system.mdg.interfaces()) > 0:
            self.interface_flux = self.ad_var_v 
            data_prev_newton.update({"AD_edge_flux": self.ad_var_v})
        # end if
        
        # Store information about the equations necessary for splitting methods
        if self.eq_name_fluid_conservation not in self.eq_names:
            self.eq_names.append(self.eq_name_fluid_conservation)
            self.ad_vars.append(p)
        # end if
        
        # Give to the equation system:
        equation_system.equations.update(
            {self.eq_name_fluid_conservation: density_wrap}
        )

        if len(equation_system.mdg.interfaces()) > 0:
            equation_system.equations.update(
                {self.eq_name_interface_flux: interface_flux}
            ) 
            
            if self.eq_name_interface_flux not in self.eq_names:
                self.eq_names.append(self.eq_name_interface_flux)
                self.ad_vars.append(self.ad_var_v)
            # end if
        # end if
        
    def get_temperature_eqs(self, equation_system, iterate=False, well_source=False):

        temp = self.ad_var_temp
        
        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )
        data_transport = d[pp.PARAMETERS][self.transport_kw]
        data_prev_time = d[pp.PARAMETERS]["previous_time_step"]

        dt = pp.ad.Scalar(data_transport["time_step"])
        
        # Upwind discretization for the convection
        upwind_temp = pp.ad.UpwindAd(
            keyword=self.temperature_kw, subdomains=equation_system.mdg.subdomains()
        )
        heat_capacity = pp.ad.MassMatrixAd(
            keyword=self.temperature_kw, subdomains=equation_system.mdg.subdomains()
        )
        if iterate:
            heat_capacity_prev = data_prev_time["heat_capacity_prev"]
            temp_prev = data_prev_time["temp_prev"]
            
        else:
            heat_capacity_prev = pp.ad.MassMatrixAd(
                keyword=self.temperature_kw, subdomains=equation_system.mdg.subdomains()
            )
            temp_prev = temp.previous_timestep()
            # Store for later use
            data_prev_time["temp_prev"] = temp_prev
            data_prev_time["heat_capacity_prev"] = heat_capacity_prev
        # end if-else

        # Temperature "scaled" by density and specific heat capacites
        rho_ad = pp.ad.Function(rho, "density")

        density = (
            rho_ad(self.ad_var_p.previous_iteration(), temp.previous_iteration())
            * constant_params.specific_heat_capacity_fluid()
        )

        # The boundary condition for the temperature
        bound_temp = (
            self.bound_temp
            * self.bound_rho
            * constant_params.specific_heat_capacity_fluid()
        )
      
        # Convection term
        convection = (
            (upwind_temp.upwind @ (density * temp)) * self.full_flux
            - (upwind_temp.bound_transport_dir @ self.full_flux) * bound_temp
            - upwind_temp.bound_transport_neu @ bound_temp
        )
        
        # Conduction
        conduction_tpfa = pp.ad.TpfaAd(
            self.temperature_kw, subdomains=equation_system.mdg.subdomains()
        )
        conduction = (
            conduction_tpfa.flux @ temp + conduction_tpfa.bound_flux @ self.bound_temp
        )
        
        # Cell-wise temperature equation
        temp_wrap = (
            (heat_capacity.mass @ temp - heat_capacity_prev.mass @ temp_prev) / dt 
            + self.divergence_single @ (convection + conduction) 
            )

        # Projections
        if len(equation_system.mdg.interfaces()) > 0:

            # Edge conduction
            temp_wrap += (
                self.divergence_single
                @ conduction_tpfa.bound_flux
                @ self.mortar_projection_single.mortar_to_primary_int
                @ self.ad_var_q
            )
            temp_wrap -= (
                self.mortar_projection_single.mortar_to_secondary_int @ 
                self.ad_var_q
                )
            
            # Edge convection
            temp_wrap -= self.divergence_single @ (
                upwind_temp.bound_transport_neu
                @ self.mortar_projection_single.mortar_to_primary_int
                @ self.ad_var_w
            )

            temp_wrap -= (
                self.mortar_projection_single.mortar_to_secondary_int @ 
                self.ad_var_w
                )
        # end if
        
        # Include possible external (well) sources
        if well_source:
            
            injection = self.injection_value(self.temperature_kw)
            t_in = density * injection 
                
            temp_wrap = temp_wrap - (
                self.injection_rate * t_in +
                self.production_rate * density * temp
                )
        # end if
        
        # convection over interface
        if len(equation_system.mdg.interfaces()) > 0:
            # Upstream convection
            convection_coupling = pp.ad.UpwindCouplingAd(
                keyword=self.temperature_kw, interfaces=equation_system.mdg.interfaces()
            )
            high_to_low_convection = (
                convection_coupling.upwind_primary
                @ self.mortar_projection_single.primary_to_mortar_avg
                @ self.trace_single.trace
                @ (density * temp)
            )

            low_to_high_convection = (
                convection_coupling.upwind_secondary
                @ self.mortar_projection_single.secondary_to_mortar_avg
                @ (density * temp)
            )

            interface_convection = (
                convection_coupling.mortar_discr @ self.ad_var_w
                - (high_to_low_convection + low_to_high_convection)
                * self.interface_flux
            )

            # Conductive flux
            temp_from_high = (
                self.mortar_projection_single.primary_to_mortar_avg @ (
                    conduction_tpfa.bound_pressure_cell @ temp + 
                    conduction_tpfa.bound_pressure_face @ (
                    self.mortar_projection_single.mortar_to_primary_int @ self.ad_var_q + 
                    self.bound_temp 
                    )
                    )
            )
            
            temp_from_low = (
                self.mortar_projection_single.secondary_to_mortar_avg @ 
                temp 
                )

            robin_temp = pp.ad.RobinCouplingAd(
                keyword=self.temperature_kw, interfaces=equation_system.mdg.interfaces()
            )

            conduction_over_interface = self.ad_var_q - robin_temp.mortar_discr @ (
                temp_from_high - temp_from_low
            )
        # end if        
         
        # Store information about the equations necessary for splitting methods
        if self.eq_name_temperature not in self.eq_names:   
            self.eq_names.append(self.eq_name_temperature)
            self.ad_vars.append(self.ad_var_temp)
        # end if
        

        # Discretise the equations
        if len(equation_system.mdg.interfaces()) > 0:
           
            temp_wrap.discretize(equation_system.mdg)
            interface_convection.discretize(equation_system.mdg)
            conduction_over_interface.discretize(equation_system.mdg)

            equation_system.equations.update(
                {
                    self.eq_name_temperature: temp_wrap,
                    self.eq_name_interface_convection: interface_convection,
                    self.eq_name_interface_conduction: conduction_over_interface,
                }
            )     
            
            if self.eq_name_interface_convection not in self.eq_names:
                self.eq_names.append(self.eq_name_interface_convection)
                self.eq_names.append(self.eq_name_interface_conduction)
                self.ad_vars.append(self.ad_var_w) 
                self.ad_vars.append(self.ad_var_q)
            # end if
        else:
                        
            temp_wrap.discretize(equation_system.mdg)
            equation_system.equations.update({self.eq_name_temperature: temp_wrap})
        # end if-else
          
    def get_tracer_eqs(self, equation_system, iterate=False, well_source=False):

        # Variable
        passive_tracer = self.ad_var_passive_tracer
        
        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )
        data_transport = d[pp.PARAMETERS][self.transport_kw]
        data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
                
        # Upwind discretization for advection.
        upwind_tracer = pp.ad.UpwindAd(
            keyword=self.tracer_kw, subdomains=equation_system.mdg.subdomains()
        )
        
        if len(equation_system.mdg.interfaces()) > 0:
            upwind_tracer_coupling = pp.ad.UpwindCouplingAd(
                keyword=self.tracer_kw, interfaces=equation_system.mdg.interfaces()
            )
        # end if
        
        # For the accumulation term, we need T and teh porosity at
        # the previous time step. These should be fixed in the Newton-iterations
        if iterate:  # Newton-iteration
            mass_tracer = pp.ad.MassMatrixAd(
                self.tracer_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_tracer_prev = data_prev_time["mass_tracer_prev"]
            passive_tracer_prev = data_prev_time["tracer_prev"]
        else:  # We are interested in "constructing" the equations

            # Mass matrix for accumulation
            mass_tracer = pp.ad.MassMatrixAd(
                self.tracer_kw, equation_system.mdg.subdomains()
            )
            mass_tracer_prev = pp.ad.MassMatrixAd(
                self.tracer_kw, equation_system.mdg.subdomains()
            )

            # The solution at time step n, i.e.
            # the one we need to use to fine the solution at time step n+1
            passive_tracer_prev = passive_tracer.previous_timestep()

            # Store for Newton iterations
            data_prev_time["tracer_prev"] = passive_tracer_prev
            data_prev_time["mass_tracer_prev"] = mass_tracer_prev
        # end if-else

        dt = pp.ad.Scalar(data_transport["time_step"])
        
        tracer_wrapper = (
            (mass_tracer.mass @ passive_tracer -
             mass_tracer_prev.mass @ passive_tracer_prev) / dt
            # advection
            + self.divergence_single @ (
               (upwind_tracer.upwind @ passive_tracer) * self.full_flux
            - (upwind_tracer.bound_transport_dir @ self.full_flux) * self.bound_tracer
            - (upwind_tracer.bound_transport_neu @ self.bound_tracer)
            )
        )
        
        # Add the projections
        if len(equation_system.mdg.interfaces()) > 0:

            tracer_wrapper -= (
                self.divergence_single
                @ upwind_tracer.bound_transport_neu
                @ self.mortar_projection_single.mortar_to_primary_int
                @ self.ad_var_zeta_tracer
            ) #/ phi

            tracer_wrapper -= (
                self.mortar_projection_single.mortar_to_secondary_int @ self.ad_var_zeta_tracer
                ) 

            # Next, tracer over the interface

            # First project the concentration from high to low
            high_to_low_tracer = (
                upwind_tracer_coupling.upwind_primary @ 
                self.mortar_projection_single.primary_to_mortar_avg @ 
                self.trace_single.trace @ passive_tracer 
            ) 

            # Next project concentration from lower onto higher dimension
            low_to_high_tracer = (
                upwind_tracer_coupling.upwind_secondary @ 
                self.mortar_projection_single.secondary_to_mortar_avg @ 
                passive_tracer
            ) 
            
            # Finally we have the transport over the interface equation
            tracer_over_interface_wrapper = (
                self.ad_var_zeta_tracer
                - (high_to_low_tracer + low_to_high_tracer) * self.interface_flux
            ) 
        # end if
        
        # Include possible external (well) sources
        if well_source:
            
            injection = self.injection_value(self.tracer_kw)
            tracer_wrapper = tracer_wrapper - (
                self.injection_rate * injection + 
                self.production_rate * passive_tracer
                )
        # end if
        
        # Help Newton
        #tracer_wrapper *= pp.ad.Scalar(constant_params.scale_const())

        # Store information about the equations necessary for splitting methods
        if self.eq_name_passive_tracer not in self.eq_names:
            self.eq_names.append(self.eq_name_passive_tracer)
            self.ad_vars.append(passive_tracer)
        # en if
        
        if len(equation_system.mdg.interfaces()) > 0:
            
            tracer_wrapper.discretize(equation_system.mdg)
            tracer_over_interface_wrapper.discretize(equation_system.mdg)
        else:
            tracer_wrapper.discretize(equation_system.mdg)
        # end if-else

        equation_system.equations.update({self.eq_name_passive_tracer: tracer_wrapper})

        if len(equation_system.mdg.interfaces()) > 0:
            equation_system.equations.update(
                {self.eq_name_tracer_interface_flux: tracer_over_interface_wrapper}
            ) 
            if self.eq_name_tracer_interface_flux not in self.eq_names:
                self.eq_names.append(self.eq_name_tracer_interface_flux)
                self.ad_vars.append(self.ad_var_zeta_tracer)
            # end if
        # end if
        
    def get_solute_transport_TC_eqs(self, equation_system, iterate=False):
        """Solute transport equations, using the so-called
        TC-formulation (de Dieuleveult et al 2009)
        
        This is the pimary choice if the total concentrations are transported
        
        """
        # Variables
        U = self.ad_var_tot_U
        U_fluid = self.ad_var_aq

        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )

        data_transport = d[pp.PARAMETERS][self.transport_kw]
        data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
        data_chemistry = d[pp.PARAMETERS][self.chemistry_kw]

        # Cell-wise chemical values
        cell_S = data_chemistry["cell_stoic_coeff_S"]
        cell_equil_comp = data_chemistry["cell_equilibrium_constants_comp"]

        # Upwind discretization for advection.
        upwind = pp.ad.UpwindAd(
            keyword=self.transport_kw, subdomains=equation_system.mdg.subdomains()
        )
        if len(equation_system.mdg.interfaces()) > 0:
            upwind_coupling = pp.ad.UpwindCouplingAd(
                keyword=self.transport_kw, interfaces=equation_system.mdg.interfaces()
            )
        # end if

        # Divergence operator. Acts on fluxes of aquatic components only.
        div = self.divergence_several  # data_grid["divergence_several"]

        # The primary variable for concentrations is on log-form,
        # while the standard term is advected.
        # Conservation applies to the linear form of the total consentrations.
        # Make a function which carries out both conversion and summation over primary
        # and secondary species.
        def log_to_linear(c: pp.ad.AdArray) -> pp.ad.AdArray:
            return pp.ad.exp(c) + cell_S.T @ cell_equil_comp @ pp.ad.exp(cell_S @ c)

        # Wrap function in an Ad function, ready to be parsed
        log2lin = pp.ad.Function(log_to_linear, "")

        # For the accumulation term, we need T and teh porosity at
        # the previous time step. These should be fixed in the Newton-iterations
        if iterate:  # Newton-iteration
            mass = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_prev = data_prev_time["mass_prev"]
            U_prev = data_prev_time["T_prev"]
        else:  # We are interested in "constructing" the equations

            # Mass matrix for accumulation
            mass = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_prev = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )

            # The solute solution at time step n,
            # i.e. the one we need to use to fine the solution at time step n+1
            U_prev = U.previous_timestep()

            # Store for Newton iterations
            data_prev_time["T_prev"] = U_prev
            data_prev_time["mass_prev"] = mass_prev
        # end if-else

        # This consists of terms for accumulation and advection, in addition to boundary conditions.
        # We need four terms for the solute transport equation:
        # 1) Accumulation
        # 2) Advection
        # 3) Boundary condition for inlet
        # 4) boundary condition for outlet.

        expanded_flux = pp.ad.SparseArray(self.extend_flux) @ self.full_flux

        all_2_aquatic = self.all_2_aquatic  
        
        dt = pp.ad.Scalar(data_transport["time_step"])

        aqueous_var = log2lin(U_fluid)
        transport = (
            (mass.mass @ U - mass_prev.mass @ U_prev) / dt
            # advection
            + all_2_aquatic.transpose()
            @ div
            @ (
                (upwind.upwind @ all_2_aquatic @ aqueous_var) * expanded_flux
                - upwind.bound_transport_dir @ expanded_flux * self.bound_transport
                - upwind.bound_transport_neu @ self.bound_transport
            )  
        )
        # Add the projections
        if len(equation_system.mdg.interfaces()) > 0:

            zeta = self.ad_var_zeta

            all_2_aquatic_at_interface = self.all_2_aquatic_at_interface

            transport -= div @ (
                upwind.bound_transport_neu
                @ self.mortar_projection_several.mortar_to_primary_avg
                @ all_2_aquatic_at_interface.transpose() # Behaves like a "coupling factor" to ensure that  the multiplication is ok
                @ zeta  
            )

            transport -= (
                self.mortar_projection_several.mortar_to_secondary_avg
                @ all_2_aquatic_at_interface.transpose()
                @ zeta
            )
        # end if

        # Transport over the interface
        if len(equation_system.mdg.interfaces()) > 0:

            # Some tools we need
            upwind_coupling_primary = upwind_coupling.upwind_primary
            upwind_coupling_secondary = upwind_coupling.upwind_secondary

            expanded_v = pp.ad.SparseArray(self.extend_edge_flux) @ self.interface_flux

            # First project the concentration from high to low
            # At the higher-dimensions, we have both fixed
            # and aqueous concentration. Only the aqueous
            # concentrations are transported
            trace_of_conc = self.trace_several.trace @ aqueous_var

            high_to_low_trans = (
                upwind_coupling_primary
                @ all_2_aquatic_at_interface
                @ self.mortar_projection_several.primary_to_mortar_avg
                @ trace_of_conc
            )

            # Next project concentration from lower onto higher dimension
            # At the lower-dimension, we also have both
            # fixed and aqueous concentration
            low_to_high_trans = (
                upwind_coupling_secondary
                @ all_2_aquatic_at_interface
                @ self.mortar_projection_several.secondary_to_mortar_avg
                @ aqueous_var
            )

            # Finally we have the transport over the interface equation
            transport_over_interface = (
                upwind_coupling.mortar_discr @ zeta
                - (high_to_low_trans + low_to_high_trans) * expanded_v
            )
        # end if
        
        # Store information about the equations necessary for splitting methods
        if self.eq_name_solute_transport_TC not in self.eq_names:
            self.eq_names.append(self.eq_name_solute_transport_TC)
            self.ad_vars.append(self.ad_var_tot_U)
        # end if
            
        if len(equation_system.mdg.interfaces()) > 0:
            transport.discretize(equation_system.mdg)
            transport_over_interface.discretize(equation_system.mdg)

            equation_system.equations.update(
                {
                    self.eq_name_solute_transport_TC: transport,
                    self.eq_name_transport_over_interface: transport_over_interface,
                }
            ) 
            
            if self.eq_name_transport_over_interface not in self.eq_names:
                self.eq_names.append(self.eq_name_transport_over_interface)
                self.ad_vars.append(self.ad_var_zeta)
            # end if
            
        else:
            transport.discretize(equation_system.mdg)
            equation_system.equations.update(
                {self.eq_name_solute_transport_TC: transport}
            )
        # end if-else

    def subdomain_porosity_scale(self, mdg: pp.MixedDimensionalGrid, kw):
        """Porosity scaling of the subdomains"""

        porosity = np.zeros(mdg.num_subdomain_cells())
        global_index = 0
        for sd, d in mdg.subdomains(return_data=True):
            
            index = slice(global_index, global_index + sd.num_cells)
            
            phi = d[pp.PARAMETERS][kw]["porosity"]
            porosity[index] = phi 
            
            global_index += sd.num_cells
        # end if else

        return pp.ad.DenseArray(porosity)
    

    def get_solute_transport_CC_eqs(self, 
                                    equation_system, 
                                    iterate=False, 
                                    U_solid=None,
                                    well_source=False
                                    ):
        """Solute transport equations, using the so-called
        CC-formulation, as described  by de Dieuleveult et al 2009.
        
        This is the primary choice if the elements are transported
   
        """
        # Variables
        U = self.ad_var_tot_U
        U_fluid = self.ad_var_aq
        
        d = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )

        data_transport = d[pp.PARAMETERS][self.transport_kw]
        data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
        
        # Upwind discretization for advection.
        upwind = pp.ad.UpwindAd(
            keyword=self.transport_kw, subdomains=equation_system.mdg.subdomains()
        )

        # For the accumulation term, we need T and the porosity at
        # the previous time step. These should be fixed in the Newton-iterations
        if iterate:  # Newton-iteration
            mass = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_prev = data_prev_time["mass_prev"]
            U_prev = data_prev_time["T_prev"]
        else: # We are interested in "constructing" the equations

            # Mass matrix for accumulation
            mass = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )
            mass_prev = pp.ad.MassMatrixAd(
                self.mass_kw, subdomains=equation_system.mdg.subdomains()
            )

            # The solute solution at time step n,
            # i.e. the one we need to use to fine the solution at time step n+1
            U_prev = U.previous_timestep() # the state values 

            # Store for Newton iterations
            data_prev_time["T_prev"] = U_prev
            data_prev_time["mass_prev"] = mass_prev
        # end if-else
        
        self.porosity_scale = self.subdomain_porosity_scale(
            equation_system.mdg, self.mass_kw
            )
       
        expanded_flux = pp.ad.SparseArray(self.extend_flux) @ self.full_flux 
        expanded_porosity = pp.ad.SparseArray(self._porosity_scale_mat) @ self.porosity_scale
        
        
        # If U_solid is None, that corresponds to only a pure aqueous chemical system
        if U_solid is None:
            U_solid = np.zeros(U_fluid.evaluate(equation_system).val.size)
        # end if
        
        # Time step
        dt = pp.ad.Scalar(data_transport["time_step"])
        
        transport = (
            # Accumulation
           (mass.mass @ (U_fluid + pp.ad.DenseArray(U_solid))
             - 
             mass_prev.mass @ U_prev  
            ) / dt
            
            # Advection
            + self.divergence_several @ (
                (upwind.upwind @ U_fluid ) * expanded_flux -  
                (upwind.bound_transport_dir @ expanded_flux) * self.bound_transport -
                upwind.bound_transport_neu @ self.bound_transport 
                ) / expanded_porosity
                # divide by porosity to account for the porosity scaling in Reaktoro 
            )
               
        # Add the projections
        if len(equation_system.mdg.interfaces()) > 0:

            transport -= (
                self.divergence_several @ 
                upwind.bound_transport_neu @ 
                self.mortar_projection_several.mortar_to_primary_int @ 
                self.ad_var_zeta
            ) 

            transport -= (
                self.mortar_projection_several.mortar_to_secondary_int @ self.ad_var_zeta
                ) 

            # Transport over the interface
            # Some tools we need
            upwind_coupling = pp.ad.UpwindCouplingAd(
                self.transport_kw, equation_system.mdg.interfaces()
            )

            expanded_v = pp.ad.SparseArray(self.extend_edge_flux) @ self.interface_flux 
            
            # First project the concentration from high to low
            # At the higher-dimensions, we have both fixed
            # and aqueous concentration. Only the aqueous
            # concentrations are transported

            high_to_low_trans = (
                upwind_coupling.upwind_primary
                @ self.mortar_projection_several.primary_to_mortar_avg
                @ self.trace_several.trace
                @ (U_fluid / expanded_porosity) 
            )
            
            # Next project concentration from lower onto higher dimension
            # At the lower-dimension, we also have both
            # fixed and aqueous concentration
            low_to_high_trans = (
                upwind_coupling.upwind_secondary
                @ self.mortar_projection_several.secondary_to_mortar_avg
                @ (U_fluid / expanded_porosity) 
            )

            # Finally we have the transport over the interface equation
            transport_over_interface = (
                self.ad_var_zeta - (high_to_low_trans + low_to_high_trans) * expanded_v
            ) 
        # end if
        
        
        # Include possible external (well) sources
        if well_source:
          
            injection = self.injection_value(self.transport_kw)
            transport = transport - (
                 (pp.ad.SparseArray(self._extend_source_for_solute_transport) @ 
                     self.injection_rate) * injection
                 + 
                (pp.ad.SparseArray(self._extend_source_for_solute_transport) @ 
                    self.production_rate ) * U_fluid
                
                )
        # end if
        
        # Store information about the equations necessary for splitting methods
        if self.eq_name_solute_transport_CC not in self.eq_names:
            self.eq_names.append(self.eq_name_solute_transport_CC)
            self.ad_vars.append(self.ad_var_aq)
        # end if
        
        if len(equation_system.mdg.interfaces()) > 0:
            transport.discretize(equation_system.mdg)
            transport_over_interface.discretize(equation_system.mdg)

            equation_system.equations.update(
                {
                    self.eq_name_solute_transport_CC: transport,
                    self.eq_name_transport_over_interface: transport_over_interface,
                }
            )
          
            if self.eq_name_transport_over_interface not in self.eq_names:
                self.eq_names.append(self.eq_name_transport_over_interface)
                self.ad_vars.append(self.ad_var_zeta)
            # end if
            
        else:
            transport.discretize(equation_system.mdg)
            equation_system.equations.update(
                {self.eq_name_solute_transport_CC: transport}
            )
        # end if-else

    def get_mineral_eqs(self, equation_system):

        data_chemistry = equation_system.mdg.subdomain_data(
            equation_system.mdg.subdomains(dim=equation_system.mdg.dim_max())[0]
        )[pp.PARAMETERS][self.chemistry_kw]

        # Cell-wise chemical values
        cell_E = data_chemistry["cell_stoic_coeff_E"]
        cell_equil_prec = data_chemistry["cell_equilibrium_constants_prec"]

        def F_matrix(x1, x2):
            """
            Construct diagonal matrices, representing active and inactive sets
            """
            f = x1 - x2 > 0
            f = f.astype(float)
            Fa = sps.diags(f)  # Active
            Fi = sps.diags(1 - f)  # Inactive
            return Fa, Fi

        def phi_min(mineral_conc, log_primary):
            """
            Evaluation of the min function

       
            """
            
            sec_conc = 1 - cell_equil_prec @ pp.ad.exp(cell_E @ log_primary)
            Fa, Fi = F_matrix(mineral_conc.val, sec_conc.val)
            eq = Fa @ sec_conc + Fi @ mineral_conc 
            return eq

        precipitate = self.ad_var_precipitate
        log_X = self.ad_var_aq

        ad_min_1 = pp.ad.Function(phi_min, "")
        mineral_eq = ad_min_1(precipitate, log_X)
       
        # Store information about the equations necessary for splitting methods
        if self.eq_name_dissolution_precipitation not in self.eq_names:
            self.eq_names.append(self.eq_name_dissolution_precipitation)
            self.ad_vars.append(self.ad_var_precipitate)
        # end if
        
        equation_system.equations.update(
            {self.eq_name_dissolution_precipitation: mineral_eq}
        )
    
    def get_all_eqs(self, equation_system, iterate=False):

        to_iterate = iterate
        self.get_flow_eqs(equation_system, to_iterate)
        self.get_incompressible_flow_eqs(equation_system)
        self.get_equilibrium_eqs(equation_system)
        self.get_solute_transport_TC_eqs(equation_system, to_iterate)
        self.get_mineral_eqs(equation_system)
        self.get_temperature_eqs(equation_system, to_iterate)
        self.get_tracer_eqs(equation_system, to_iterate)
        
    def remove_equations(self, eq_name_list=None):
        """
        Remove given equation names, and assosiated variables
        
        Parameters:
            eq_name_list: list of variable name to be removed. Example ["pressure"]
                    
        If `None`is given, all equations and names are removes
        
        """

        if eq_name_list is None:
            self.eq_names.clear()
            self.ad_vars.clear()
        
        else: # Remove desired equations only
            i=0
            while i < len(self.eq_names) :
                if self.eq_names[i] in eq_name_list:
                    self.eq_names.remove(self.eq_names[i])
                    self.ad_vars.remove(self.ad_vars[i])
                    
                    i = 0 # since the removement reduce the list by one, 
                        # this is to avoid possiblity ot overlook any elements 
                        # and indexes outside the array    
                else:
                    i+=1
                # end if
            # end i-loop
        # end if-else
    