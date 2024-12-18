#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:38:01 2024

@author: uw
"""

import porepy as pp
import numpy as np
import constant_params
  

def set_bc_param_mc(mdg: pp.MixedDimensionalGrid):
    """Set boundary conditions for MC
    
    """
    
    # Keywords
    flow_kw = "flow"
    temperature = "temperature"
    tracer = "passive_tracer"
    
    for sd, d in mdg.subdomains(return_data=True):
 
        # Boundary faces
        bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        
        # The faces
        faces = np.array(["neu"] * bound_faces.size)
      
        if bound_faces.size != 0: # Global boundaries
            
            # For "special" BC values
            bound_faces_centers = sd.face_centers[:, bound_faces]
            
            left = bound_faces_centers[0, :] < np.min(sd.face_centers[0]) + 1e-4
            right = bound_faces_centers[0, :] > np.max(sd.face_centers[0]) - 1e-4
            top = bound_faces_centers[1, :] > np.max(sd.face_centers[1]) - 1e-4
            bottom = bound_faces_centers[1, :] < np.min(sd.face_centers[1]) + 1e-4
            
            faces[left] = "dir"
            faces[right] = "dir"
            faces[top] = "dir"
            faces[bottom] = "dir"
            
            #breakpoint()
            bound_faces_centers = sd.face_centers[:, bound_faces]

            # Set the different BC labels
            labels_for_flow = faces.copy()            
            
            bound_for_flow = pp.BoundaryCondition( 
                sd, faces=bound_faces, cond=labels_for_flow
            )
       
            labels_for_temp = faces.copy()
            bound_for_temp = pp.BoundaryCondition(
                sd, faces=bound_faces, cond=labels_for_temp
            )

            labels_for_tracer = faces.copy()
            bound_for_tracer = pp.BoundaryCondition(
                sd, faces=bound_faces, cond=labels_for_tracer
            )
            
            unity = np.ones(sd.num_faces)
            
            # Set BC values
    
            # -- Flow -- #
            bc_values_for_flow = 0 * unity 
            bc_values_for_flow[bound_faces[
                np.logical_or(left, right)
                ]] = constant_params.ref_p() 
            bc_values_for_flow[bound_faces[
                np.logical_or(top, bottom)
                ]] = constant_params.ref_p() 
                 
            # -- Temp -- #
            bc_values_for_temp = 0 * unity
            bc_values_for_temp[bound_faces[
                np.logical_or(left, right) 
                ]] = constant_params.ref_temp()

            bc_values_for_temp[bound_faces[
                np.logical_or(top, bottom)
                ]] = constant_params.ref_temp()  
            
            # -- tracer transport -- #
            bc_values_for_tracer = np.zeros(sd.num_faces)  
            bc_values_for_tracer[bound_faces[
                np.logical_or (left, right)
                ]] = 1.0 
            bc_values_for_tracer[bound_faces[
                np.logical_or(top, bottom)
                ]] = 1.0 
            
        else:       
            # Neumann faces        
            cond = pp.BoundaryCondition(sd)
        
            bc_values_for_flow = np.zeros(sd.num_faces)
            bound_for_flow = cond
            
            bc_values_for_temp = np.zeros(sd.num_faces)
            bound_for_temp = cond
            
            bc_values_for_tracer = np.zeros(sd.num_faces)
            bound_for_tracer = cond      
        # end if    
        
        # end if-else

        # Set the values in dictionaries
        flow_data = {
            "bc_values": bc_values_for_flow,
            "bc": bound_for_flow,
        }

        temp_data = {   
            "bc_values": bc_values_for_temp,
            "bc": bound_for_temp
        }

        passive_tracer_data = {
            "bc_values": bc_values_for_tracer,
            "bc": bound_for_tracer,
        }
        
        # Set in parameter dictionary
        d[pp.PARAMETERS][flow_kw].update(flow_data)
        d[pp.PARAMETERS][temperature].update(temp_data)
        d[pp.PARAMETERS][tracer].update(passive_tracer_data)
        
    # end mdg-loop
# end function


def set_bc_param_3d(mdg: pp.MixedDimensionalGrid):
    """Set boundary conditions for the 3D simulations
 
    """
    
    # Keywords
    flow_kw = "flow"  
    temperature = "temperature"
    tracer = "passive_tracer"
    
    for sd, d in mdg.subdomains(return_data=True):
  
        # Boundary faces
        bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        
        # The faces
        faces = np.array(["neu"] * bound_faces.size)
      
        if bound_faces.size != 0: # Global boundaries
            
            # For "special" BC values
            bound_faces_centers = sd.face_centers[:, bound_faces]
            
            z_end   = bound_faces_centers[2, :] > np.max(sd.face_centers[2]) - 1e-4
            z_start = bound_faces_centers[2, :] < np.min(sd.face_centers[2]) + 1e-4
            y_end   = bound_faces_centers[1, :] > np.max(sd.face_centers[1]) - 1e-4
            y_start = bound_faces_centers[1, :] < np.min(sd.face_centers[1]) + 1e-4
            x_end   = bound_faces_centers[0, :] > np.max(sd.face_centers[0]) - 1e-4
            x_start = bound_faces_centers[0, :] < np.min(sd.face_centers[0]) + 1e-4
            
            faces[z_end]   = "dir"
            faces[z_start] = "dir"
            faces[y_end]   = "dir"
            faces[y_start] = "dir"
            faces[x_end]   = "dir"
            faces[x_start] = "dir"
            
            #breakpoint()
            bound_faces_centers = sd.face_centers[:, bound_faces]

            # Set the different BC labels
            labels_for_flow = faces.copy()            
            
            bound_for_flow = pp.BoundaryCondition( 
                sd, faces=bound_faces, cond=labels_for_flow
            )
       
            labels_for_temp = faces.copy()
            bound_for_temp = pp.BoundaryCondition(
                sd, faces=bound_faces, cond=labels_for_temp
            )
                
            labels_for_tracer = faces.copy()
            bound_for_tracer = pp.BoundaryCondition(
                sd, faces=bound_faces, cond=labels_for_tracer
            )
            
            unity = np.ones(sd.num_faces)
            
            # Set BC values
    
            # -- Flow -- #
            bc_values_for_flow = 0 * unity 
            
            bc_values_for_flow[
                bound_faces[np.logical_or(z_start, z_end)]
                ] = constant_params.ref_p() 
                 
            bc_values_for_flow[
                bound_faces[np.logical_or(x_start, x_end)]
                ] = constant_params.ref_p() 
           
            bc_values_for_flow[
                bound_faces[np.logical_or(y_start, y_end)]
                ] = constant_params.ref_p() 
            
            
            # -- Temp -- #
            bc_values_for_temp = 0 * unity
            
            bc_values_for_temp[
                bound_faces[np.logical_or(x_start, x_end)]
                ] = constant_params.ref_temp()
            bc_values_for_temp[
                bound_faces[np.logical_or(y_start, y_end)]
                ] = constant_params.ref_temp()    
            bc_values_for_temp[
                bound_faces[np.logical_or(z_start, z_end)]
                ] = constant_params.ref_temp()    
            
            # -- Tracer -- #

            bc_values_for_tracer = 0 * unity
            
            bc_values_for_tracer[
                bound_faces[np.logical_or(x_start, x_end)]
                ] = 1.0  
            bc_values_for_tracer[
                bound_faces[np.logical_or(y_start, y_end)]
                ] = 1.0   
            bc_values_for_tracer[
                bound_faces[np.logical_or(z_start, z_end)]
                ] = 1.0               
            
        else:       
            
            cond = pp.BoundaryCondition(sd)
        
            bc_values_for_flow = np.zeros(sd.num_faces)
            bound_for_flow = cond
            
            bc_values_for_temp = np.zeros(sd.num_faces)
            bound_for_temp = cond
            
            bc_values_for_tracer = np.zeros(sd.num_faces)
            bound_for_tracer = cond      
        # end if    
        
        # end if-else

        # Set the values in dictionaries
        flow_data = {
            "bc_values": bc_values_for_flow,
            "bc": bound_for_flow,
        }

        temp_data = {   
            "bc_values": bc_values_for_temp,
            "bc": bound_for_temp
        }

        passive_tracer_data = {
            "bc_values": bc_values_for_tracer,
            "bc": bound_for_tracer,
        }
        
        # Set in parameter dictionary
        d[pp.PARAMETERS][flow_kw].update(flow_data)
        d[pp.PARAMETERS][temperature].update(temp_data)
        d[pp.PARAMETERS][tracer].update(passive_tracer_data)
        
    # end mdg-loop
# end function

#%% Set source

def set_source_rate(mdg: pp.MixedDimensionalGrid):
    """Set the (well) sources injection value"""
    
    # Keywords
    temperature = "temperature"
    tracer = "passive_tracer"
    
    injection_kw = "injection_value"
    
    for sd, d in mdg.subdomains(return_data=True):
  
        # The sources are only in the matrix
        if sd.dim == mdg.dim_max():
    
            # Temperature
            source_temp = np.array([constant_params.ref_temp() - 50])
             
            # Tracer
            source_tracer = np.array([0.0])
        # end if
        
        temp_data = {   
            injection_kw: source_temp
        }
        
        passive_tracer_data = {
            injection_kw: source_tracer
        }
        
        # Set in parameter dictionary       
        d[pp.PARAMETERS][temperature].update(temp_data)
        d[pp.PARAMETERS][tracer].update(passive_tracer_data)
    
