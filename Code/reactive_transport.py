#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:56:55 2023

@author: uw
"""

class ReactiveTransport:
    "To keep track of both transport and chemical equations"
    def __init__(
            self,
            pde_solver=None,
            chemical_solver=None
            ):
        
        if pde_solver is not None:
            self.pde_solver=pde_solver
        # end if
        
        if chemical_solver is not None:
            self.chemical_solver=chemical_solver
        # end if
    # end init