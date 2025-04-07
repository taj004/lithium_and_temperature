#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:46:36 2023

Utillity functions needed for the RT calulations

@author: uw
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import os
from pathlib import Path
import pickle

def repeat(v, reps, equation_system, abs_val=True, to_ad=True):
    """
    Repeat a vector v, reps times
    Main target is the flux vectors, needed in the transport calculations
    """

    if isinstance(v, np.ndarray):
        num_v = v

    # Get expression for evaluation and check attributes
    expression_v = v.evaluate(equation_system)
    if isinstance(expression_v, np.ndarray):
        num_v = expression_v
    elif hasattr(expression_v, "val"):
        num_v = expression_v.val
    else:
        raise ValueError("Cannot repeat this vector")
    # end if-else

    v_reps = np.repeat(num_v, reps)

    # Wrap the array in Ad.
    # Note that we return absolute value of the expanded vectors
    # The reason is these vectors are ment to use as scale values
    # in the solute transport equation. The signs are handled in the upwind discretization

    if abs_val is True:
        v_reps = np.abs(v_reps)
    # end if

    if to_ad:
        v_reps = pp.ad.Array(v_reps)

    return v_reps


def to_vec(gb, param_kw, param, size_type="cells", to_ad=False):
    """
    Make an Ad-vector of a parameter param, from a param_kw
    """

    if size_type == "cells":
        vec = np.zeros(gb.num_cells())
    elif size_type == "faces":
        vec = np.zeros(gb.num_faces())
    else:
        raise ValueError("Unknown size type")
    # end if-else

    val = 0
    for g, d in gb:

        if size_type == "cells":
            num_size = g.num_cells
        else:
            num_size = g.num_faces
        # end if-else

        # Extract from the dictionary
        x = d[pp.PARAMETERS][param_kw][param]

        if isinstance(x, np.ndarray) is False:
            x *= np.ones(num_size)
        # end if

        # Indices
        inds = slice(val, val + num_size)
        # Store values and update for the next grid
        vec[inds] = x.copy()
        val += num_size
    # end g,d-loop

    if to_ad:
        return pp.ad.Array(vec)
    else:
        return vec


def enlarge_mat(n, m):
    """
    Construct a matrix s to expand an vector v of size n, m times
    The function can be though of an variant of np.tile
    but applicable to, say, pp.Ad_arrays
    """

    # if m<n:
    #     raise ValueError("The new size must be larger than the original one")

    e = sps.eye(n, n)
    s = sps.bmat([[e] for i in range(m)]).tocsr()
    return s


def enlarge_mat_2(n, m):
    """
    Construct a matrix s to expand an vector v of size n, m times
    The function can be though of an variant of np.tile
    but applicable to, say, pp.Ad_arrays

    """
    # Assert the sizes a sensible
    if m <= n:
        raise ValueError("The new size must be larger than the original one")

    k = m / n
    if k % 1 == 0:
        k = int(k)
        e = np.ones((k, 1))
        s = sps.lil_matrix((m, n))
        for i in range(n):
            s[i * k : (i + 1) * k, i] = e
        # end i-loop
    else:
        s = sps.bmat([[sps.eye(n, n)], [sps.eye(m - n, n)]])
    # end if-else
    s = s.tocsr()
    return s


def all_2_aquatic_mat(
    aq_components: np.ndarray,
    num_components: int,
    num_aq_components: int,
    mdg_size: int,
):
    """
    Create a mapping between the aqueous and fixed species

    Input:
        aq_componets: np.array of aqueous componets
        num_components: int, number of components
        num_aq : int, number of aqueous components
        mdg_size: number of cells, faces or mortar cells in the given mixed-dim grid
    """
    
    #num_aq_components = aq_components.size
    cols = np.ravel(
        aq_components.reshape((-1, 1)) + (num_components * np.arange(mdg_size)),
        order="F",
    )
    sz = num_aq_components * mdg_size
    rows = np.arange(sz)
    matrix_vals = np.ones(sz)
    
    # Mapping from all to aquatic components. The reverse map is achieved by a transpose.
    all_2_aquatic = pp.ad.SparseArray(  # Matrix
        sps.coo_matrix(
            (matrix_vals, (rows, cols)),
            shape=(num_aq_components * mdg_size, num_components * mdg_size),
        ).tocsr()
    )

    return all_2_aquatic


def save_mdg(mdg, folder_name=None, file_name=None):
    """Save a mdg during a simulation

    input:
        - mdg to be stored
        - folder_name, the folder where the mdg is stored
        - file_name, the name the stored mdg has. It should end with a "_" so the
        name is "file_name_mdg"

    """

    def remove_ad_operators(mdg):
        # First get data
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        d = mdg.subdomain_data(sd)

        # Data parameters
        d_prev_time_step = d[pp.PARAMETERS]["previous_time_step"]

        # Remove Ad-related variables, execpt time steps
        for keys in d_prev_time_step.keys():
            if keys != "time_step":
                d_prev_time_step[keys] = []
            # end if
        # end keys-loop

        # Data parameters
        d_prev_newton_step = d[pp.PARAMETERS]["previous_newton_iteration"]

        # Remove Ad-related variables, execpt sequential/Newton iterations
        for keys in d_prev_newton_step.keys():
            if keys != "Number_of_Newton_iterations":
                d_prev_newton_step[keys] = []
            # end if
        # end keys-loop

        # return the mdg
        return mdg

    # Remove an pp.ad.operations
    mdg = remove_ad_operators(mdg)

    mdg_list = [mdg]

    # Make folder, if necessary
    if folder_name is None:  # A default option
        folder_name = ""
    else:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # end if
    # end if-else

    # The mdg-file name
    if file_name is None:
        mdg_name = folder_name + "mdg"
    else:
        mdg_name = folder_name + file_name + "mdg"
    # end if

    def write_pickle(obj, path):
        """
        Store the current grid bucket
        """
        path = Path(path)
        raw = pickle.dumps(obj)
        with open(path, "wb") as f:
            raw = f.write(raw)
        return

    # Save
    write_pickle(mdg_list, mdg_name)
