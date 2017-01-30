""" Implementation of a coarsening model for cementite in 4340 steel """

from __future__ import division
import pandas as pd
import numpy as np

from scipy.constants import R
from scipy.integrate import odeint

import matplotlib.pyplot as plt

# Thermodynamic and kinetic data
# Thermodynamic: partition coefficients in ferrite
def partition_coeffs(T):
    partition_data = pd.read_csv("partition_data.csv")
    DG = 1540.0 + 1.47 * T
    k_si = 0.05
    k_m = np.exp((partition_data.A + partition_data.B * T - DG) * 4.184 / R / T)
    labels = [element.upper() for element in partition_data.M] + ['SI']
    k_p = pd.Series(np.concatenate((k_m.values, [k_si])), index=labels)
    
    return k_p

def diffusion_data(T):
    """ Calculate the mobilities in alpha-iron (tracers) for a given system sys
        elements in sys are Fe, C, Mn, Si, Ni, Cr, Mo, Cu """    
    MOB = dict()
    diff_coeffs = pd.read_csv("bulk_diff_coeffs.csv")
    MOB['FE'] = 121e-4 * np.exp(-281.6e3 / R / T)
    X = 1e4 / T
    MOB['C'] = 10 ** (-4.9064 - 0.5199 * X + 1.61e-3 * X ** 2)
    labels = [element for element in diff_coeffs.M] + ['FE', 'C']
    D = diff_coeffs.D0 * np.exp(-diff_coeffs.Q / R / T)
    D = pd.Series(np.concatenate((D.values, [MOB['FE'], MOB['C']])), index=labels)
    # elt_labels = pd.Series(np.concatenate((diff_coeffs.M, ['FE', 'C'])))
            
    return D

def weight_to_mole(sys):
    """ Converts weight fractions to mole fractions.
        sys = dict('elt' : concentration), where elt is any element contained
        in the UNARY_PROPERTIES.MTC file """
    # Open and read the molar masses for the element in the system    
    ppt_file = open('UNARY_PROPERTIES.MTC', 'r')
    ppt_li = ppt_file.readlines()
    ppt_file.close()
    
    MM = dict()
    
    # Read molar masses in file
    for line in ppt_li:
        li = line.split()
        if len(li) > 4 and li[1] in sys.keys():
            MM[li[1]] = float(li[3])
    tot = 0.0
    for i, elt in enumerate(sys):
        tot = tot + float(sys[elt]) / MM[elt]
    mol = dict()
    for i, elt in enumerate(sys):
        mol[elt] = sys[elt] / MM[elt] / tot
        
    return mol

def nominal_composition(content):
    content["FE"] = 1.0
    for elt in content:
        if elt not in "FE":
            content["FE"] -= content[elt]
    return weight_to_mole(content)

def get_system_data(data_type, system_elements):
    return pd.Series([data_type[element] for element in system_elements])
    

def coarsening_with_ferrite_growth(y, t, *args):
    """ The coarsening rate differential equations when 
        ferrite grain coarsening is taken into account. """
    T, cem_fr, sig, c_m, k_p, d_l = args
    
    from scipy.constants import R
    Vcem = 24.0e-6
    # GB movement kinetic factor
    k = 3.3e-8 * np.exp(-171544 / R / T)
    # Gl free growth for a spherical grain containing a fraction f of cementite
    G_l = 8.0 * y[1] / 3.0 / cem_fr
    # K0 cementite free growth factor
    K0 = 8.0 * sig * Vcem / 81.0 / R / T
    # D_eff: effective diffusivity in matrix and dislocation pipes
    d_m = d_l + 2.0 * 5.4e-14 * np.exp(-171544.0 / R / T) / y[0]
    D_eff = 1. / (c_m * (1 - k_p) ** 2 / d_m).sum()
    
    dydt = [k * (1.0 - y[0] / G_l) / 2.0 / y[0],
            K0 * D_eff / y[1] ** 2]
    return dydt

if __name__ == '__main__':
    
    # Temperature (K) (constant for now)
    T = 923.0
    # Get thermodynamic and kinetic data
    eq_data = partition_coeffs(T)
    diff_data = diffusion_data(T)
    # substitutional elements
    subs = ["CR", "MN", "MO", "NI", "SI"]
    # reduce the data to the wanted system
    k_p = get_system_data(eq_data, subs)
    d_m = get_system_data(diff_data, subs)
    # Equilibrium cementite fraction
    feq_cem = 0.06
    # Set-up Matrix content
    alloy_wt = {"C": 0.4e-2, "CR": 0.8e-2, "MN": 0.7e-2, "MO": 0.25e-2, "NI": 1.825e-2, "SI": 0.225e-2}
    alloy_mol = nominal_composition(alloy_wt)
    alloy_subs_content = get_system_data(alloy_mol, subs)
    c_m_s = alloy_subs_content / (1.0 + (k_p - 1.0) * feq_cem)
    
    # Initial values
    r0_c = ((2.26 - 6.4e-3 * T + 4.6e-6 * T **2 ) * feq_cem * 1e-3) ** (1.0 / 3.0) * 1e-6
    G0 = (2.7e-3 * T - 2.027) * 1e-6
    #print G0, r0_c
    y0 = [G0, r0_c]
    # Interfacial energy
    int_en = 0.550
    # Pack the arguments (temperature, cementite eq fraction, interfacial energy, matrix equilbrium, 
    args = (T, feq_cem, int_en, c_m_s, k_p, d_m)
    # Set-up time discretization
    t = np.logspace(0, 6, num = 500)
    # Solve the ode system
    sol = odeint(coarsening_with_ferrite_growth, y0, t, args=args)
    
    time = [1, 600, 3600, 14400, 43200, 86400, 174000]
    rs_cube = [5.83e-23, 7.48e-23, 2.68e-22, 4.55e-22, 1.62e-21, 3.73e-21, 4.73e-21]
    rs = [ r_cube ** (1/3.0) * 1e6 for r_cube in rs_cube ]       
    
    plt.semilogx(t, sol[:, 0] * 1e6, 'b', lw=2, label='Ferrite Grain Growth')
    plt.semilogx(t, sol[:, 1] * 1e6, 'g', lw=2, label='Cementite Growth')
    # plt.semilogx(time, rs, "og")
    plt.legend(loc='best')
    plt.xlabel(r't (sec.)', fontsize=14)
    plt.ylabel(r'radius ($\mu$m)', fontsize=14)
    plt.ylim(0, 2.5)
    plt.grid()
    plt.savefig("results_sig_55mJ.png", dpi=400)
    
    
    
    