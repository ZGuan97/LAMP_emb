#!/usr/bin/env python3 

"""
This module collects some constants defined in the code
written by Feng Wu. calcmadModule.f90
"""

pi = 3.14159265358979323846e0
epsilon0 = 8.85418781762038985053656303171e-12
electronChgUnit = 1.602176487e-19
Angstrom = 1.0e-10

# conversion factor to electrostaic potential in unit of V from lengths of Angstrom
coef_pot = 0.25e0 / pi / epsilon0 * electronChgUnit / Angstrom