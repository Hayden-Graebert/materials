from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
from mace.calculators import mace_mp

calc = mace_mp(model="medium")
atoms.calc = calc
