from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
from mace.calculators import mace_mp

calc = mace_mp(model="medium")
volumes, energies = [], []

for a in np.linspace(4.025, 4.075, 10):
    atoms = bulk("Al", "fcc", a=a, cubic=True)
    atoms.calc = calc
    volumes.append(atoms.get_volume() / len(atoms))
    energies.append(atoms.get_potential_energy() / len(atoms))

c2, c1, c0 = np.polyfit(volumes, energies, 2)
V0 = -c1 / (2 * c2)
curvature = 2 * c2
v_fit = np.linspace(min(volumes), max(volumes), 100)
e_fit = c2 * v_fit**2 + c1 * v_fit + c0

plt.plot(volumes, energies, "o", label="MACE Data")
plt.plot(v_fit, e_fit, "--", label="Parabolic Fit")
plt.axvline(V0, color="red", linestyle=":", label=f"V0 = {V0:.2f} Å³")
plt.xlabel("Volume per atom (Å³)")
plt.ylabel("Energy per atom (eV)")
plt.title("Aluminium Energy vs Volume Surface Fit")
plt.legend()
plt.savefig("aluminium_fit.png", dpi=150)

print(f"Equilibrium Volume (V0): {V0:.4f} Å³")
print(f"Curvature (d²E/dV²):     {curvature:.4f} eV/Å⁶")

bulk_modulus_eV = V0 * curvature
bulk_modulus_GPa = bulk_modulus_eV * 160.21766
print(bulk_modulus_GPa)
