from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt

volumes, energies = [], []
for a in np.linspace(3.4, 3.9, 15):
    atoms = bulk("Cu", "fcc", a=a, cubic=True)
    atoms.calc = EMT()
    volumes.append(atoms.get_volume() / len(atoms))
    energies.append(atoms.get_potential_energy() / len(atoms))

plt.plot(volumes, energies, "o")
plt.xlabel("Volume per atom (Å³)")
plt.ylabel("Energy per atom (eV)")
plt.savefig("first_curve.png", dpi=150)
print("done")
