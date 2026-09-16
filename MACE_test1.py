from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
from mace.calculators import mace_mp

calc = mace_mp(model="medium")

def run_sweep(element, calc, a_min, a_max, n_points):
    a_vals = np.linspace(a_min, a_max, n_points)
    volumes, energies = [], []

    for a in a_vals:
        atoms = bulk(element, "fcc", a=a, cubic=True)
        atoms.calc = calc
        volumes.append(atoms.get_volume() / len(atoms))
        energies.append(atoms.get_potential_energy() / len(atoms))

    return a_vals, np.array(volumes), np.array(energies)

def optimal_fit(volumes, energies):
    min_idx = np.argmin(energies)
    best_rmse = float("inf")
    best_results = None
    # tests different windows for the fit
    for radius in [2, 3, 4, 5]:
        start = max(0, min_idx - radius)
        end = min(len(energies), min_idx + radius + 1)

        v_win = volumes[start:end]
        e_win = energies[start:end]

        if len(v_win) < 5:
            continue

        # quadratic fit
        c2, c1, c0 = np.polyfit(v_win, e_win, 2)

        # Calculates RMSE
        e_pred = c2 * v_win**2 + c1 * v_win + c0
        rmse = np.sqrt(np.mean((e_win - e_pred) ** 2))
        # calculates bulk modulus
        if rmse < best_rmse:
            best_rmse = rmse
            v0 = -c1 / (2 * c2)
            curvature = 2 * c2
            b_gpa = v0 * curvature * 160.21766
            best_results = {
                "v0": v0,
                "b_gpa": b_gpa,
                "rmse": rmse,
                "points": len(v_win),
                "c0": c0,
                "c1": c1,
                "c2": c2
            }

    return best_results

a_coarse, v_coarse, e_coarse = run_sweep("Al", calc, 3, 6, 300)
min_idx = np.argmin(e_coarse)
a_min_energy = a_coarse[min_idx]
a_fine, v_fine, e_fine = run_sweep("Al", calc, a_min_energy*0.97, a_min_energy*1.03, 20)
results = optimal_fit(v_fine, e_fine)

v_fit = np.linspace(min(v_fine), max(v_fine), 100)
e_fit = results["c2"] * v_fit**2 + results["c1"] * v_fit + results["c0"]

label_text = "bulk modulus = {:.2f} GPa".format(results["b_gpa"])
plt.plot(v_fine, e_fine, "o", label="MACE Data")
plt.plot(v_fit, e_fit, "--", label="Parabolic Fit")
plt.plot([], [], " ", label=f"Equilibrium Volume = {results['v0']:.2f} Å³")
plt.plot([], [], " ", label=label_text)
plt.xlabel("Volume per atom (Å³)")
plt.ylabel("Energy per atom (eV)")
plt.title("Aluminium Energy vs Volume Surface Fit")
plt.legend()
plt.savefig("aluminium_fit.png", dpi=150)