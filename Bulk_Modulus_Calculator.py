from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
from mace.calculators import mace_mp

calc = mace_mp(model="medium")

materials = {
    "Al": {"struct": "fcc"},
    "Fe": {"struct": "bcc"},
    "Cu": {"struct": "fcc"},
    "Pd": {"struct": "fcc"},
    "Ag": {"struct": "fcc"},
    "Sn": {"struct": "bcc"},
    "W": {"struct": "bcc"},
    "Ir": {"struct": "fcc"},
    "Pt": {"struct": "fcc"},
    "Au": {"struct": "fcc"},
    "Pb": {"struct": "fcc"}
}

def run_sweep(element, crystal_struct, calc, a_min, a_max, n_points):
    a_vals = np.linspace(a_min, a_max, n_points)
    volumes, energies = [], []

    for a in a_vals:
        atoms = bulk(element, crystal_struct, a=a, cubic=True)
        atoms.calc = calc
        volumes.append(atoms.get_volume() / len(atoms))
        energies.append(atoms.get_potential_energy() / len(atoms))

    return a_vals, np.array(volumes), np.array(energies)

def optimal_fit(volumes, energies):
    min_idx = np.argmin(energies)
    best_rmse = float("inf")
    best_results = None

    for radius in [2, 3, 4, 5]: # Tests different fit windows
        start = max(0, min_idx - radius)
        end = min(len(energies), min_idx + radius + 1)

        v_win = volumes[start:end]
        e_win = energies[start:end]

        c2, c1, c0 = np.polyfit(v_win, e_win, 2) # Quadratic fit

        e_pred = c2 * v_win**2 + c1 * v_win + c0
        rmse = np.sqrt(np.mean((e_win - e_pred) ** 2)) # Calculates RMSE

        if rmse < best_rmse:
            best_rmse = rmse
            v0 = -c1 / (2 * c2)
            curvature = 2 * c2
            b_gpa = v0 * curvature * 160.21766 #Calculates Bulk Modulus
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

results = {}

for element, config in materials.items():
    a_c, v_c, e_c = run_sweep(element, config["struct"], calc, 3, 6, 300)
    a_min = a_c[np.argmin(e_c)]
    a_f, v_f, e_f = run_sweep(element, config["struct"], calc, a_min*0.97, a_min*1.03, 20)

    results[element] = optimal_fit(v_f, e_f) | {"element": element}

    v_fit = np.linspace(min(v_f), max(v_f), 100)
    e_fit = results[element]["c2"] * v_fit**2 + results[element]["c1"] * v_fit + results[element]["c0"]

    label_text = "Bulk Modulus = {:.2f} GPa".format(results[element]["b_gpa"])
    plt.plot(v_f, e_f, "o", label="MACE Data")
    plt.plot(v_fit, e_fit, "--", label="Parabolic Fit")
    plt.plot([], [], " ", label=f"Equilibrium Volume = {results[element]['v0']:.2f} Å³")
    plt.plot([], [], " ", label=label_text)
    plt.xlabel("Volume per atom (Å³)")
    plt.ylabel("Energy per atom (eV)")
    plt.title(results[element]["element"] + " Energy vs Volume Fit")
    plt.legend()
    plt.savefig(results[element]["element"] + "_fit.png", dpi=150)