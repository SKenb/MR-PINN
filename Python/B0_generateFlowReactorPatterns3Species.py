from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PyFlow.FlowReactor import FlowReactor


R_GAS = 8.31446261815324


def arrhenius_rate(temperature_c: float, k_ref: float, ea_j_per_mol: float, t_ref_c: float = 150.0) -> float:
    temperature_k = temperature_c + 273.15
    t_ref_k = t_ref_c + 273.15
    return k_ref * np.exp((ea_j_per_mol / R_GAS) * ((1.0 / t_ref_k) - (1.0 / temperature_k)))


def build_inlet_profiles(time_vec: np.ndarray) -> np.ndarray:
    cin = np.zeros((3, len(time_vec)), dtype=np.float64)

    # Species 1 (A): early pulse, then a smaller late pulse.
    cin[0, :] = 0.0
    cin[0, (time_vec >= 0) & (time_vec < 900)] = 1.00
    cin[0, (time_vec >= 900) & (time_vec < 1800)] = 0.55
    cin[0, (time_vec >= 1800) & (time_vec < 2700)] = 1.20
    cin[0, (time_vec >= 2700)] = 0.35

    # Species 2 (B): staggered pulse train to drive A + B -> C conversion.
    cin[1, :] = 0.0
    cin[1, (time_vec >= 300) & (time_vec < 1200)] = 0.85
    cin[1, (time_vec >= 1500) & (time_vec < 2400)] = 1.05
    cin[1, (time_vec >= 3000)] = 0.70

    # Species 3 (C): product inlet is zero.
    cin[2, :] = 0.0

    return cin


def make_reaction_callback(k_ref: float, ea_j_per_mol: float):
    def reaction_network(C: np.ndarray, species_idx: int, temperature_c: float):
        a = np.clip(C[0, :], 0.0, None)
        b = np.clip(C[1, :], 0.0, None)
        rate = arrhenius_rate(temperature_c, k_ref=k_ref, ea_j_per_mol=ea_j_per_mol) * a * b

        if species_idx == 0:
            return -rate
        if species_idx == 1:
            return -rate
        if species_idx == 2:
            return rate

        raise ValueError(f"Unsupported species index: {species_idx}")

    return reaction_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-species flow reactor data without overwriting existing results.")
    parser.add_argument("--output-path", type=str, default="./Data/flowReactorData_3species.pkl")
    parser.add_argument("--figure-path", type=str, default="./Data/flowReactorData_3species.png")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--flow-rate", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=150.0)
    parser.add_argument("--k-ref", type=float, default=0.08)
    parser.add_argument("--ea-kj-per-mol", type=float, default=35.0)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the output files.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    figure_path = Path(args.figure_path)

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing data file: {output_path}")
    if figure_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing figure file: {figure_path}")

    reactor = FlowReactor(length=5.0, diameter=0.8e-3)
    reactor.setReactionNetworkCallback(make_reaction_callback(args.k_ref, args.ea_kj_per_mol * 1000.0))
    print(reactor)

    time_vec = np.linspace(0.0, args.hours * 3600.0, args.points)
    cin = build_inlet_profiles(time_vec)

    tout, cout, cspatial = reactor.simulate(
        timeVec=time_vec,
        Cin=cin,
        flowRate=args.flow_rate,
        temperature=args.temperature,
        isVolumetricFlowRate=False,
    )

    fig = plt.figure(figsize=(16, 9))
    reactor.plot(Cin=cin, CinTimeVec=time_vec, figure=fig, showPlot=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    if args.show_plot:
        plt.show()
    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(
            {
                "timeInVec": time_vec,
                "timeOutVec": tout,
                "Cout": cout,
                "Cin": cin,
                "Cspatial": cspatial,
                "flowRate": args.flow_rate,
                "temperature": args.temperature,
                "speciesNames": ["A", "B", "C"],
                "reaction": "A + B -> C",
                "kRef": args.k_ref,
                "eaKJPerMol": args.ea_kj_per_mol,
                "hours": args.hours,
                "points": args.points,
            },
            handle,
        )

    print(f"Saved data to {output_path}")
    print(f"Saved figure to {figure_path}")
    print(f"Cin shape: {cin.shape}")
    print(f"Cout shape: {cout.shape}")
    print(f"Cspatial shape: {cspatial.shape}")


if __name__ == "__main__":
    main()