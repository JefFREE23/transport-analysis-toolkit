"""Command-line entry point for transport analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transport_toolkit.conductivity.tensor import conductivity_tensor
from transport_toolkit.drude.one_carrier import fit_one_carrier
from transport_toolkit.drude.two_carrier import fit_two_carrier
from transport_toolkit.hall.hall_analysis import analyze_hall
from transport_toolkit.io.loaders import load_data
from transport_toolkit.mr.magnetoresistance import calculate_mr
from transport_toolkit.msa.mobility_spectrum import run_msa
from transport_toolkit.preprocessing.symmetrization import symmetrize_transport
from transport_toolkit.utils.results import result_to_dict


def _load_processed(path: str | Path):
    frame = load_data(path)
    return symmetrize_transport(frame)


def _print_result(result, *, as_json: bool) -> None:
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = result_to_dict(result)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def command_hall(args) -> None:
    processed = _load_processed(args.file)
    result = analyze_hall(processed["B"], processed["Rxy_anti"], geometry=args.geometry)
    _print_result(result, as_json=args.json)


def command_mr(args) -> None:
    processed = _load_processed(args.file)
    result = calculate_mr(processed["B"], processed["Rxx_sym"])
    _print_result(result, as_json=args.json)


def command_tensor(args) -> None:
    processed = _load_processed(args.file)
    sigma_xx, sigma_xy = conductivity_tensor(processed["Rxx_sym"], processed["Rxy_anti"], convention=args.convention)
    payload = {"B": processed["B"].to_numpy(), "sigma_xx": sigma_xx, "sigma_xy": sigma_xy}
    if args.json:
        print(json.dumps({key: list(value) for key, value in payload.items()}, indent=2))
    else:
        print(f"computed {len(processed)} conductivity tensor points")
        print(f"sigma_xx[0]: {sigma_xx[0]}")
        print(f"sigma_xy[0]: {sigma_xy[0]}")


def command_analyze(args) -> None:
    processed = _load_processed(args.file)
    hall = analyze_hall(processed["B"], processed["Rxy_anti"], geometry=args.geometry)
    mr = calculate_mr(processed["B"], processed["Rxx_sym"])
    sigma_xx, sigma_xy = conductivity_tensor(processed["Rxx_sym"], processed["Rxy_anti"], convention=args.convention)
    payload = {
        "hall": hall.to_dict(),
        "magnetoresistance": mr.to_dict(),
        "tensor_points": len(processed),
        "sigma_xx_first": float(sigma_xx[0]),
        "sigma_xy_first": float(sigma_xy[0]),
    }
    _print_result(payload, as_json=args.json)


def command_fit_one(args) -> None:
    processed = _load_processed(args.file)
    sigma_xx, sigma_xy = conductivity_tensor(processed["Rxx_sym"], processed["Rxy_anti"], convention=args.convention)
    result = fit_one_carrier(processed["B"], sigma_xx, sigma_xy, carrier=args.carrier)
    _print_result(result, as_json=args.json)


def command_fit_two(args) -> None:
    processed = _load_processed(args.file)
    sigma_xx, sigma_xy = conductivity_tensor(processed["Rxx_sym"], processed["Rxy_anti"], convention=args.convention)
    result = fit_two_carrier(processed["B"], sigma_xx, sigma_xy)
    _print_result(result, as_json=args.json)


def command_run_msa(args) -> None:
    processed = _load_processed(args.file)
    sigma_xx, sigma_xy = conductivity_tensor(processed["Rxx_sym"], processed["Rxy_anti"], convention=args.convention)
    result = run_msa(processed["B"], sigma_xx, sigma_xy, alpha=args.alpha)
    _print_result(result, as_json=args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transport", description="Electrical transport analysis toolkit")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("file", help="CSV or TXT data file")
        subparser.add_argument("--geometry", choices=["bulk", "sheet"], default="bulk")
        subparser.add_argument("--convention", choices=["physics", "msa"], default="physics")

    analyze = subparsers.add_parser("analyze", help="run Hall, MR, and tensor analysis")
    add_common(analyze)
    analyze.set_defaults(func=command_analyze)

    hall = subparsers.add_parser("hall", help="run Hall analysis")
    add_common(hall)
    hall.set_defaults(func=command_hall)

    mr = subparsers.add_parser("mr", help="run magnetoresistance analysis")
    add_common(mr)
    mr.set_defaults(func=command_mr)

    tensor = subparsers.add_parser("tensor", help="compute conductivity tensor")
    add_common(tensor)
    tensor.set_defaults(func=command_tensor)

    fit_one = subparsers.add_parser("fit-one-carrier", help="fit one-carrier Drude model")
    add_common(fit_one)
    fit_one.add_argument("--carrier", choices=["auto", "electron", "hole"], default="auto")
    fit_one.set_defaults(func=command_fit_one)

    fit_two = subparsers.add_parser("fit-two-carrier", help="fit two-carrier Drude model")
    add_common(fit_two)
    fit_two.set_defaults(func=command_fit_two)

    msa = subparsers.add_parser("run-msa", help="run mobility spectrum analysis")
    add_common(msa)
    msa.add_argument("--alpha", type=float, default=1e-3)
    msa.set_defaults(func=command_run_msa)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
