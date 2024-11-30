from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

#from ase.optimize import BFGS
from ase.vibrations import Vibrations
from scipy.stats import pearsonr

from fairchem.core.common.relaxation.ase_utils import OCPCalculator


def validate_freq_scans_cpu(
    freq_traj_dir: str,
    num_workers: int | None = None,  # Make num_workers optional
    calculator_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run frequency validations in parallel using CPU processes.
    This function processes multiple trajectory files in parallel to validate frequency
    calculations, using CPU-based multiprocessing. It aggregates results and returns
    summary statistics including RMSE, MAE, and correlation coefficients.
        Directory containing trajectory files to process
        Number of worker processes to use. If None, uses all available CPU cores
        Configuration dictionary for the calculator settings
    Returns
    -------
    dict[str, Any]
        Dictionary containing validation results with the following keys:
        - n_total: Total number of calculations attempted
        - n_success: Number of successful calculations
        - rmse: Root mean square error for successful calculations
        - mae: Mean absolute error for successful calculations
        - corr: Mean correlation coefficient for successful calculations
    Notes
    -----
    Uses Python's ProcessPoolExecutor for parallelization and sets multiprocessing
    start method to "spawn" for better cross-platform compatibility.
    """
    multiprocessing.set_start_method("spawn", force=True)

    # Get total number of CPU cores if num_workers not specified
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    traj_dir = Path(freq_traj_dir)
    traj_paths = sorted(traj_dir.glob("*.traj"))

    # Configure worker function
    worker_fn = partial(_run_single_freq, calculator_config=calculator_config)

    # Run parallel calculations
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker_fn, traj_paths))

    # Flatten and aggregate results
    results = [item for sublist in results for item in sublist]

    n_total = len(results)
    n_success = sum(1 for r in results if r["success"])

    # Calculate metrics
    rmse = np.sqrt(np.mean([r["rmse"] ** 2 for r in results if r["success"]]))
    mae = np.mean([r["mae"] for r in results if r["success"]])
    corr = np.mean([r["corr"] for r in results if r["success"]])

    return {
        "freqs_success_rate": n_success / n_total,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
    }

def convert_freq_format(freqs):
    """Convert complex frequencies to real numbers and sort them in ascending order.

    This function takes complex frequencies and converts them to real numbers based on the
    following rules:
    - For complex frequencies, converts to negative absolute value of imaginary part
    - For real frequencies, keeps the real part
    - Returns the sorted array of converted frequencies

    Parameters
    ----------
    freqs : array-like
        Input array of frequencies that may contain complex numbers

    Returns
    -------
    ndarray
        Sorted array of real numbers derived from input frequencies according to conversion rules

    Examples
    --------
    >>> freqs = [1.0, 2.0+3.0j, -4.0-5.0j]
    >>> convert_freq_format(freqs)
    array([-5., -3.,  1.])
    """
    return np.sort([(-abs(f.imag) if np.iscomplex(f) else f.real) for f in freqs])


def _run_single_freq(
    traj_path: Path, calculator_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate frequencies and metrics for a single structure"""

    # Set up calculator
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        calc = OCPCalculator(**calculator_config)

    # Read structure
    mols = ase.io.read(traj_path)

    results = []
    for i, mol in enumerate(mols):
        try:
            ref_freqs = np.array(mol.info["freqs"])

            mol.calc = calc

            # Calculate frequencies
            vib = Vibrations(mol, name="vib/"+str(mol.info["index"]))
            vib.run()
            calc_freqs = convert_freq_format(vib.get_frequencies())
            # only use the last len(ref_freqs) frequencies
            calc_freqs = calc_freqs[-len(ref_freqs) :]

            vib.clean()

            # Calculate metrics for this structure
            results.append({
                "rmse": np.sqrt(np.mean((ref_freqs - calc_freqs) ** 2)),
                "mae": np.mean(np.abs(ref_freqs - calc_freqs)),
                "corr": pearsonr(ref_freqs, calc_freqs)[0],
                "success": True,
            })

        except Exception as e:
            print(f"Error in {traj_path.name} structure {i}: {e}")
            results.append({
                    "rmse": np.nan,
                    "mae": np.nan,
                    "corr": np.nan,
                    "success": False,
            })

    return results
