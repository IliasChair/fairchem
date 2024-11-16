from pathlib import Path
from multiprocessing import cpu_count
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.mep import NEB
from ase.mep.neb import NEBTools
from ase.optimize import BFGS
import ase.io
from concurrent.futures import ProcessPoolExecutor
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.preprocessing import AtomsToGraphs
import numpy as np
from functools import partial
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import multiprocessing



def validate_neb_scans_cpu(neb_traj_dir:str,
                        num_workers:int,
                        neb_config:dict,
                        calculator_config:dict=None) -> dict:
    multiprocessing.set_start_method('spawn', force=True)

    """Run NEB validations in parallel using CPU processes"""
    traj_dir = Path(neb_traj_dir)
    traj_paths = sorted(traj_dir.glob("*.traj"))
    traj_paths = traj_paths[:4] #for testing only

    # Configure worker function
    worker_fn = partial(_run_single_neb,
                        n_images=neb_config.get("n_images"),
                        fmax=neb_config.get("fmax"),
                        max_steps=neb_config.get("max_steps"),
                        max_attempts=neb_config.get("max_attempts"),
                        fmax_adjustment_factor=neb_config.get("fmax_adjustment_factor"),
                        n_images_increment=neb_config.get("n_images_increment"),
                        calculator_config=calculator_config
                        )


    # Run parallel calculations
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker_fn, traj_paths))


    # Compute metrics in evaluator format
    n_total = len(results)
    n_success = sum(1 for r in results if r['success'])
    success_rate = n_success / n_total if n_total > 0 else 0.0

    metrics = {
        "neb_success_rate": {
            "metric": success_rate,
            "total": n_success,
            "numel": n_total
        }
    }

    barriers = [r['barrier'] for r in results if r['success'] and r['barrier'] is not None]
    if barriers:
        metrics["neb_avg_barrier"] = {
            "metric": np.mean(barriers),
            "total": np.sum(barriers),
            "numel": len(barriers)
        }
        metrics["neb_std_barrier"] = {
            "metric": np.std(barriers),
            "total": np.sum((np.array(barriers) - np.mean(barriers))**2),
            "numel": len(barriers)
        }

    return metrics


def _run_single_neb(traj_path: Path,
                    n_images:int=7,
                    fmax:float=0.05,
                    max_steps:int = 500,
                    max_attempts:int=1,
                    fmax_adjustment_factor=None,
                    n_images_increment=None,
                    calculator_config:dict={}) -> dict:
    if max_attempts != 1:
        assert fmax_adjustment_factor is not None, (
            "fmax_reduction_factor must be provided if max_attempts is "
            "provided."
        )
        assert n_images_increment is not None, (
            "n_images_increment must be provided if max_attempts is "
            "provided."
        )

    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        calc = OCPCalculator(
            checkpoint_path=calculator_config.get("checkpoint_path"),
            cutoff=calculator_config.get("cutoff"),
            max_neighbors=calculator_config.get("max_neighbors"),
            cpu=calculator_config.get("cpu"),
            seed=calculator_config.get("seed"))


    frames = ase.io.read(traj_path, index=":")
    if len(frames) not in [2, 3]:
        raise ValueError("Trajectory must contain 2 or 3 frames, "
                        f"found {len(frames)}")

    try:

        # Create NEB images
        if len(frames) == 2:
            # Initial and final only
            images = [frames[0]]
            images += [frames[0].copy() for _ in range(n_images)]
            images.append(frames[1])
        else:
            # Initial, TS, and final case (3 frames)
            images = [frames[0]]  # Initial state
            images += [frames[1].copy() for _ in range(n_images)]  # Copies of TS guess
            images.append(frames[2])  # Final state
        for atoms in images:
            atoms.calc = calc

        neb = NEB(images, allow_shared_calculator = True)
        neb.interpolate()

        # Optimize
        optimizer = BFGS(neb)
        optimizer.run(fmax=fmax, steps=max_steps)

        neb_tools = NEBTools(images)

        barrier = neb_tools.get_barrier()

        # Get energy profile
        path = neb_tools.get_path()
        energies = neb_tools.get_energies()

        # Extract results
        if optimizer.converged():
            energies = [image.get_potential_energy() for image in neb.images]
            barrier = max(energies) - energies[0]
            return {
                'success': True,
                'barrier': barrier,
                'energies': energies,
                'max_force': max([max(abs(image.get_forces().flatten()))
                                for image in neb.images])
            }

        return {'success': False, 'barrier': None}

    except Exception as e:
        return {'success': False, 'barrier': None, 'error': str(e)}