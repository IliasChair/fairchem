"""
Script to precompute displacement structures and their graph representations
for vibrational analysis.
"""

import argparse
import logging
import pickle
from pathlib import Path

import ase.io
from ase.vibrations import Vibrations
from tqdm import tqdm

from fairchem.core.preprocessing import AtomsToGraphs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args):
    """
    Generates and saves precomputed displacement data.

    For each molecule in the input trajectory, this script computes all
    finite-difference-displaced structures, converts them to graph objects,
    and saves a list of these (displacement_object, graph_data, n_atoms)
    tuples to a .pt file specific to that molecule.
    """
    input_traj_path = Path(args.input_traj_path)
    output_dir = Path(args.output_precompute_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Initializing AtomsToGraphs converter...")
    # Ensure r_pbc is True if your systems are periodic
    a2g = AtomsToGraphs(
        max_neigh=60,
        radius=15.0,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_pbc=True,
    )

    logging.info(f"Reading trajectory from: {input_traj_path}")
    try:
        # Read all frames into memory for easier processing here.
        # For very large trajectories, an iterator approach might be needed,
        # but this simplifies molecule indexing.
        trajectory_frames = list(ase.io.read(input_traj_path, index=":"))
        if not trajectory_frames:
            logging.error("Input trajectory is empty.")
            return
    except Exception as e:
        logging.error(f"Failed to read trajectory: {e}")
        return

    logging.info(
        f"Starting precomputation for {len(trajectory_frames)} molecules..."
    )

    for mol_idx, atoms_frame in enumerate(tqdm(trajectory_frames, desc="Processing molecules")):
        molecule_identifier = atoms_frame.info.get("index", f"mol_{mol_idx}")
        if not molecule_identifier: # Handle cases where index might be empty string
             molecule_identifier = f"mol_{mol_idx}"


        precomputed_for_molecule = []

        # Use a temporary Vibrations object for generating displacements
        # No "name" is given to vib_temp to prevent ASE from attempting
        # its own caching during this precomputation.
        vib_temp = Vibrations(atoms_frame, delta=args.delta)

        for _disp_idx_for_mol, (
            displacement_obj,
            disp_atoms_structure,
        ) in enumerate(vib_temp.iterdisplace()):
            try:
                graph_data_item = a2g.convert(disp_atoms_structure)
                # Store essential displacement info and the graph data
                # The Displacement object itself can be large if it retains
                # references to the full Vibrations and Atoms objects.
                # We will pickle the displacement object as ASE expects it.

                # Make sure displacement_obj.vib.atoms.info["index"] is set
                # for downstream use in the trainer
                if "index" not in displacement_obj.vib.atoms.info:
                    displacement_obj.vib.atoms.info["index"] = molecule_identifier


                precomputed_for_molecule.append(
                    {
                        "graph_data": graph_data_item,
                        "displacement_obj": displacement_obj, # ASE Displacement object
                        "n_atoms": len(disp_atoms_structure),
                    }
                )
            except Exception as e:
                logging.error(
                    f"Failed to convert displaced structure for molecule "
                    f"{molecule_identifier}, displacement {_disp_idx_for_mol}: {e}"
                )
                continue # Skip this displacement

        if precomputed_for_molecule:
            # Save as a list of dicts, using torch.save for PyG Data objects
            # and pickle for other objects if mixed.
            # Using pickle for the whole list of dicts to handle Displacement obj.
            output_file_path = output_dir / f"{molecule_identifier}.pkl"
            try:
                with open(output_file_path, "wb") as f_out:
                    pickle.dump(precomputed_for_molecule, f_out)
                logging.debug(
                    f"Saved {len(precomputed_for_molecule)} items for "
                    f"{molecule_identifier} to {output_file_path}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to save precomputed data for "
                    f"{molecule_identifier} to {output_file_path}: {e}"
                )
        else:
            logging.warning(
                f"No precomputed items generated for molecule {molecule_identifier}."
            )

    logging.info(
        f"Precomputation finished. Data saved to: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute displacement graph data for vibrational analysis."
    )
    parser.add_argument(
        "--input-traj-path",
        type=str,
        required=True,
        help="Path to the input ASE trajectory file.",
    )
    parser.add_argument(
        "--output-precompute-dir",
        type=str,
        required=True,
        help="Directory to save the precomputed .pkl files.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Displacement distance for ASE Vibrations (default: 0.01).",
    )
    # AtomsToGraphs parameters are now hardcoded
    # parser.add_argument(
    #     "--max-neigh",
    #     type=int,
    #     default=60, # Default from BaseTrainer
    #     help="Maximum number of neighbors for AtomsToGraphs.",
    # )
    # parser.add_argument(
    #     "--radius",
    #     type=float,
    #     default=15.0, # Default from BaseTrainer
    #     help="Cutoff radius for AtomsToGraphs.",
    # )
    # parser.add_argument(
    #     "--r-energy",
    #     action=argparse.BooleanOptionalAction,
    #     default=False, # Exclude energy
    #     help="Include energy in graph (AtomsToGraphs).",
    # )
    # parser.add_argument(
    #     "--r-forces",
    #     action=argparse.BooleanOptionalAction,
    #     default=False, # Exclude forces
    #     help="Include forces in graph (AtomsToGraphs).",
    # )
    # parser.add_argument(
    #     "--r-distances",
    #     action=argparse.BooleanOptionalAction,
    #     default=False, # Exclude distances (usually computed from vectors)
    #     help="Include distances in graph (AtomsToGraphs).",
    # )
    # parser.add_argument(
    #     "--r-edges",
    #     action=argparse.BooleanOptionalAction,
    #     default=True, # Keep edges (needed for connectivity)
    #     help="Include edges in graph (AtomsToGraphs).",
    # )
    # parser.add_argument(
    #     "--r-pbc", # Default from BaseTrainer for Vibrations was True
    #     action=argparse.BooleanOptionalAction, # Allows --r-pbc / --no-r-pbc
    #     default=False,
    #     help="Include periodic boundary conditions (AtomsToGraphs).",
    # )

    parsed_args = parser.parse_args()
    main(parsed_args)