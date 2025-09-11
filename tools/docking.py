#!/usr/bin/env python3
import os
import subprocess
import time
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rich.progress import track
import numpy as np
from vina import Vina

from tools import generate_molecules, convert_to_3D


def prep_receptor(inputfile, outputfile):
    obabel_rec_cmd = (
        """obabel -i pdb {input_file} -o pdbqt -O {output_file} -xrh --ph 7.4"""
    )
    print(obabel_rec_cmd.format(input_file=inputfile, output_file=outputfile))
    result = subprocess.run(
        obabel_rec_cmd.format(input_file=inputfile, output_file=outputfile).split(),
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
    return outputfile


def validate_ligand_sdf(ligand_file):
    """Ensure ligand SDF contains at least one atom with 3D coordinates."""
    suppl = Chem.SDMolSupplier(ligand_file, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    if not mols:
        raise ValueError(f"Ligand {ligand_file} could not be read or is empty")
    conf = mols[0].GetConformer()
    if conf is None or conf.GetNumAtoms() == 0:
        raise ValueError(f"Ligand {ligand_file} does not contain valid 3D coordinates")
    return mols[0]

def compute_center_from_ligand(ligand_file):
    """
    Compute geometric center of a ligand
    from its SDF using RDKit and NumPy.
    """
    mol = validate_ligand_sdf(ligand_file)
    conf = mol.GetConformer()
    
    coords = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()])
    
    com = coords.mean(axis=0)
    return com

def prep_ligand(ligand_sdf, ligand_pdbqt):
    """
    Prepare ligand PDBQT directly from an existing 3D SDF file.
    
    Parameters:
        ligand_sdf (str): Input SDF file with 3D coordinates.
        ligand_pdbqt (str): Output PDBQT file.
    """
    # Convert directly to PDBQT using OpenBabel
    cmd = f"obabel -i sdf {ligand_sdf} -o pdbqt -O {ligand_pdbqt} -xh"
    print(cmd)
    subprocess.run(cmd.split(), check=True)

def vina_dock(
    lig_path,
    receptor_path,
    com,
    box_size=[30, 30, 30],
    verbosity=1,
    score_only=True,
    n_poses=5,
    exhaustiveness=32,
):
    """
    Dock a ligand using Vina Python API and optionally save the protein–ligand complex.

    Parameters
    ----------
    lig_path : str
        Path to ligand PDBQT file.
    receptor_path : str
        Path to receptor PDBQT file.
    com : list[float]
        Docking box center [x, y, z].
    box_size : list[float], optional
        Docking box dimensions [x, y, z], default [30, 30, 30].
    verbosity : int, optional
        Vina verbosity level, default 1.
    score_only : bool, optional
        If True, only compute score. If False, perform docking.
    save_complex : str or None, optional
        If a filepath is given, saves the best docked complex as PDB.
    n_poses : int, optional
        Number of poses to generate when docking, default 5.
    exhaustiveness : int, optional
        Docking exhaustiveness, default 32.

    Returns
    -------
    float
        Docking score (affinity in kcal/mol) for the best pose.
    """
    v = Vina(sf_name="vina", cpu=16, verbosity=verbosity)
    v.set_receptor(receptor_path)
    v.set_ligand_from_file(lig_path)
    v.compute_vina_maps(center=com, box_size=box_size)
    if score_only:
        # Just score the existing pose without docking
        energy = v.score()[0]
    else:
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        energy = v.energies(n_poses=1)[0][0]  # best score
    return energy

# -------------------------------
# Main Workflow
# -------------------------------

def main(args):
    

    receptor_pdbqt = prep_receptor(args.protein, args.protein.replace(".pdb", ".pdbqt"))
    docking_center = compute_center_from_ligand(args.reference_sdf)
    print(f"Docking center: {docking_center}")
    
    # Prepare ligand PDBQT
    ligand_pdbqt = args.ligand_sdf.replace(".sdf", ".pdbqt")
    prep_ligand(args.ligand_sdf, ligand_pdbqt)

    score = vina_dock(ligand_pdbqt, receptor_pdbqt, docking_center, box_size=[20,20,20], score_only=False)
    print(f"Docking score for: {score}")

# -------------------------------
# Argument Parsing
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Vina docking for one ligand")
    parser.add_argument("--reference_sdf", type=str, required=True, help="Input reference ligand SDF file")
    parser.add_argument("--protein", type=str, required=True, help="Protein PDB file")
    parser.add_argument("--ligand_sdf", type=str, required=True, help="Ligand SDF file to dock")
    
    args = parser.parse_args()
    main(args)
