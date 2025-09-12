import subprocess
import os
import json
import logging
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import rdChemReactions, Draw
import argparse
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "tools/public_data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.yml")
OUTPUT_HDF5 = os.path.join(REPO_ROOT, "tools/results.hdf5")
IMAGE_FOLDER = os.path.join(REPO_ROOT, "tools/route_images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def download_aizynth_data(target_dir: str):
    """Download AiZynthFinder public data if not already present."""
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        cmd = ["python", "-m", "aizynthfinder.tools.download_public_data", target_dir]
        subprocess.run(cmd, check=True)
        logger.info(f"Data downloaded to {target_dir}")
    else:
        logger.info(f"Data already exists in {target_dir}, skipping download.")


def run_retrosynthesis(smiles: str) -> Dict:
    """Run retrosynthesis for a target SMILES using AiZynthFinder CLI."""
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    cmd = [
        "aizynthcli",
        "--smiles", smiles,
        "--config", CONFIG_PATH,
        "--output", OUTPUT_HDF5
    ]
    subprocess.run(cmd, check=True)
    logger.info(f"Retrosynthesis completed. Routes saved in {OUTPUT_HDF5}")

    json_path = OUTPUT_HDF5.replace(".hdf5", ".json")
    data = pd.read_json(json_path)

    return data


def recursive_rxn_smiles(data: dict, collected: List[str]) -> List[str]:
    """Recursively extract reaction SMILES from retrosynthesis tree."""
    rxn_smiles = data.get("metadata", {}).get("mapped_reaction_smiles")
    if rxn_smiles:
        products, reactants = rxn_smiles.split(">>")
        
        def remove_mapping(smiles: str) -> str:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                return Chem.MolToSmiles(mol)
            return smiles

        products = [remove_mapping(p) for p in products.split(".")]
        reactants = [remove_mapping(r) for r in reactants.split(".")]
        collected.append(".".join(reactants) + ">>" + ".".join(products))

    for child in data.get("children", []):
        recursive_rxn_smiles(child, collected)

    return collected


def get_all_rxn_smiles(data: dict) -> List[str]:
    """Get all reaction SMILES from a retrosynthesis tree."""
    return recursive_rxn_smiles(data, [])


def retro_planner():
    """Simplified CLI that only requires SMILES as input."""
    parser = argparse.ArgumentParser(description="Run AiZynthFinder retrosynthesis.")
    parser.add_argument("--smiles", type=str, required=True, help="Target molecule SMILES.")
    args = parser.parse_args()

    download_aizynth_data(DATA_DIR)
    data = run_retrosynthesis(args.smiles)
    all_rxn_smiles = get_all_rxn_smiles(data.children.values[0][0])
    
    logger.info(f"Retrosynthesis completed for {args.smiles}")
    logger.info(f"Extracted {len(all_rxn_smiles)} reactions.")
    for idx, rxn in enumerate(all_rxn_smiles):
        # display the reaction tree
        rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
        img = Draw.ReactionToImage(rxn, subImgSize=(400, 200))
        logger.info(rxn)
        save_path = os.path.join(IMAGE_FOLDER, f"reaction_{idx}.png")
        img.save(save_path)
        logger.info(f"Reaction image saved to {save_path}")

if __name__ == "__main__":
    retro_planner()
