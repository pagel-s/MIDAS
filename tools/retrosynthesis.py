import subprocess
import os
import json
import logging
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import rdChemReactions, Draw
import argparse
import pandas as pd
from collections import defaultdict
import json as _json

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

    json_path = OUTPUT_HDF5
    data = pd.read_json(json_path)

    return data


def recursive_rxn_smiles(data, all_rxn_smiles, level=0):
    for child in data:
        
        if "metadata" in child:
            if "mapped_reaction_smiles" in child["metadata"]:
                rxn_smiles = child["metadata"]["mapped_reaction_smiles"]
            else:
                return all_rxn_smiles
        else:
            return all_rxn_smiles
        products = rxn_smiles.split(">>")[0].split(".")
        reactants = rxn_smiles.split(">>")[1].split(".")

        # remove atom mapping numbers
        def remove_atom_mapping(smiles):
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol)

        products = [remove_atom_mapping(p) for p in products]
        reactants = [remove_atom_mapping(r) for r in reactants]


        # rejoing the reactants and products
        rxn_smiles = ".".join(reactants) + ">>" + ".".join(products)
        all_rxn_smiles[level].append(rxn_smiles)
        if "children" in child:
            for child in child["children"]:
                if "children" in child:
                    recursive_rxn_smiles(child["children"], all_rxn_smiles, level+1)

        return all_rxn_smiles

def get_all_rxn_smiles(data: dict, all_rxn_smiles: defaultdict, level: int) -> List[str]:
    """Get all reaction SMILES from a retrosynthesis tree."""
    return recursive_rxn_smiles(data, all_rxn_smiles, level)


def retro_planner():
    """Simplified CLI that only requires SMILES as input."""
    parser = argparse.ArgumentParser(description="Run AiZynthFinder retrosynthesis.")
    parser.add_argument("--smiles", type=str, required=True, help="Target molecule SMILES.")
    args = parser.parse_args()

    download_aizynth_data(DATA_DIR)
    data = run_retrosynthesis(args.smiles)
    all_rxn_smiles = get_all_rxn_smiles(data["children"].values[0], defaultdict(list), 0)

    logger.info(f"Retrosynthesis completed for {args.smiles}")
    logger.info(f"Extracted {len(all_rxn_smiles)} reactions.")
    counter = 0
    for idx, rxn_list in enumerate(all_rxn_smiles.values()):
        # display the reaction tree
        for rxn in rxn_list:
            rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
            img = Draw.ReactionToImage(rxn, subImgSize=(400, 200))
            logger.info(rxn)
            save_path = os.path.join(IMAGE_FOLDER, f"reaction_{counter}.png")
            img.save(save_path)
            logger.info(f"Reaction image saved to {save_path}")
        counter += 1


def _collect_reaction_smiles_from_tree(tree: Any) -> List[str]:
    """
    Traverse an AiZynthFinder output tree (dict or list) and collect mapped reaction SMILES.
    """
    reactions: List[str] = []

    def _walk(node):
        if isinstance(node, list):
            for child in node:
                _walk(child)
            return
        if not isinstance(node, dict):
            return
        meta = node.get("metadata", {})
        rxn = meta.get("mapped_reaction_smiles")
        if rxn:
            reactions.append(rxn)
        children = node.get("children")
        if children:
            _walk(children)

    _walk(tree)
    return reactions


def run_retrosynthesis_images(smiles: str, output_dir: str = IMAGE_FOLDER, sub_img_size=(400, 200)) -> List[str]:
    """
    Run retrosynthesis and save reaction-step images.

    Returns a list of saved PNG paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    download_aizynth_data(DATA_DIR)
    data = run_retrosynthesis(smiles)

    # Locate tree payload from dict or pandas DataFrame-like structure
    if isinstance(data, dict):
        tree_payload = data.get("children", data)
    else:
        try:
            tree_payload = data["children"].values[0]
        except Exception:
            tree_payload = data

    rxns = _collect_reaction_smiles_from_tree(tree_payload)
    saved_paths: List[str] = []
    counter = 0
    for rxn in rxns:
        try:
            rxn_obj = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
            img = Draw.ReactionToImage(rxn_obj, subImgSize=sub_img_size)
            save_path = os.path.join(output_dir, f"reaction_{counter}.png")
            img.save(save_path)
            saved_paths.append(save_path)
            counter += 1
        except Exception:
            # Skip any rendering errors
            continue

    return saved_paths


if __name__ == "__main__":
    retro_planner()
