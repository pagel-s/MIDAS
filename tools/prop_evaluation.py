# -------------------------------
# Calculate the pharmacophoric property of the
# generated molecules
# -------------------------------

import os
import sys
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import QED

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import Draw, ChemicalFeatures
from rdkit import RDConfig
import os


def calculate_properties(smiles: str) -> dict:
    """Calculate a set of pharmacophoric properties for a given molecule.
    Args:
        smiles (str): SMILES string of the molecule.
    Returns:
        dict: A dictionary containing the calculated properties.
    """
    properties = {}
    mol = Chem.MolFromSmiles(smiles)
    properties["MW"] = round(Descriptors.MolWt(mol), 2)
    properties["LogP"] = round(Crippen.MolLogP(mol), 2)
    properties["HBA"] = Lipinski.NumHAcceptors(mol)
    properties["HBD"] = Lipinski.NumHDonors(mol)
    properties["TPSA"] = round(rdMolDescriptors.CalcTPSA(mol), 2)
    properties["QED"] = round(QED.qed(mol), 2)
    return properties

def draw_pharmacophore_features(smiles: str):
    """
    Draw a molecule with automatically detected pharmacophore features highlighted.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to visualize.

    Pharmacophore features highlighted
    ----------------------------------
    - HBD : Hydrogen Bond Donors (red)
    - HBA : Hydrogen Bond Acceptors (blue)
    - Aromatic : Aromatic ring atoms (purple)
    - PosIonizable : Positively ionizable groups (green)
    - NegIonizable : Negatively ionizable groups (orange)
    - Hydrophobe : Hydrophobic groups (brown)

    Notes
    -----
    Uses RDKit's ChemicalFeatures with the default BaseFeatures.fdef file.
    Colors are assigned per feature type, and atoms belonging to multiple
    features will be highlighted with the last assigned color.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # improve feature detection

    # Load default pharmacophore feature definitions
    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    features = factory.GetFeaturesForMol(mol)

    # Define colors for each feature type
    colors = {
        "Donor": (1.0, 0.4, 0.4),        # red
        "Acceptor": (0.4, 0.4, 1.0),     # blue
        "Aromatic": (0.8, 0.4, 0.8),     # purple
        "PosIonizable": (0.4, 1.0, 0.4), # green
        "NegIonizable": (1.0, 0.7, 0.2), # orange
        "Hydrophobe": (0.6, 0.4, 0.2),   # brown
    }

    # Map atoms → feature colors
    highlight_atom_colors = {}
    for f in features:
        ftype = f.GetFamily()
        if ftype not in colors:
            continue
        for idx in f.GetAtomIds():
            highlight_atom_colors[idx] = colors[ftype]

    # Draw molecule with highlighted features
    img = Draw.MolsToGridImage(
        [mol],
        highlightAtomLists=[list(highlight_atom_colors.keys())],
        highlightAtomColors=[highlight_atom_colors],
        subImgSize=(300, 300)
    )
    img.show()
    img.save("pharmacophore_features.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate pharmacophoric properties of molecules")
    parser.add_argument("smiles", type=str, help="Input SMILES string of the molecule")
    args = parser.parse_args()

    props = calculate_properties(args.smiles)
    props["SMILES"] = args.smiles

    # Print properties
    for key, value in props.items():
        print(f"{key}: {value}")

    # Draw pharmacophore features
    draw_pharmacophore_features(args.smiles)


# examples of usage using smiles
# python tools/prop_evaluation.py "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"