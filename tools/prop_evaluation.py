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


# examples of usage using smiles
# python tools/prop_evaluation.py "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"