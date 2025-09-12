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

from rdkit import RDConfig
from PIL import Image, ImageDraw, ImageFont

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import Draw, ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import rdCoordGen
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


def draw_pharmacophore_features(smiles: str, output_path: str | None = None):
    """
    Draw a molecule with automatically detected pharmacophore features highlighted,
    including a legend for the colors. The molecule is centered in the final image.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to visualize.
    output_path : str | None
        If provided, saves the image to this path; otherwise saves to 'pharmacophore_features.png'.

    Pharmacophore features highlighted
    ----------------------------------
    - HBD : Hydrogen Bond Donors (red)
    - HBA : Hydrogen Bond Acceptors (blue)
    - Aromatic : Aromatic ring atoms (purple)
    - PosIonizable : Positively ionizable groups (green)
    - NegIonizable : Negatively ionizable groups (orange)
    - Hydrophobe : Hydrophobic groups (brown)
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # improve feature detection
    rdCoordGen.AddCoords(mol)
    
    # Load default pharmacophore feature definitions
    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    features = factory.GetFeaturesForMol(mol)

    # Define colors and labels for each feature type
    feature_definitions = {
        "Donor": {"color": (1.0, 0.4, 0.4), "label": "HBD: Hydrogen Bond Donor"},
        "Acceptor": {"color": (0.4, 0.4, 1.0), "label": "HBA: Hydrogen Bond Acceptor"},
        "Aromatic": {"color": (0.8, 0.4, 0.8), "label": "Aromatic"},
        "PosIonizable": {"color": (0.4, 1.0, 0.4), "label": "Positive Ionizable"},
        "NegIonizable": {"color": (1.0, 0.7, 0.2), "label": "Negative Ionizable"},
        "Hydrophobe": {"color": (0.6, 0.4, 0.2), "label": "Hydrophobe"},
    }

    # Map atoms to feature colors
    highlight_atom_colors = {}
    for f in features:
        ftype = f.GetFamily()
        if ftype in feature_definitions:
            color = feature_definitions[ftype]["color"]
            for idx in f.GetAtomIds():
                highlight_atom_colors[idx] = color

    mol_img_size = (500, 500)
    mol_img = Draw.MolsToGridImage(
        [mol],
        highlightAtomLists=[list(highlight_atom_colors.keys())],
        highlightAtomColors=[highlight_atom_colors],
        subImgSize=mol_img_size,
        useSVG=False
    )

    legend_width = 250
    legend_padding = 10
    line_height = 25
    font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    legend_height = len(feature_definitions) * line_height + 2 * legend_padding
    legend_img = Image.new("RGB", (legend_width, mol_img_size[1]), "white")
    draw = ImageDraw.Draw(legend_img)
    y = legend_padding

    for ftype, props in feature_definitions.items():
        color = tuple(int(c * 255) for c in props["color"])
        label = props["label"]
        draw.rectangle([legend_padding, y, legend_padding + 15, y + 15], fill=color, outline="black")
        draw.text((legend_padding + 25, y), label, fill="black", font=font)
        y += line_height

    # --- 3. Combine molecule and legend ---
    total_width = mol_img.width + legend_width
    final_img = Image.new("RGB", (total_width, mol_img.height), "white")
    
    # Paste the molecule image, ensuring it's centered in its allocated space
    final_img.paste(mol_img, (0, 0))
    final_img.paste(legend_img, (mol_img.width, 0))

    if output_path is None:
        output_path = "pharmacophore_features.png"
    final_img.save(output_path)
    return output_path


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
