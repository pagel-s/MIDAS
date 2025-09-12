import gradio as gr
from gradio_molecule3d import Molecule3D
import torch
import tempfile
import os
from rdkit import Chem

import torch
from argparse import Namespace
import yaml
import sys
import numpy as np  
from pathlib import Path
from rdkit import Chem
import random

import sys
from pathlib import Path
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# from lightning_modules import LigandPocketDDPM
from main import load_model_using_config
from config import CKPT_PATH, CONFIG_YML


# Default visualization styles: single color for protein cartoon
reps = [
    {"model": 0, "style": "cartoon", "color": "grey"},  # single color for protein
    {"model": 1, "style": "stick", "color": "greenCarbon"},
]

CKPT_PATH = CKPT_PATH
CONFIG_YML = CONFIG_YML

model = load_model_using_config(CONFIG_YML, CKPT_PATH)


# -----------------------------
# Molecule generation functions
# -----------------------------
def generate_molecules(ligand_file, pdb_file, instructions, n_samples=1, n_nodes=15):
    """Generate molecules and return file paths."""
    with torch.no_grad():
        molecules = model.generate_ligands(
            pdb_file=pdb_file,
            n_samples=n_samples,
            text_description=instructions,
            ref_ligand=ligand_file,
            sanitize=True,
            largest_frag=True,
            relax_iter=200,
            n_nodes_min=n_nodes,
        )

    file_paths = []
    for i, mol in enumerate(molecules):
        save_path = os.path.join(tempfile.gettempdir(), f"generated_ligand_{i}.sdf")
        writer = Chem.SDWriter(save_path)
        writer.write(mol)
        writer.close()
        file_paths.append(save_path)

    return file_paths

def predict(ligand_file, pdb_file, instructions, n_samples=1, n_nodes=15):
    """Generate molecules and return initial view, file paths, and loading text."""
    # Show loading text
    loading = gr.update(value="Generating molecule... ⏳", visible=True)

    file_paths = generate_molecules(ligand_file, pdb_file, instructions, n_samples, n_nodes)
    first_molecule = Molecule3D([pdb_file, file_paths[0]], reps=reps)

    # Build default protein-ligand complex file
    complex_path = os.path.join(tempfile.gettempdir(), "complex_1.pdb")
    with open(complex_path, "w") as f_out:
        with open(pdb_file) as f_p:
            f_out.write(f_p.read())
        with open(file_paths[0]) as f_l:
            f_out.write("\n")
            f_out.write(f_l.read())

    # Hide loading text after generation
    loading_done = gr.update(value="", visible=False)

    return (
        first_molecule,
        gr.update(visible=True),
        file_paths,
        gr.update(minimum=1, maximum=len(file_paths), value=1, step=1, visible=True),
        file_paths[0],
        gr.update(visible=True),
        complex_path,
        gr.update(visible=True),
        loading_done,
    )

def update_view(index, pdb_file, file_paths):
    """Update Molecule3D view and download paths when user selects a ligand."""
    idx = int(index) - 1
    selected_ligand = file_paths[idx]

    # Build complex file
    complex_path = os.path.join(tempfile.gettempdir(), f"complex_{idx+1}.pdb")
    with open(complex_path, "w") as f_out:
        with open(pdb_file) as f_p:
            f_out.write(f_p.read())
        with open(selected_ligand) as f_l:
            f_out.write("\n")
            f_out.write(f_l.read())

    return Molecule3D([pdb_file, selected_ligand], reps=reps), selected_ligand, complex_path

# -----------------------------
# Gradio app
# -----------------------------
css = """
.centered {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
#main-content {
    max-width: 1200px;
    margin:auto;
}
"""

with gr.Blocks(css=css) as demo:
    # Title
    gr.Markdown("<h1 style='text-align:center'>🧬 MIDAS - Molecule Generator</h1>", elem_classes="centered")

    with gr.Column(elem_id="main-content", elem_classes="centered"):
        # Upload section
        with gr.Row():
            pdb_inp = gr.File(label="Upload PDB File", type="filepath", file_types=[".pdb"])
            ligand_inp = gr.File(label="Upload Reference Ligand File", type="filepath", file_types=[".sdf"])

        # Molecule viewer
        out = Molecule3D(label="Generated Molecule", reps=reps, visible=False)

        # Instruction input
        instruction_inp = gr.Textbox(
            label="Enter your instructions",
            placeholder="e.g., generate a ligand that binds strongly in pocket...",
        )

        # Sliders for number of samples and nodes
        with gr.Row():
            with gr.Column():
                n_samples_inp = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Samples")
            with gr.Column():
                n_nodes_inp = gr.Slider(minimum=10, maximum=20, value=15, step=1, label="Number of Nodes")

        # Slider to select which generated ligand to view
        ligand_selector = gr.Slider(label="Select Generated Ligand", minimum=1, maximum=1, value=1, step=1, visible=False)
        file_paths_state = gr.State([])

        # Download buttons
        download_ligand_btn = gr.File(label="Download Selected Ligand", visible=False, type="filepath")
        download_complex_btn = gr.File(label="Download Protein+Ligand Complex", visible=False, type="filepath")

        # Loading text
        loading_text = gr.Markdown("", visible=False, elem_classes="centered")

        # Generate button
        btn = gr.Button("Generate Molecules", variant="primary", elem_classes="orange-btn")
        btn.click(
            predict,
            inputs=[ligand_inp, pdb_inp, instruction_inp, n_samples_inp, n_nodes_inp],
            outputs=[out, out, file_paths_state, ligand_selector, download_ligand_btn, download_ligand_btn, download_complex_btn, download_complex_btn, loading_text],
        )

        # Update view when slider changes
        ligand_selector.change(
            update_view,
            inputs=[ligand_selector, pdb_inp, file_paths_state],
            outputs=[out, download_ligand_btn, download_complex_btn],
        )

if __name__ == "__main__":
    demo.launch()
