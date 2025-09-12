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

# Configure project-local temp dirs to avoid /tmp permission issues
TMP_ROOT = os.path.join(REPO_ROOT, "tmp")
GRADIO_TMP_DIR = os.path.join(TMP_ROOT, "gradio")
os.makedirs(GRADIO_TMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP_DIR

def _app_temp_dir():
    d = os.path.join(TMP_ROOT, "frontend")
    os.makedirs(d, exist_ok=True)
    return d

def _normalize_path(p):
    try:
        if isinstance(p, dict) and "path" in p:
            return p["path"]
    except Exception:
        pass
    return p

def _copy_to_tmp(src_path):
    import shutil
    if not src_path:
        return None
    src_path = _normalize_path(src_path)
    if not os.path.exists(src_path):
        return None
    dst_dir = _app_temp_dir()
    base = os.path.basename(src_path)
    dst = os.path.join(dst_dir, base)
    try:
        shutil.copy(src_path, dst)
        return dst
    except Exception:
        return src_path

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# from lightning_modules import LigandPocketDDPM
from main import load_model_using_config
from config import CKPT_PATH, CONFIG_YML
from midas_agent.router import route as agent_route
from midas_agent.router import _generate_ligands_tool, _pubchem_similarity, _dock_ligand
from tools.prop_evaluation import calculate_properties


reps = [
    {"model": 0, "style": "cartoon", "color": "blue"}, 
    {"model": 1, "style": "stick", "color": "greenCarbon"},
]

CKPT_PATH = CKPT_PATH
CONFIG_YML = CONFIG_YML

model = load_model_using_config(CONFIG_YML, CKPT_PATH)


# -----------------------------
# Minimal agent actions
# -----------------------------

def agent_send(user_message, history, pdb_state, ref_state):
    if not user_message:
        return history, history, gr.update(), gr.update(), gr.update()
    pdb_path = _copy_to_tmp(pdb_state)
    ref_sdf = _copy_to_tmp(ref_state)
    text = (user_message or "").lower()

    # Single tool: generate molecules when asked
    if any(k in text for k in ["generate", "design", "sample", "create"]):
        if not pdb_path or not os.path.exists(pdb_path):
            reply = "Please upload a protein PDB first."
            new_hist = history + [(user_message, reply)]
            return new_hist, new_hist, gr.update(), gr.update(), gr.update()
        result = _generate_ligands_tool(
            pdb_file=pdb_path,
            instructions=user_message,
            n_samples=10,
            n_nodes_min=15,
            ref_ligand=ref_sdf,
            sanitize=True,
            largest_frag=True,
            relax_iter=200,
        )
        if "error" in result:
            reply = f"Generation failed: {result['error']}"
            new_hist = history + [(user_message, reply)]
            return new_hist, new_hist, gr.update(), gr.update(), gr.update()
        file_paths = result.get("generated_sdf_paths", [])
        if not file_paths:
            reply = "No molecules generated."
            new_hist = history + [(user_message, reply)]
            return new_hist, new_hist, gr.update(), gr.update(), gr.update()
        # choose largest
        sizes = []
        for p in file_paths:
            try:
                ms = Chem.SDMolSupplier(p, removeHs=False)
                mols = [m for m in ms if m is not None]
                sizes.append(mols[0].GetNumAtoms() if mols else 0)
            except Exception:
                sizes.append(0)
        idx_best = int(max(range(len(file_paths)), key=lambda i: sizes[i])) if file_paths else 0
        viewer = gr.update(value=[pdb_path, file_paths[idx_best]], visible=True)
        slider = gr.update(visible=True, minimum=1, maximum=len(file_paths), value=idx_best + 1, step=1)
        reply = f"Generated {len(file_paths)} molecule(s). Use the slider to browse; showing the largest complex."
        new_hist = history + [(user_message, reply)]
        return new_hist, new_hist, viewer, slider, file_paths

    # General chat fallback with context
    context = f"Protein PDB file: {pdb_path}. Reference ligand SDF: {ref_sdf}. "
    try:
        reply = agent_route(context + user_message)
    except Exception as e:
        reply = f"Agent error: {e}"
    new_hist = history + [(user_message, reply)]
    return new_hist, new_hist, gr.update(), gr.update(), gr.update()


def on_pdb_upload(p):
    path = _copy_to_tmp(p)
    if not path:
        return gr.update(), ""
    return gr.update(value=[path], visible=True), path


def on_ref_upload(r):
    path = _copy_to_tmp(r)
    return path or ""


def on_slider_change(index, pdb_state, paths_state):
    try:
        idx = int(index) - 1
    except Exception:
        idx = 0
    pdb_path = _normalize_path(pdb_state)
    all_paths = paths_state or []
    if not all_paths:
        return gr.update()
    if idx < 0 or idx >= len(all_paths):
        idx = 0
    return gr.update(value=[pdb_path, all_paths[idx]], visible=True)


def _get_selected_path(index, paths_state):
    try:
        idx = int(index) - 1
    except Exception:
        idx = 0
    paths = paths_state or []
    if not paths:
        return None
    if idx < 0 or idx >= len(paths):
        idx = 0
    return paths[idx]


def do_props(index, paths_state):
    path = _get_selected_path(index, paths_state)
    if not path or not os.path.exists(path):
        return {"error": "No ligand selected"}
    try:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return {"error": "Failed to read SDF"}
        smiles = Chem.MolToSmiles(mols[0])
        props = calculate_properties(smiles)
        props["SMILES"] = smiles
        return props
    except Exception as e:
        return {"error": str(e)}


def do_similarity(index, paths_state):
    path = _get_selected_path(index, paths_state)
    print("path:", path)
    if not path or not os.path.exists(path):
        return {"error": "No ligand selected"}
    try:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return {"error": "Failed to read SDF"}
        smiles = Chem.MolToSmiles(mols[0])
        print("smiles:", smiles)
        out = _pubchem_similarity(smiles, num_results=10, threshold=85)
        print("out:", out)
        return out
    except Exception as e:
        return {"error": str(e)}


def do_dock(index, pdb_state, ref_state, paths_state):
    path = _get_selected_path(index, paths_state)
    pdb_path = _normalize_path(pdb_state)
    ref_path = _normalize_path(ref_state)
    if not path or not os.path.exists(path):
        return {"error": "No ligand selected"}
    if not pdb_path or not os.path.exists(pdb_path):
        return {"error": "No protein uploaded"}
    try:
        result = _dock_ligand(
            reference_sdf=ref_path,
            protein_pdb=pdb_path,
            ligand_sdf=path,
            smiles=None,
            score_only=True,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Minimal Gradio app
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
    gr.Markdown("<h1 style='text-align:center'>🧬 MIDAS - Agentic Molecule Generator</h1>", elem_classes="centered")

    with gr.Column(elem_id="main-content", elem_classes="centered"):
        with gr.Row():
            pdb_inp = gr.File(label="Upload Protein PDB File", type="filepath", file_types=[".pdb"])
            ref_inp = gr.File(label="Upload Reference Ligand (optional)", type="filepath", file_types=[".sdf"])

        out_view = Molecule3D(label="Protein / Complex", reps=reps, visible=False)

        chat = gr.Chatbot(height=300)
        chat_input = gr.Textbox(label="Chat with Agent", placeholder="Describe the molecule to generate, or ask questions...")
        chat_history = gr.State([])
        pdb_state = gr.State("")
        ref_state = gr.State("")
        gen_paths_state = gr.State([])

        ligand_selector = gr.Slider(label="Select Generated Ligand", minimum=1, maximum=1, value=1, step=1, visible=False)

        with gr.Row():
            props_btn = gr.Button("Properties")
            sim_btn = gr.Button("PubChem Similarity")
            dock_btn = gr.Button("Dock")

        props_json = gr.JSON(label="Properties")
        sim_json = gr.JSON(label="Similarity")
        dock_json = gr.JSON(label="Docking")

        send_btn = gr.Button("Send")

        pdb_inp.change(
            on_pdb_upload,
            inputs=[pdb_inp],
            outputs=[out_view, pdb_state],
        )

        ref_inp.change(
            on_ref_upload,
            inputs=[ref_inp],
            outputs=[ref_state],
        )

        send_btn.click(
            agent_send,
            inputs=[chat_input, chat_history, pdb_state, ref_state],
            outputs=[chat, chat_history, out_view, ligand_selector, gen_paths_state],
        ).then(lambda: "", None, [chat_input])

        ligand_selector.change(
            on_slider_change,
            inputs=[ligand_selector, pdb_state, gen_paths_state],
            outputs=[out_view],
        )

        props_btn.click(
            do_props,
            inputs=[ligand_selector, gen_paths_state],
            outputs=[props_json],
        )

        sim_btn.click(
            do_similarity,
            inputs=[ligand_selector, gen_paths_state],
            outputs=[sim_json],
        )

        dock_btn.click(
            do_dock,
            inputs=[ligand_selector, pdb_state, ref_state, gen_paths_state],
            outputs=[dock_json],
        )

if __name__ == "__main__":
    demo.launch()
