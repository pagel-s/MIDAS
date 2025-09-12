import gradio as gr
from gradio_molecule3d import Molecule3D
import torch
import tempfile
import os
from rdkit import Chem
import requests
import json
from PIL import Image

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

from main import load_model_using_config
from config import CKPT_PATH, CONFIG_YML
from midas_agent.router import run_agent_with_context
from tools.prop_evaluation import calculate_properties, draw_pharmacophore_features
from midas_agent.router import _pubchem_similarity
from midas_agent.router import _dock_ligand


reps = [
    {"model": 0, "style": "cartoon", "color": "orange"},
    {"model": 1, "style": "stick",   "color": "cyanCarbon"},
]

CKPT_PATH = CKPT_PATH
CONFIG_YML = CONFIG_YML

model = load_model_using_config(CONFIG_YML, CKPT_PATH)


# -----------------------------
# Minimal agent actions
# -----------------------------

def _extract_updates_from_steps(steps):
    props_update = None
    sim_update = None
    dock_update = None
    gen_paths = None
    retro_update = None
    try:
        for s in steps or []:
            observation = None
            if isinstance(s, (list, tuple)) and len(s) == 2:
                observation = s[1]
            elif isinstance(s, dict):
                observation = s.get("observation")
            if observation is None:
                continue
            try:
                data = observation if isinstance(observation, dict) else json.loads(str(observation))
            except Exception:
                continue
            t = data.get("type")
            d = data.get("data", {})
            if t == "generate":
                paths = d.get("generated_sdf_paths", [])
                if paths:
                    gen_paths = paths
            elif t == "props":
                props_update = d
            elif t == "similarity":
                sim_update = d
            elif t == "dock":
                dock_update = d
            elif t == "retrosyn":
                retro_update = d
    except Exception:
        pass
    return props_update, sim_update, dock_update, gen_paths, retro_update


def agent_send(user_message, history, pdb_state, ref_state, slider_value, gen_paths_state):
    if not user_message:
        return (
            history, history, gr.update(), gr.update(), gr.update(),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        )
    pdb_path = _copy_to_tmp(pdb_state)
    ref_sdf = _copy_to_tmp(ref_state)

    selected_idx = 0
    try:
        selected_idx = int(slider_value) - 1
    except Exception:
        selected_idx = 0
    sel_lig = None
    if gen_paths_state and isinstance(gen_paths_state, list) and len(gen_paths_state) > 0:
        if 0 <= selected_idx < len(gen_paths_state):
            sel_lig = gen_paths_state[selected_idx]
        else:
            sel_lig = gen_paths_state[0]

    ctx = {
        "pdb_path": pdb_path,
        "ref_sdf": ref_sdf,
        "selected_ligand": sel_lig,
        "instructions": user_message,
        "n_samples": 10,
        "n_nodes_min": 15,
    }

    res = run_agent_with_context(user_message, ctx)
    output_text = res.get("output", "")
    steps = res.get("intermediate_steps", [])

    props_update, sim_update, dock_update, gen_paths, retro_update = _extract_updates_from_steps(steps)

    out_view_update = gr.update()
    slider_update = gr.update()
    gen_paths_state_update = gr.update()
    props_img_update = gr.update(visible=False)

    if gen_paths:
        sizes = []
        for p in gen_paths:
            try:
                ms = Chem.SDMolSupplier(p, removeHs=False)
                mols = [m for m in ms if m is not None]
                sizes.append(mols[0].GetNumAtoms() if mols else 0)
            except Exception:
                sizes.append(0)
        idx_best = int(max(range(len(gen_paths)), key=lambda i: sizes[i])) if gen_paths else 0
        out_view_update = gr.update(value=[pdb_path, gen_paths[idx_best]], visible=True)
        slider_update = gr.update(visible=True, minimum=1, maximum=len(gen_paths), value=idx_best + 1, step=1)
        gen_paths_state_update = gen_paths
        output_text = f"Generated {len(gen_paths)} molecule(s). Use the slider to browse"

    sim_gallery_update = gr.update(visible=False)
    if sim_update and isinstance(sim_update, dict):
        results = sim_update.get("results", [])
        items = []
        for r in results[:10]:
            cid = r.get("cid")
            if cid is None:
                continue
            smi = None
            try:
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                resp = requests.get(prop_url, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("PropertyTable", {}).get("Properties", [])
                    if props:
                        smi = props[0].get("CanonicalSMILES")
            except Exception:
                smi = None
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=300x300"
            caption = f"CID {cid}\n{(smi or '')}"
            items.append([img_url, caption])
        sim_gallery_update = gr.update(value=items, visible=True)
        output_text = f"Found {len(items)} similar molecule(s)."

    props_json_update = gr.update(visible=False)
    if props_update:
        props_json_update = gr.update(value=props_update, visible=True)
        img_path = props_update.get("pharmacophore_image") if isinstance(props_update, dict) else None
        if img_path and os.path.exists(img_path):
            props_img_update = gr.update(value=Image.open(img_path), visible=True)
        else:
            props_img_update = gr.update(visible=False)
        output_text = "Computed properties for the current molecule."

    dock_json_update = gr.update(visible=False)
    if dock_update:
        dock_json_update = gr.update(value=dock_update, visible=True)
        try:
            score = dock_update.get("score")
            if score is not None:
                output_text = f"Docking completed. Score: {float(score):.2f}"
            else:
                output_text = "Docking completed."
        except Exception:
            output_text = "Docking completed."

    retro_gallery_update = gr.update(visible=False)
    if retro_update and isinstance(retro_update, dict):
        imgs = retro_update.get("images", [])
        if imgs:
            # Convert to [[path, caption], ...]
            items = [[p, os.path.basename(p)] for p in imgs if os.path.exists(p)]
            if items:
                retro_gallery_update = gr.update(value=items, visible=True)
                output_text = f"Retrosynthesis completed. {len(items)} step image(s)."

    new_hist = history + [(user_message, output_text or "Done.")]
    return (
        new_hist,
        new_hist,
        out_view_update,
        slider_update,
        gen_paths_state_update if gen_paths_state_update else gr.update(),
        props_json_update,
        sim_gallery_update,
        dock_json_update,
        props_img_update,
        retro_gallery_update,
    )


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
        return gr.update(value={"error": "No ligand selected"}, visible=True)
    try:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return gr.update(value={"error": "Failed to read SDF"}, visible=True)
        smiles = Chem.MolToSmiles(mols[0])
        props = calculate_properties(smiles)
        props["SMILES"] = smiles
        # create pharmacophore image
        img_path = os.path.join(_app_temp_dir(), "pharmacophore_features.png")
        try:
            draw_pharmacophore_features(smiles, output_path=img_path)
            props["pharmacophore_image"] = img_path
        except Exception:
            props["pharmacophore_image"] = None
        return gr.update(value=props, visible=True)
    except Exception as e:
        return gr.update(value={"error": str(e)}, visible=True)


def do_similarity(index, paths_state):
    path = _get_selected_path(index, paths_state)
    if not path or not os.path.exists(path):
        return gr.update(value=[], visible=True)
    try:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return gr.update(value=[], visible=True)
        smiles = Chem.MolToSmiles(mols[0])
        out = _pubchem_similarity(smiles, num_results=10, threshold=85)
        results = out.get("results", []) if isinstance(out, dict) else []
        items = []
        for r in results[:10]:
            cid = r.get("cid")
            if cid is None:
                continue
            smi = None
            try:
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                resp = requests.get(prop_url, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("PropertyTable", {}).get("Properties", [])
                    if props:
                        smi = props[0].get("CanonicalSMILES")
            except Exception:
                smi = None
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=300x300"
            caption = f"CID {cid}\n{(smi or '')}"
            items.append([img_url, caption])
        return gr.update(value=items, visible=True)
    except Exception:
        return gr.update(value=[], visible=True)


def do_dock(index, pdb_state, ref_state, paths_state):
    path = _get_selected_path(index, paths_state)
    pdb_path = _normalize_path(pdb_state)
    ref_path = _normalize_path(ref_state)
    if not path or not os.path.exists(path):
        return gr.update(value={"error": "No ligand selected"}, visible=True)
    if not pdb_path or not os.path.exists(pdb_path):
        return gr.update(value={"error": "No protein uploaded"}, visible=True)
    try:
        result = _dock_ligand(
            reference_sdf=ref_path,
            protein_pdb=pdb_path,
            ligand_sdf=path,
            smiles=None,
            score_only=False,
        )
        return gr.update(value=result, visible=True)
    except Exception as e:
        return gr.update(value={"error": str(e)}, visible=True)


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
    gr.Markdown("<h1 style='text-align:center'>🧬 MIDAS</h1>", elem_classes="centered")

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

        # with gr.Row():
        #     props_btn = gr.Button("Properties")
        #     sim_btn = gr.Button("PubChem Similarity")
        #     dock_btn = gr.Button("Dock")

        props_json = gr.JSON(label="Properties", visible=False)
        props_img = gr.Image(label="Pharmacophore", visible=False)
        sim_gallery = gr.Gallery(label="Similar molecules (CID + SMILES)", columns=5, height=340, visible=False)
        dock_json = gr.JSON(label="Docking", visible=False)
        retro_gallery = gr.Gallery(label="Retrosynthesis routes", columns=2, height=340, visible=False)

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
            inputs=[chat_input, chat_history, pdb_state, ref_state, ligand_selector, gen_paths_state],
            outputs=[chat, chat_history, out_view, ligand_selector, gen_paths_state, props_json, sim_gallery, dock_json, props_img, retro_gallery],
        ).then(lambda: "", None, [chat_input])

        ligand_selector.change(
            on_slider_change,
            inputs=[ligand_selector, pdb_state, gen_paths_state],
            outputs=[out_view],
        )

        # props_btn.click(
        #     do_props,
        #     inputs=[ligand_selector, gen_paths_state],
        #     outputs=[props_json],
        # ).then(
        #     lambda d: gr.update(value=Image.open(d.get("pharmacophore_image")), visible=True) if isinstance(d, dict) and d.get("pharmacophore_image") and os.path.exists(d.get("pharmacophore_image")) else gr.update(visible=False),
        #     inputs=[props_json],
        #     outputs=[props_img]
        # )

        # sim_btn.click(
        #     do_similarity,
        #     inputs=[ligand_selector, gen_paths_state],
        #     outputs=[sim_gallery],
        # )

        # dock_btn.click(
        #     do_dock,
        #     inputs=[ligand_selector, pdb_state, ref_state, gen_paths_state],
        #     outputs=[dock_json],
        # )

if __name__ == "__main__":
    demo.launch()
