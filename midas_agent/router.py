import os
import sys
from typing import List, Optional, Dict, Any
import json


def _ensure_repo_on_path() -> None:
    """
    Ensure project root is on sys.path so local imports work when running as a script.
    """
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except Exception:
        repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.append(repo_root)


_ensure_repo_on_path()


def _lazy_import_langchain():
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool
    from langchain.agents import create_structured_chat_agent, AgentExecutor
    try:
        from langchain import hub
    except Exception:
        hub = None
    return BaseModel, Field, StructuredTool, create_structured_chat_agent, AgentExecutor, hub


# -----------------------------
# Tool wrappers
# -----------------------------

def _app_temp_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tmpdir = os.path.join(repo_root, "tmp", "midas_agent")
    os.makedirs(tmpdir, exist_ok=True)
    return tmpdir


def _calc_properties(smiles: str) -> dict:
    from tools.prop_evaluation import calculate_properties
    props = calculate_properties(smiles)
    return {"properties": props}


def _pubchem_similarity(smiles: str, num_results: int = 10, threshold: int = 85) -> dict:
    try:
        from tools.similarity_search import pubchem_similarity_search, cids_to_inchi
    except Exception as e:
        return {"error": f"similarity_search deps missing: {e}"}

    cids = pubchem_similarity_search(smiles, num_results=num_results, threshold=threshold)
    inchi_list = []
    try:
        inchi_list = cids_to_inchi(cids)
    except Exception:
        inchi_list = []
    results = []
    base = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    for i, cid in enumerate(cids):
        results.append({
            "cid": cid,
            "url": f"{base}{cid}",
            "inchi": inchi_list[i] if i < len(inchi_list) else None,
        })
    return {"results": results}


_MODEL_CACHE = {"model": None}


def _get_model():
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]
    try:
        from main import load_model_using_config
        from config import CKPT_PATH, CONFIG_YML
    except Exception as e:
        raise RuntimeError(f"Failed to import model loader/config: {e}")
    model = load_model_using_config(CONFIG_YML, CKPT_PATH)
    _MODEL_CACHE["model"] = model
    return model


def _generate_ligands_tool(pdb_file: str,
                            instructions: str,
                            n_samples: int = 1,
                            n_nodes_min: int = 15,
                            ref_ligand: Optional[str] = None,
                            sanitize: bool = True,
                            largest_frag: bool = True,
                            relax_iter: int = 200) -> dict:
    try:
        from rdkit import Chem
        import torch
    except Exception as e:
        return {"error": f"Required deps missing (torch/rdkit): {e}"}

    try:
        model = _get_model()
        with torch.no_grad():
            molecules = model.generate_ligands(
                pdb_file=pdb_file,
                n_samples=int(n_samples),
                text_description=instructions,
                ref_ligand=ref_ligand,
                sanitize=sanitize,
                largest_frag=largest_frag,
                relax_iter=int(relax_iter),
                n_nodes_min=int(n_nodes_min),
            )
        file_paths = []
        out_dir = _app_temp_dir()
        for i, mol in enumerate(molecules):
            save_path = os.path.join(out_dir, f"generated_ligand_{i}.sdf")
            writer = Chem.SDWriter(save_path)
            writer.write(mol)
            writer.close()
            file_paths.append(save_path)
        return {"generated_sdf_paths": file_paths, "count": len(file_paths)}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Context-aware tools for UI state
# -----------------------------
_UI_CONTEXT: Dict[str, Any] = {}


def set_ui_context(ctx: Dict[str, Any]) -> None:
    global _UI_CONTEXT
    _UI_CONTEXT = ctx or {}


def _get_ctx(key: str, default=None):
    return _UI_CONTEXT.get(key, default)


def tool_generate_current() -> str:
    pdb_path = _get_ctx("pdb_path")
    ref_sdf = _get_ctx("ref_sdf")
    instructions = _get_ctx("instructions", "Generate ligands for the provided pocket.")
    n_samples = int(_get_ctx("n_samples", 10))
    n_nodes_min = int(_get_ctx("n_nodes_min", 15))
    # print the context
    print(f"Context: {_UI_CONTEXT}")
    
    res = _generate_ligands_tool(
        pdb_file=pdb_path,
        instructions=instructions,
        n_samples=n_samples,
        n_nodes_min=n_nodes_min,
        ref_ligand=ref_sdf,
        sanitize=True,
        largest_frag=True,
        relax_iter=200,
    )
    return json.dumps({"type": "generate", "data": res})


def tool_props_current() -> str:
    try:
        from rdkit import Chem
    except Exception as e:
        return json.dumps({"type": "props", "data": {"error": str(e)}})
    lig = _get_ctx("selected_ligand")
    if not lig or not os.path.exists(lig):
        return json.dumps({"type": "props", "data": {"error": "No ligand selected"}})
    try:
        suppl = Chem.SDMolSupplier(lig, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return json.dumps({"type": "props", "data": {"error": "Failed to read SDF"}})
        from tools.prop_evaluation import calculate_properties
        smiles = Chem.MolToSmiles(mols[0])
        props = calculate_properties(smiles)
        props["SMILES"] = smiles
        return json.dumps({"type": "props", "data": props})
    except Exception as e:
        return json.dumps({"type": "props", "data": {"error": str(e)}})


def tool_similarity_current() -> str:
    try:
        from rdkit import Chem
    except Exception as e:
        return json.dumps({"type": "similarity", "data": {"error": str(e)}})
    lig = _get_ctx("selected_ligand")
    if not lig or not os.path.exists(lig):
        return json.dumps({"type": "similarity", "data": {"results": []}})
    try:
        suppl = Chem.SDMolSupplier(lig, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return json.dumps({"type": "similarity", "data": {"results": []}})
        smiles = Chem.MolToSmiles(mols[0])
        sim = _pubchem_similarity(smiles, num_results=10, threshold=85)
        return json.dumps({"type": "similarity", "data": sim})
    except Exception as e:
        return json.dumps({"type": "similarity", "data": {"error": str(e)}})


def tool_dock_current() -> str:
    pdb_path = _get_ctx("pdb_path")
    ref_sdf = _get_ctx("ref_sdf")
    lig = _get_ctx("selected_ligand")
    if not lig or not os.path.exists(lig):
        return json.dumps({"type": "dock", "data": {"error": "No ligand selected"}})
    try:
        from tools.docking import (
            prep_receptor,
            compute_center_from_ligand,
            prep_ligand,
            vina_dock,
        )
        box_size = [20, 20, 20]
        receptor_pdbqt = prep_receptor(pdb_path, pdb_path.replace(".pdb", ".pdbqt"))
        docking_center = compute_center_from_ligand(ref_sdf if ref_sdf else lig)
        lig_pdbqt = lig.replace(".sdf", ".pdbqt")
        prep_ligand(lig, lig_pdbqt)
        score = vina_dock(
            lig_pdbqt,
            receptor_pdbqt,
            docking_center,
            box_size=box_size,
            score_only=False,
            n_poses=5,
            exhaustiveness=32,
        )
        data = {
            "score": float(score),
            "receptor_pdbqt": receptor_pdbqt,
            "ligand_pdbqt": lig_pdbqt,
            "center": [float(x) for x in docking_center],
            "box_size": box_size,
        }
        return json.dumps({"type": "dock", "data": data})
    except Exception as e:
        return json.dumps({"type": "dock", "data": {"error": str(e)}})


def build_tools():
    BaseModel, Field, StructuredTool, _, _, _ = _lazy_import_langchain()

    class CalcPropsInput(BaseModel):
        smiles: str = Field(..., description="SMILES string of the molecule")

    class PubChemSimilarityInput(BaseModel):
        smiles: str = Field(..., description="SMILES string for similarity search")
        num_results: int = Field(10, description="Max number of similar compounds")
        threshold: int = Field(85, description="Similarity threshold (0-100)")

    class GenerateLigandsInput(BaseModel):
        pdb_file: str = Field(..., description="Path to protein PDB file")
        instructions: str = Field(..., description="Text instructions for generation")
        n_samples: int = Field(1, description="Number of ligands to generate")
        n_nodes_min: int = Field(15, description="Minimum number of nodes in generated ligands")
        ref_ligand: Optional[str] = Field(None, description="Optional reference ligand SDF path")
        sanitize: bool = Field(True, description="Sanitize molecules")
        largest_frag: bool = Field(True, description="Keep largest fragment only")
        relax_iter: int = Field(200, description="Relaxation iterations")

    class DockLigandInput(BaseModel):
        reference_sdf: Optional[str] = Field(None, description="Reference ligand SDF to set docking center")
        protein_pdb: str = Field(..., description="Path to protein PDB file")
        ligand_sdf: Optional[str] = Field(None, description="Path to ligand SDF to dock (provide this OR smiles)")
        smiles: Optional[str] = Field(None, description="Ligand SMILES to build 3D (provide this OR ligand_sdf)")
        score_only: Optional[bool] = Field(None, description="Override auto behavior: SDF->True, SMILES->False")
        box_size: Optional[List[float]] = Field(None, description="Docking box size [x,y,z]")
        exhaustiveness: int = Field(32, description="Vina exhaustiveness")
        n_poses: int = Field(5, description="Number of poses to generate")

    tools = [
        StructuredTool.from_function(
            func=_calc_properties,
            name="calc_properties",
            description=(
                "Calculate basic pharmacophoric properties (MW, LogP, HBA, HBD, TPSA, QED) "
                "for a given SMILES string. Returns a JSON with properties."
            ),
            args_schema=CalcPropsInput,
        ),
        StructuredTool.from_function(
            func=_pubchem_similarity,
            name="pubchem_similarity",
            description=(
                "Search PubChem for molecules similar to a SMILES. "
                "Returns CIDs, optional InChI, and URLs."
            ),
            args_schema=PubChemSimilarityInput,
        ),
        StructuredTool.from_function(
            func=_generate_ligands_tool,
            name="generate_ligands",
            description=(
                "Generate ligands for a given protein PDB conditioned on text instructions. "
                "Optionally provide a reference ligand SDF. Returns paths to generated SDF files."
            ),
            args_schema=GenerateLigandsInput,
        ),
        StructuredTool.from_function(
            func=tool_generate_current,
            name="generate_current",
            description=(
                "Generate ligands using the CURRENT UI context (pdb/ref/instructions) with defaults."
            ),
        ),
        StructuredTool.from_function(
            func=tool_props_current,
            name="props_current",
            description=(
                "Compute properties for the CURRENTLY SELECTED ligand from the UI."
            ),
        ),
        StructuredTool.from_function(
            func=tool_similarity_current,
            name="similarity_current",
            description=(
                "Run PubChem similarity for the CURRENTLY SELECTED ligand from the UI."
            ),
        ),
        StructuredTool.from_function(
            func=tool_dock_current,
            name="dock_current",
            description=(
                "Dock the CURRENTLY SELECTED ligand against the CURRENT protein using Vina."
            ),
        ),
    ]
    return tools


def build_context_tools():
    BaseModel, Field, StructuredTool, _, _, _ = _lazy_import_langchain()
    tools = [
        StructuredTool.from_function(func=tool_generate_current, name="generate_current",
                                     description="Generate ligands using the CURRENT UI context (pdb/ref/instructions)."),
        StructuredTool.from_function(func=tool_props_current, name="props_current",
                                     description="Compute properties for the CURRENTLY SELECTED ligand from the UI."),
        StructuredTool.from_function(func=tool_similarity_current, name="similarity_current",
                                     description="Run PubChem similarity for the CURRENTLY SELECTED ligand from the UI."),
        StructuredTool.from_function(func=tool_dock_current, name="dock_current",
                                     description="Dock the CURRENTLY SELECTED ligand against the CURRENT protein using Vina."),
    ]
    return tools


def get_agent(llm=None, verbose: bool = True, context_only: bool = False):
    BaseModel, Field, StructuredTool, create_structured_chat_agent, AgentExecutor, hub = _lazy_import_langchain()
    tools = build_context_tools() if context_only else build_tools()
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as e:
            raise RuntimeError(
                f"No LLM provided and failed to create default OpenAI LLM: {e}. "
                f"Set OPENAI_API_KEY or pass a custom llm."
            )

    prompt = None
    if hub is not None:
        try:
            prompt = hub.pull("hwchase17/structured-chat-agent")
        except Exception:
            prompt = None

    if prompt is None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.prompts import MessagesPlaceholder
        sys_msg = (
            "You are a helpful agent for molecular design.\n"
            "You have access to tools that operate on the CURRENT UI context (protein path, reference path, currently selected ligand).\n"
            "Always use the provided tools to take actions; do not fabricate file paths."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

    agent_runnable = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=True,
    )
    return agent_executor


def route(query: str) -> str:
    agent = get_agent()
    result = agent.invoke({"input": query})
    if isinstance(result, dict) and "output" in result:
        return result["output"]
    return str(result)


def run_agent_with_context(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Set UI context, run the agent, return output and intermediate steps."""
    set_ui_context(context)
    agent = get_agent(context_only=True)
    res = agent.invoke({"input": prompt})
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDAS Agent CLI")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the agent. If empty, starts REPL.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    args = parser.parse_args()

    if args.prompt:
        text = " ".join(args.prompt)
        print(route(text))
    else:
        print("Starting MIDAS Agent REPL. Type 'exit' to quit.")
        agent = get_agent(verbose=args.verbose)
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in {"exit", "quit"}:
                break
            out = agent.run(user)
            print(f"Agent: {out}")


