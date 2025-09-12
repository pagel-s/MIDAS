import os
import sys
from typing import List, Optional


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
        # pubchempy may be missing; still return CIDs
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


def _dock_ligand(reference_sdf: str,
                  protein_pdb: str,
                  ligand_sdf: Optional[str] = None,
                  smiles: Optional[str] = None,
                  score_only: Optional[bool] = None,
                  box_size: Optional[List[float]] = None,
                  exhaustiveness: int = 32,
                  n_poses: int = 5) -> dict:
    try:
        from tools.docking import (
            prep_receptor,
            compute_center_from_ligand,
            prep_ligand,
            vina_dock,
        )
    except Exception as e:
        return {"error": f"docking deps missing: {e}"}

    if box_size is None:
        box_size = [20, 20, 20]

    # Validate input choice
    if (ligand_sdf is None and not smiles) or (ligand_sdf and smiles):
        return {"error": "Provide exactly one of ligand_sdf or smiles"}

    used_ligand_sdf = ligand_sdf
    generated_from_smiles = False

    # If SMILES provided, build 3D SDF first
    if smiles:
        try:
            import tempfile
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception as e:
            return {"error": f"RDKit required to build 3D from SMILES: {e}"}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 1337
            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                return {"error": "Failed to embed 3D conformer from SMILES"}
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            tmp_sdf = os.path.join(tempfile.gettempdir(), "ligand_from_smiles.sdf")
            writer = Chem.SDWriter(tmp_sdf)
            writer.write(mol)
            writer.close()
            used_ligand_sdf = tmp_sdf
            generated_from_smiles = True
        except Exception as e:
            return {"error": f"Failed to generate 3D from SMILES: {e}"}

    # Determine effective score_only behavior
    # Default: SDF -> True, SMILES (built 3D) -> False
    score_only_effective = True if not generated_from_smiles else False
    if score_only is not None:
        score_only_effective = bool(score_only)

    try:
        receptor_pdbqt = prep_receptor(protein_pdb, protein_pdb.replace(".pdb", ".pdbqt"))
        docking_center = compute_center_from_ligand(reference_sdf)
        lig_pdbqt = used_ligand_sdf.replace(".sdf", ".pdbqt")
        prep_ligand(used_ligand_sdf, lig_pdbqt)
        score = vina_dock(
            lig_pdbqt,
            receptor_pdbqt,
            docking_center,
            box_size=box_size,
            score_only=score_only_effective,
            n_poses=n_poses,
            exhaustiveness=exhaustiveness,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "score": float(score),
        "receptor_pdbqt": receptor_pdbqt,
        "ligand_pdbqt": lig_pdbqt,
        "used_ligand_sdf": used_ligand_sdf,
        "generated_from_smiles": generated_from_smiles,
        "center": [float(x) for x in docking_center],
        "box_size": box_size,
    }


def build_tools():
    BaseModel, Field, StructuredTool, _, _, _ = _lazy_import_langchain()

    class CalcPropsInput(BaseModel):
        smiles: str = Field(..., description="SMILES string of the molecule")

    class PubChemSimilarityInput(BaseModel):
        smiles: str = Field(..., description="SMILES string for similarity search")
        num_results: int = Field(10, description="Max number of similar compounds")
        threshold: int = Field(85, description="Similarity threshold (0-100)")

    class DockLigandInput(BaseModel):
        reference_sdf: str = Field(..., description="Path to reference ligand SDF to set docking center")
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
            func=_dock_ligand,
            name="dock_ligand",
            description=(
                "Dock a ligand into a protein PDB using Vina. Input can be an SDF file (scoring only) "
                "or a SMILES string (build 3D first and dock). Provide reference SDF to set docking center."
            ),
            args_schema=DockLigandInput,
        ),
    ]
    return tools


def get_agent(llm=None, verbose: bool = True):
    BaseModel, Field, StructuredTool, create_structured_chat_agent, AgentExecutor, hub = _lazy_import_langchain()
    tools = build_tools()
    if llm is None:
        # Default to OpenAI via langchain if API key is available
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as e:
            raise RuntimeError(
                f"No LLM provided and failed to create default OpenAI LLM: {e}. "
                f"Set OPENAI_API_KEY or pass a custom llm."
            )

    # Use the structured chat agent which supports multi-argument tools
    prompt = None
    if hub is not None:
        try:
            prompt = hub.pull("hwchase17/structured-chat-agent")
        except Exception:
            prompt = None

    if prompt is None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.prompts import MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful agent for molecular design. Use the provided tools when helpful. If a tool fails, explain the failure clearly."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

    agent_runnable = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=verbose)
    return agent_executor


def route(query: str) -> str:
    agent = get_agent()
    result = agent.invoke({"input": query})
    # AgentExecutor returns a dict with 'output'
    if isinstance(result, dict) and "output" in result:
        return result["output"]
    return str(result)


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


