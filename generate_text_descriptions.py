import argparse
import os
import csv
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import multiprocessing as mp

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rdMD, rdmolops
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import exmol
import random

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


def sdf_to_smiles_list(sdf_path: Path) -> List[str]:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    smiles_list: List[str] = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        smi = Chem.MolToSmiles(mol)
        if smi:
            smiles_list.append(smi)
    return smiles_list


# -------------------- NEW: Pharmacophore and PhysChem helpers --------------------
_PHARM_FACTORY = None


def _get_pharmacophore_factory():
    global _PHARM_FACTORY
    if _PHARM_FACTORY is None:
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        _PHARM_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    return _PHARM_FACTORY


def get_pharmacophore_counts(mol: Chem.Mol) -> Dict[str, int]:
    factory = _get_pharmacophore_factory()
    feats = factory.GetFeaturesForMol(mol)
    counts: Dict[str, int] = {}
    for f in feats:
        fam = f.GetFamily()
        counts[fam] = counts.get(fam, 0) + 1
    return counts


def describe_pharmacophores(counts: Dict[str, int]) -> str:
    if not counts:
        return "Pharmacophore features: none identified."
    # Order common families for readability if present
    order = [
        'Donor', 'Acceptor', 'Aromatic', 'Hydrophobe',
        'PosIonizable', 'NegIonizable', 'LumpedHydrophobe', 'ZnBinder'
    ]
    parts = []
    for fam in order:
        if fam in counts and counts[fam] > 0:
            parts.append(f"{fam.lower()}:{counts[fam]}")
    # Add any remaining families
    for fam, val in counts.items():
        if fam not in order and val > 0:
            parts.append(f"{fam.lower()}:{val}")
    return "Pharmacophores: " + (", ".join(parts) if parts else "none identified.")


def calc_physchem_props(mol: Chem.Mol) -> Dict[str, float]:
    props: Dict[str, float] = {}
    props['MW'] = float(Descriptors.MolWt(mol))
    props['logP'] = float(Crippen.MolLogP(mol))
    props['TPSA'] = float(rdMD.CalcTPSA(mol))
    props['HBD'] = int(rdMD.CalcNumHBD(mol))
    props['HBA'] = int(rdMD.CalcNumHBA(mol))
    props['RotB'] = int(rdMD.CalcNumRotatableBonds(mol))
    props['Rings'] = int(rdMD.CalcNumRings(mol))
    props['AromRings'] = int(rdMD.CalcNumAromaticRings(mol))
    props['fSP3'] = float(rdMD.CalcFractionCSP3(mol))
    try:
        props['FormalCharge'] = int(rdmolops.GetFormalCharge(mol))
    except Exception:
        props['FormalCharge'] = 0
    return props


def describe_physchem(props: Dict[str, float]) -> str:
    return (
        "Properties: "
        f"MW {props['MW']:.1f}, logP {props['logP']:.2f}, TPSA {props['TPSA']:.1f} Å^2, "
        f"HBD {props['HBD']}, HBA {props['HBA']}, rotatable bonds {props['RotB']}, "
        f"rings {props['Rings']} (aromatic {props['AromRings']}), fSP3 {props['fSP3']:.2f}, "
        f"formal charge {props['FormalCharge']}"
    )


# -------------------- NEW: LLM renderers for pharm + physchem --------------------
def describe_pharmacophores_llm(counts: Dict[str, int], client: Optional[object], model: str) -> str:
    if client is None:
        return ""
    parts = []
    for fam, val in sorted(counts.items()):
        if val:
            parts.append(f"{fam}:{val}")
    pharm_str = ", ".join(parts) if parts else "none"
    # system_msg = (
    #     "You are a medicinal chemist. Write a single concise sentence describing the molecule's pharmacophore profile "
    #     "in qualitative terms for a generation model. Avoid listing raw counts; translate into phrases like 'multiple hydrogen-bond donors', 'aromatic character', 'hydrophobic regions'."
    # )
    # user_msg = f"Pharmacophore counts (family:count): {pharm_str}"
    
    import random
    opener = [
        "Design",
        "Target",
        "Prioritize",
        "Seek",
        "Synthesize",
        "Aim for",
        "Generate",
        "A",
        "Generate a molecule with",
        "Create",
        "Construct",
        "Devise",
        "Produce",
        "Assemble",
        "Formulate",
        "Invent",
        "Compose",
        "Engineer",
        "Draft",
        "Model",
        "Simulate",
        "Propose",
        "Derive",
        "Fabricate",
        "Conceive",
        "Iteratively generate",
        "Augment",
        "Mutate",
        "Explore",
        "Optimize generation of",
        "Generate variations of",
        "Generate candidates for",
        "Expand",
        "Refine generated",
        "Transform",
        "Generate analogs of",
        "Generate potential",
        "Generate configurations of",
        "Automate generation of"
    ]
    random.shuffle(opener)

    system_msg = (
    "As a medicinal chemist, describe the molecule's pharmacophore profile in one concise, qualitative sentence. "
    f"Use phrases like 'multiple hydrogen-bond donors', 'aromatic character', and 'hydrophobic regions' instead of numerical counts.  Use an opener inspired by: {opener}."
)
    user_msg = f"Pharmacophore counts (family:count): {pharm_str}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def describe_physchem_llm(props: Dict[str, float], client: Optional[object], model: str) -> str:
    if client is None:
        return ""
    # make a compact props string
    keys = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AromRings", "fSP3", "FormalCharge"]
    kv = ", ".join(f"{k}={props.get(k)}" for k in keys if k in props)
    # system_msg = (
    #     "You are a medicinal chemist. Summarize the physicochemical profile as a single prescriptive sentence for a generator. "
    #     "Do not repeat exact numbers; map them to qualitative aims (e.g., 'moderate lipophilicity', 'low polar surface area', 'few rotors')."
    # )
    # user_msg = f"Properties: {kv}"
        
    import random
    opener = [
        "Design",
        "Target",
        "Prioritize",
        "Seek",
        "Synthesize",
        "Aim for",
        "Generate",
        "A",
        "Generate a molecule with",
        "Create",
        "Construct",
        "Devise",
        "Produce",
        "Assemble",
        "Formulate",
        "Invent",
        "Compose",
        "Engineer",
        "Draft",
        "Model",
        "Simulate",
        "Propose",
        "Derive",
        "Fabricate",
        "Conceive",
        "Iteratively generate",
        "Augment",
        "Mutate",
        "Explore",
        "Optimize generation of",
        "Generate variations of",
        "Generate candidates for",
        "Expand",
        "Refine generated",
        "Transform",
        "Generate analogs of",
        "Generate potential",
        "Generate configurations of",
        "Automate generation of"
    ]
    random.shuffle(opener)

    system_msg = (
    "You are a medicinal chemist. Generate a single, prescriptive sentence guiding an AI molecule generator on the desired physicochemical profile. "
    f"Replace specific numerical values with qualitative descriptions, e.g., 'moderate lipophilicity', 'low polar surface area', 'limited rotatable bonds' . Use an opener inspired by: {opener}."
)
    user_msg = f"Properties: {kv}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def extract_functional_group_names(smiles: str) -> List[str]:
    try:
        fgs = exmol.get_functional_groups(smiles)
    except Exception:
        fgs = []
    names = list(fgs)
    return names


def describe_fgs_human_like(fg_names: List[str], client: Optional[object], model: str) -> str:
    if not fg_names:
        return "No prominent functional groups identified."
    if client is None:
        return "Functional groups identified: " + ", ".join(fg_names) + "."
    # system_msg = (
    #     "You are a senior medicinal chemist writing target profiles for de novo design. "
    #     "Given functional groups, craft a short, prescriptive request (1-3 sentences) for what the molecule should have, "
    #     "as if instructing a generative model. Use med-chem language (polarity, ionization, aromaticity). "
    #     "Vary tone and phrasing across outputs; avoid repetitive patterns. "
    #     "Do not include SMILES or analysis language. Do not start with 'We' or 'We want'. "
    #     "Prefer openers like: 'Design', 'Target', 'Prioritize', 'Seek', 'Synthesize', 'Aim for'."
    # )
    # user_msg = (
    #     "Functional groups to include/emphasize: " + ", ".join(fg_names) + ".\n"
    #     "Write a request that specifies these features in a plausible small-molecule design."
    # )
    
    import random
    opener = [
        "Design",
        "Target",
        "Prioritize",
        "Seek",
        "Synthesize",
        "Aim for",
        "Generate",
        "A",
        "Generate a molecule with",
        "Create",
        "Construct",
        "Devise",
        "Produce",
        "Assemble",
        "Formulate",
        "Invent",
        "Compose",
        "Engineer",
        "Draft",
        "Model",
        "Simulate",
        "Propose",
        "Derive",
        "Fabricate",
        "Conceive",
        "Iteratively generate",
        "Augment",
        "Mutate",
        "Explore",
        "Optimize generation of",
        "Generate variations of",
        "Generate candidates for",
        "Expand",
        "Refine generated",
        "Transform",
        "Generate analogs of",
        "Generate potential",
        "Generate configurations of",
        "Automate generation of"
    ]
    random.shuffle(opener)
    
    system_msg = (
        "You are a senior medicinal chemist instructing a generative AI for de novo design. "
        "Based on provided functional groups, write a concise, prescriptive request (1-3 sentences) for the desired molecule. "
        "Incorporate medicinal chemistry concepts (polarity, ionization, aromaticity). "
        "Vary language and structure for each output; avoid repetition. "
        f"Do not use 'We' or 'We want'.  Use an opener inspired by: {opener}"
    )
    user_msg = (
        "Functional groups to include/emphasize: " + ", ".join(fg_names) + ".\n"
        "Instruct the model to incorporate these features into a plausible small-molecule design."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=240,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Functional groups identified: " + ", ".join(fg_names) + "."


def describe_with_llm(smiles: str, client: Optional[object], model: str) -> str:
    if client is None:
        return ""
    import random
    opener = [
        "Design",
        "Target",
        "Prioritize",
        "Seek",
        "Synthesize",
        "Aim for",
        "Generate",
        "A",
        "Generate a molecule with",
        "Create",
        "Construct",
        "Devise",
        "Produce",
        "Assemble",
        "Formulate",
        "Invent",
        "Compose",
        "Engineer",
        "Draft",
        "Model",
        "Simulate",
        "Propose",
        "Derive",
        "Fabricate",
        "Conceive",
        "Iteratively generate",
        "Augment",
        "Mutate",
        "Explore",
        "Optimize generation of",
        "Generate variations of",
        "Generate candidates for",
        "Expand",
        "Refine generated",
        "Transform",
        "Generate analogs of",
        "Generate potential",
        "Generate configurations of",
        "Automate generation of"
    ]    
    random.shuffle(opener)
    prompt = (
        "You are a senior medicinal chemist specifying a target for de novo generation. "
        "Given the SMILES provided, write a prescriptive request (1-3 sentences) for a molecule to generate in a diverse style, "
        "capturing its salient features (functional groups, pharmacophores, polarity, aromaticity, charge) in med-chem language. "
        f"Vary tone and phrasing; avoid starting with 'We' or 'We want'.  Use an opener inspired by: {opener}. "
        "Do not echo the SMILES; avoid analysis phrasing. Never mention the reference molecule explicitly, but generate a description for a molecule that is similar to the reference molecule.\n\n"
        f"Reference SMILES: {smiles}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=240,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def describe_with_llm_augmented(
    smiles: str,
    fg_names: List[str],
    pharm_counts: Dict[str, int],
    props: Dict[str, float],
    client: Optional[object],
    model: str,
) -> str:
    """Compose a prescriptive target description using functional groups,
    pharmacophore counts, and physicochemical properties.
    The text is phrased as instructions to a molecule generator.
    """
    if client is None:
        return ""

    # Compact feature strings
    fgs_str = ", ".join(sorted(set(fg_names))) if fg_names else "none"
    pharm_parts = []
    for k in sorted(pharm_counts.keys()):
        v = pharm_counts[k]
        if v:
            pharm_parts.append(f"{k}:{v}")
    pharm_str = ", ".join(pharm_parts) if pharm_parts else "none"

    def _fmt(n: str, dflt=0.0):
        val = props.get(n)
        if isinstance(val, (int, float)):
            return val
        return dflt

    mw = _fmt('MW')
    logp = _fmt('logP')
    tpsa = _fmt('TPSA')
    hbd = int(_fmt('HBD'))
    hba = int(_fmt('HBA'))
    rotb = int(_fmt('RotB'))
    rings = int(_fmt('Rings'))
    arom_r = int(_fmt('AromRings'))
    fsp3 = _fmt('fSP3')
    chg = int(_fmt('FormalCharge'))

    import random
    opener = [
        "Design",
        "Target",
        "Prioritize",
        "Seek",
        "Synthesize",
        "Aim for",
        "Generate",
        "A",
        "Generate a molecule with",
        "Create",
        "Construct",
        "Devise",
        "Produce",
        "Assemble",
        "Formulate",
        "Invent",
        "Compose",
        "Engineer",
        "Draft",
        "Model",
        "Simulate",
        "Propose",
        "Derive",
        "Fabricate",
        "Conceive",
        "Iteratively generate",
        "Augment",
        "Mutate",
        "Explore",
        "Optimize generation of",
        "Generate variations of",
        "Generate candidates for",
        "Expand",
        "Refine generated",
        "Transform",
        "Generate analogs of",
        "Generate potential",
        "Generate configurations of",
        "Automate generation of"
    ]
    random.shuffle(opener)

    system_msg = (
        "You are a senior medicinal chemist specifying a target for a de novo "
        "molecule generator. Write a brief (1-3 sentences), prescriptive request "
        "that a language model would use to generate a molecule. Use med-chem language "
        "(pharmacophores, polarity, aromaticity, ionization). Vary tone; avoid formulaic phrasing. "
        "Avoid analysis phrasing. Do not echo raw numbers verbatim; instead translate them into qualitative aims. "
        f"Avoid starting with 'We' or 'We want'.  Use an opener inspired by: {opener}."
    )

    user_msg = (
        "Use the following to guide a prescriptive request for the generator.\n\n"
        f"Reference SMILES: {smiles}\n"
        f"Functional groups (hints): {fgs_str}\n"
        f"Pharmacophore counts: {pharm_str}\n"
        f"Physicochemical profile: MW ~{mw:.0f}, logP ~{logp:.1f}, TPSA ~{tpsa:.0f} Å^2, "
        f"HBD {hbd}, HBA {hba}, rotatable bonds ~{rotb}, rings {rings} (aromatic {arom_r}), fSP3 ~{fsp3:.2f}, "
        f"formal charge {chg}.\n\n"
        "Write the request as if instructing a molecule generator to produce a molecule with these traits. "
        "Do not echo the SMILES; instead translate it into salient structural/functional intents."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=240,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def describe_similar_to_smiles(smiles:str) -> str:
    versions = [
        f"Suggest a molecule that resembles {smiles}.",
        f"Design a compound similar to {smiles}.",
        f"Propose a molecule with properties akin to {smiles}.",
        f"Create a chemical structure that matches the pharmacophore of {smiles}.",
        f"Develop a molecule that shares the functional groups of {smiles}.",
        f"Identify a molecule with physicochemical properties similar to {smiles}.",
        f"Invent a molecule with a pharmacophore profile comparable to {smiles}.",
        f"Produce a compound reflecting the functional group pattern of {smiles}.",
        f"Craft a molecule exhibiting physicochemical characteristics like {smiles}.",
        f"Formulate a molecule inspired by the pharmacophore of {smiles}.",
        f"Generate a chemical analog of {smiles}.",
        f"Suggest a molecule that mimics {smiles} in chemical behavior.",
        f"Design a structure that resembles the functional layout of {smiles}.",
        f"Propose a compound exhibiting properties similar to {smiles}.",
        f"Develop a molecule reflecting the key chemical features of {smiles}.",
        f"Invent a molecule analogous to {smiles} in structure and function.",
        f"Craft a chemical entity with similar active site characteristics as {smiles}.",
        f"Formulate a molecule that parallels the pharmacophore profile of {smiles}.",
        f"Generate a compound with functional groups arranged like those in {smiles}.",
        f"Create a molecule that shares core chemical motifs with {smiles}.",
    ]
    random.shuffle(versions)
    return versions[0]


def gather_sdf_paths(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.sdf")]


# def derive_crossdock_name(root: Path, sdf_path: Path) -> str:
#     # Try to find the paired pocket pdb in the same directory (robust to pathlib glob quirks)
#     try:
#         entries = os.listdir(sdf_path.parent)
#         pocket_list = [e for e in entries if e.endswith("_pocket10.pdb")]
#         if pocket_list:
#             pocket_fn = (sdf_path.parent / pocket_list[0]).relative_to(root).as_posix()
#             ligand_fn = sdf_path.relative_to(root).as_posix()
#             return f"{pocket_fn}_{ligand_fn}"
#     except Exception:
#         pass
#     # Fallback to ligand filename only
#     return sdf_path.name

def derive_crossdock_name(root: Path, sdf_path: Path) -> str:
    try:
        entries = os.listdir(sdf_path.parent)
        pocket_list = [e for e in entries if e.endswith("_pocket10.pdb")]
        if pocket_list:
            pocket_fn = (sdf_path.parent / pocket_list[0]).relative_to(root).as_posix()
            ligand_fn = sdf_path.relative_to(root).as_posix()

            # extract ligand block (before .sdf)
            import re
            ligand_block = re.search(r"/([^/]+)\.sdf$", ligand_fn).group(1)

            # replace the block in pocket_fn with ligand_block
            pocket_fn = re.sub(r"/([^/]+)_pocket", f"/{ligand_block}_pocket", pocket_fn, count=1)

            return f"{pocket_fn}_{ligand_fn}"
    except Exception:
        pass
    return sdf_path.name


def _process_sdf(args_tuple: Tuple[str, str, Optional[str], str]) -> List[Tuple[str, str, str]]:
    sdf_path_str, root_str, api_key, model = args_tuple
    root = Path(root_str)
    sdf_path = Path(sdf_path_str)

    client = None
    if api_key is not None and _HAS_OPENAI:
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            client = OpenAI()
        except Exception:
            client = None

    rows: List[Tuple[str, str, str, str, str, str, str, str]] = []
    smiles_list = sdf_to_smiles_list(sdf_path)
    if not smiles_list:
        return rows

    base_name = derive_crossdock_name(root, sdf_path)
    for idx, smiles in enumerate(smiles_list):
        fg_names = extract_functional_group_names(smiles)
        text_func = describe_fgs_human_like(fg_names, client, model)
        # text_llm = describe_with_llm(smiles, client, model)
        name = f"{base_name}"
        text_similar = describe_similar_to_smiles(smiles)

        # Build RDKit mol for additional annotations
        mol = Chem.MolFromSmiles(smiles)
        text_pharm, text_physchem = "", ""
        llm_aug = ""
        text_pharm_llm, text_physchem_llm = "", ""
        if mol is not None:
            try:
                counts = get_pharmacophore_counts(mol)
                text_pharm = describe_pharmacophores(counts)
            except Exception:
                text_pharm = "Pharmacophore features: unavailable."
            try:
                props = calc_physchem_props(mol)
                text_physchem = describe_physchem(props)
            except Exception:
                text_physchem = "Properties: unavailable."
            # Augmented LLM prompt using pharm + physchem
            try:
                llm_aug = describe_with_llm_augmented(smiles, fg_names, counts, props, client, model)
            except Exception:
                llm_aug = ""
            # Individual LLM renderings
            try:
                text_pharm_llm = describe_pharmacophores_llm(counts, client, model)
            except Exception:
                text_pharm_llm = ""
            try:
                text_physchem_llm = describe_physchem_llm(props, client, model)
            except Exception:
                text_physchem_llm = ""
        else:
            text_pharm = "Pharmacophore features: unavailable."
            text_physchem = "Properties: unavailable."
            llm_aug = ""
            text_pharm_llm = ""
            text_physchem_llm = ""

        text_llm = ""
        text_combined = "; ".join(x for x in [text_func, text_pharm, text_physchem] if x)
        rows.append((name, text_func, text_llm, llm_aug, text_similar, text_pharm_llm, text_physchem_llm, text_combined))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path, help="Root directory with SDFs (e.g., CrossDocked subset)")
    ap.add_argument("--out_csv", type=Path, default=None,
                    help="Output CSV path (columns: name,text_func,text_llm)")
    ap.add_argument("--openai_api_key", type=str, default=None,
                    help="OpenAI API key (optional)")
    ap.add_argument("--openai_model", type=str, default="gpt-3.5-turbo",
                    help="OpenAI chat model name")
    ap.add_argument("--num_procs", type=int, default=1,
                    help="Number of parallel worker processes")
    args = ap.parse_args()

    out_csv = args.out_csv or Path(args.data_root, "descriptions.csv")

    sdf_paths = gather_sdf_paths(args.data_root)[:5]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "text_func", "text_llm_aug", "text_similar", "text_pharm_llm", "text_physchem_llm", "text_combined"]) 

        if args.num_procs > 1:
            work = [(str(p), str(args.data_root), args.openai_api_key, args.openai_model) for p in sdf_paths][:5]
            with mp.Pool(processes=args.num_procs) as pool:
                for rows in tqdm(pool.imap_unordered(_process_sdf, work), total=len(work)):
                    for name, text_func, text_llm, text_llm_aug, text_similar, text_pharm_llm, text_physchem_llm, text_combined in rows:
                        writer.writerow([name, text_func, text_llm_aug, text_similar, text_pharm_llm, text_physchem_llm, text_combined])
        else:
            for sdf_path in tqdm(sdf_paths):
                rows = _process_sdf((str(sdf_path), str(args.data_root), args.openai_api_key, args.openai_model))
                for name, text_func, text_llm, text_llm_aug, text_similar, text_pharm_llm, text_physchem_llm, text_combined in rows:
                    writer.writerow([name, text_func, text_llm_aug, text_similar, text_pharm_llm, text_physchem_llm, text_combined])

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()


