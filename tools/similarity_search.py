import subprocess
import os
import requests
from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import argparse
try:
    import pubchempy as pcp
except Exception:
    pcp = None


def pubchem_similarity_search(smiles: str, num_results: int = 10, threshold: int = 85):
    """
    Perform a similarity search in PubChem using a SMILES string.

    Args:
        smiles (str): The SMILES string of the query molecule.
        num_results (int): The maximum number of similar compounds to return.
        threshold (int): Similarity threshold (0-100).

    Returns:
        list: A list of PubChem CIDs for similar compounds.
    """
    # Canonicalize and URL-encode SMILES for reliable API calls
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
    except Exception:
        pass
    enc = quote(smiles, safe="")

    def _call(thr: int):
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/"
            f"smiles/{enc}/cids/JSON?Threshold={thr}&MaxRecords={num_results}"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("IdentifierList", {}).get("CID", [])
        return []

    cids = _call(threshold)
    if not cids and threshold > 70:
        # Try a slightly lower threshold if nothing found
        cids = _call(70)
    return cids

def cids_to_inchi(cids: list) -> list:
    """
    Convert a list of PubChem CIDs to InChI strings.

    Args:
        cids (list): List of PubChem CIDs.
    Returns:
        list: List of InChI strings corresponding to the CIDs.
    """
    inchi_list = []
    if pcp is None:
        return [None for _ in cids]
    for cid in cids:
        try:
            compound = pcp.Compound.from_cid(cid)
            inchi_list.append(getattr(compound, "inchi", None))
        except Exception:
            inchi_list.append(None)
    return inchi_list


if __name__ == "__main__":
    #  get a simles find similar molecules in pubchem, resolve cid returend using pubchempy. commandline interface
    parser = argparse.ArgumentParser(description="PubChem similarity search")
    parser.add_argument("smiles", type=str, help="Input SMILES string")
    parser.add_argument("--num_results", type=int, default=10, help="Number of similar compounds to return (default: 10)")
    args = parser.parse_args()
    HYPERLINK_BASE = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    cids = pubchem_similarity_search(args.smiles, num_results=args.num_results)
    inchi_list = cids_to_inchi(cids)
    print(cids)
    for inchi, cid in zip(inchi_list, cids):
        print(inchi, HYPERLINK_BASE + str(cid))

    #  example usage
    # python tools/tools.py "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" --threshold 85