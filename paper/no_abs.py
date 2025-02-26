import os

from collections import defaultdict
import pandas as pd
import pickle
import requests
import time


def get_antigens():
    """
    input: sorted test systems
    output: pdb_em with no abs in it
    """
    fab_csvs = "../data/csvs/sorted_filtered_test.csv"
    nano_csvs = "../data/nano_csvs/sorted_filtered_test.csv"
    fab_df = pd.read_csv(fab_csvs)
    nano_df = pd.read_csv(nano_csvs)

    def df_to_chains(df):
        pdb_to_chains = dict()
        for pdb in df['pdb'].unique():
            a = df[df['pdb'] == pdb]['antigen_chain'].values
            vals = list(set([y for x in a for y in x.split(" | ")]))
            pdb_to_chains[pdb] = vals
        return pdb_to_chains

    fab_dict = df_to_chains(fab_df)
    nano_dict = df_to_chains(nano_df)
    merged_dict = defaultdict(set)

    for d in (fab_dict, nano_dict):  # Iterate over both dictionaries
        for pdb, chains in d.items():
            merged_dict[pdb].update(chains)  # Add chains to the set (to avoid duplicates)

    # Convert sets back to lists
    merged_dict = {pdb: sorted(list(chains)) for pdb, chains in merged_dict.items()}
    return merged_dict


def get_uniprots_from_pdb(pdb_id):
    """
    Get UniProt accession(s) for a PDB entry
    Returns:
        dict: Mapping of chain IDs to UniProt accessions
    """
    sifts_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    sifts_response = requests.get(sifts_url)

    uniprot_mappings = defaultdict(list)
    if sifts_response.status_code == 200:
        sifts_data = sifts_response.json()
        for uniprot, res_dict in sifts_data[pdb_id.lower()]['UniProt'].items():
            for hit in res_dict['mappings']:
                chain_id_from_api = hit['chain_id']
                uniprot_mappings[chain_id_from_api].append(uniprot)
    return uniprot_mappings


def get_all_uniprots(recompute=False):
    outfile = "all_uniprots.p"
    if not os.path.exists(outfile) or recompute:
        antigens = get_antigens()
        all_uniprots = []
        for i, (pdb, chains) in enumerate(antigens.items()):
            try:
                chain_uniprot = get_uniprots_from_pdb(pdb)
                relevant_uniprots = {uniprots for chain in chains for uniprots in chain_uniprot[chain]}
                all_uniprots.extend(list(relevant_uniprots))
            except Exception as e:
                pass
            time.sleep(0.02)
        all_uniprots = list(set(all_uniprots))
        pickle.dump(all_uniprots, open(outfile, "wb"))
    all_uniprots = pickle.load(open(outfile, "rb"))
    return all_uniprots


def get_pdb_structures_from_uniprot(uniprot_id):
    """
    Get all PDB structures associated with a UniProt accession.

    Args:
        uniprot_id (str): UniProt accession

    Returns:
        list: List of PDB structures with metadata
    """
    try:
        uniprot_api_url = "https://rest.uniprot.org/uniprotkb/"

        # Get the UniProt entry data
        params = {'format': 'json', 'size': 1}
        response = requests.get(f"{uniprot_api_url}{uniprot_id}", params=params)
        response.raise_for_status()

        # Get cross-references to PDB
        pdb_structures = []
        data = response.json()
        for ref in data['uniProtKBCrossReferences']:
            if ref["database"] == 'PDB':
                pdb_id = ref['id'].lower()

                # Extract method and resolution if available
                for elt in ref.get('properties', []):
                    if elt['key'] == 'Method':
                        if elt['value'] == 'EM':
                            pdb_structures.append(pdb_id)
                        continue

        time.sleep(0.02)  # Be gentle with the API
        return pdb_structures

    except Exception as e:
        print(f"Error getting PDB structures for UniProt {uniprot_id}: {e}")
        return []


def get_all_pdbs(recompute=False):
    outfile = "all_pdbs.p"
    if not os.path.exists(outfile) or recompute:
        ags_uniprots = get_all_uniprots()
        all_structures = []
        for uniprot in ags_uniprots:
            structures = get_pdb_structures_from_uniprot(uniprot)
            structures = list(set(structures))
            all_structures.append((uniprot, structures))

        fab_csvs = "../data/csvs/fabs.tsv"
        nano_csvs = "../data/nano_csvs/nanobodies.tsv"
        fab_df = pd.read_csv(fab_csvs, sep='\t')
        nano_df = pd.read_csv(nano_csvs, sep='\t')
        pdb_in_ours = set(fab_df['pdb'].unique()).union(set(nano_df['pdb'].unique()))
        filtered_pdbs = [(uniprot, [pdb for pdb in pdbs if pdb not in pdb_in_ours])
                         for (uniprot, pdbs) in all_structures]
        selected_pdb_ids = [pdbs[0] for _, pdbs in filtered_pdbs if len(pdbs) > 0]
        pickle.dump(selected_pdb_ids, open(outfile, "wb"))
    selected_pdb_ids = pickle.load(open(outfile, "rb"))
    return selected_pdb_ids


if __name__ == "__main__":
    # ags = get_antigens()
    uniprots = get_all_uniprots(recompute=False)
    selected_pdbs = get_all_pdbs(recompute=False)
    # print(selected_pdbs)
