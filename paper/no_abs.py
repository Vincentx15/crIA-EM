import os
import sys

from collections import defaultdict
import pandas as pd
import pickle
import requests
import time
import torch
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.predict_coords import predict_coords
from learning.SimpleUnet import SimpleHalfUnetModel
from prepare_database.download_data import get_mapping_ids, download_one_mrc, download_one_cif


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
    outfile = "no_abs_uniprots.p"
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
    """
    try:
        # Get the UniProt entry data
        uniprot_api_url = "https://rest.uniprot.org/uniprotkb/"
        params = {'format': 'json', 'size': 1}
        response = requests.get(f"{uniprot_api_url}{uniprot_id}", params=params)
        response.raise_for_status()

        # Get cross-references to PDB that were obtained with EM
        pdb_structures = []
        data = response.json()
        for ref in data['uniProtKBCrossReferences']:
            if ref["database"] == 'PDB':
                pdb_id = ref['id'].lower()
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
    outfile = "no_abs_pdbs.txt"
    if not os.path.exists(outfile) or recompute:
        ags_uniprots = get_all_uniprots()

        # For each uniprot, get the corresponding PDBs
        all_structures = []
        for uniprot in ags_uniprots:
            structures = get_pdb_structures_from_uniprot(uniprot)
            structures = list(set(structures))
            all_structures.append((uniprot, structures))

        # Filter all such PDBs to not contain Abs
        fab_csvs = "../data/csvs/latest.tsv"
        nano_csvs = "../data/nano_csvs/latest.tsv"
        fab_df = pd.read_csv(fab_csvs, sep='\t')
        nano_df = pd.read_csv(nano_csvs, sep='\t')
        pdb_in_ours_latest = set(fab_df['pdb'].unique()).union(set(nano_df['pdb'].unique()))
        filtered_pdbs = [(uniprot, [pdb for pdb in pdbs if pdb not in pdb_in_ours_latest])
                         for (uniprot, pdbs) in all_structures]
        selected_pdb_ids = [pdbs[0] for _, pdbs in filtered_pdbs if len(pdbs) > 0]

        # Save results to file
        with open(outfile, "w") as f:
            for item in selected_pdb_ids:
                f.write(item + "\n")
    with open(outfile, "r") as f:
        selected_pdb_ids = [line.strip() for line in f]
    return selected_pdb_ids


def build_pdb_em(selected_pdbs, out_dir="../data/noabs_pdb_em/"):
    os.makedirs(out_dir, exist_ok=True)
    ems = get_mapping_ids(selected_pdbs)
    for i, (pdb, em) in enumerate(ems.items()):
        print(f"Done {pdb, em} {i}/{len(ems)}")
        em_id = em.split('-')[1]
        dirname = f"{pdb}_{em_id}"
        os.makedirs(os.path.join(out_dir, dirname), exist_ok=True)
        download_one_cif(pdb_id=pdb, outdir=os.path.join(out_dir, dirname))
        download_one_mrc(emd_id=em_id, outdir=os.path.join(out_dir, dirname))


def predict_all(selected, pdb_em_dir="../data/noabs_pdb_em/"):
    # Get model
    classif_nano = True
    model_path = os.path.join(script_dir, '../saved_models/ns_final_last.pth')
    model = SimpleHalfUnetModel(classif_nano=classif_nano, num_feature_map=32)
    model.load_state_dict(torch.load(model_path))

    todo = [pdb_em for pdb_em in os.listdir(pdb_em_dir) if pdb_em[:4].lower() in selected]
    volumes = []
    results = {}
    # for pdb_em in tqdm(todo):
    for pdb_em in todo:
        system_dir = os.path.join(pdb_em_dir, pdb_em)
        em_id = pdb_em.split('_')[1]

        import xmltodict
        import numpy as np
        # filter out viruses (huge systems)
        in_xml = os.path.join(system_dir, f"emd-{em_id}.xml")
        with open(in_xml, "r", encoding="utf-8") as file:
            xml_content = file.read()
            data_dict = xmltodict.parse(xml_content)
        grid_count = [int(x) for x in data_dict['emd']['map']['dimensions'].values()]
        cell_size = [float(x['#text']) for x in data_dict['emd']['map']['pixel_spacing'].values()]
        volume = np.prod(np.array(grid_count + cell_size))
        volumes.append((volume, pdb_em))
        if volume > 100000000:
            continue

        in_mrc = os.path.join(system_dir, f"emd_{em_id}.map.gz")
        output = os.path.join(system_dir, f"predicted.pdb")

        try:
            predictions = predict_coords(mrc_path=in_mrc, outname=output, model=model, thresh=0.2,
                                         classif_nano=classif_nano, use_pd=True)
        except:
            predictions = [i for i in range(100)]
        results[pdb_em] = len(predictions)

    for vol, pdb in list(sorted(volumes)):
        print(pdb, vol)

    total_preds = sum(results.values())
    total_preds_no_failures = sum([i for i in results.values() if i < 30])
    print(results)
    print("Total false positive predictions:", total_preds)
    print("Total false positive predictions no failures:", total_preds_no_failures)
    pickle.dump(results, open("no_abs_predicted.p", "wb"))


# Actual systems (6): 8C03_16354, 9F33_50168, 8JRU_36606, 6DZY_8942, 8SAT_40279, 7L89_23226
# Maybe actual ? (2): 8HK5_34846, 8U4N_41888
# Error + disappears (2): 8F7S_28909, 7RU2_24694
# Error + stays (2): 7ANW_11834, 8IWP_35776
# 7Y1Q_33570   7622111.231999999
# 8HK5_34846   8741816.0               Actual antibody ? stays
# 8U4N_41888   9103145.472             Actual antibody ? disappear
# 7SYF_25524   9250398.87962905
# 8F7S_28909   9943923.032063998       1 extra in artifacts, disappear
# 7YDQ_33756  10021812.416000001
# 7TX6_26155  10077696.0
# 8U8F_42023  10671248.900096
# 8C03_16354  12872131.505856          Synthetic nano missing in Sabdab-nano
# 8DMI_27539  13413413.376
# 8DGS_27428  16581375.0
# 7PVD_13667  18763418.123108353
# 9F33_50168  19421724.                Actual Fv missing in sabdab
# 7Y1R_33571  19700501.184511997
# 3J0G_5275   20552811.020288
# 8JRU_36606  20610489.595723774       Actual fabs/nano missing in sabdab
# 8DNG_27566  22425768.0
# 7ANW_11834  24250498.008624997       Two extra, stays
# 8IWP_35776  27653197.823999997       Extra nano in helical, stays
# 6DZY_8942   30723115.968000002       Actual fabs missing in SabDab
# 7ZJ6_14742  34012224.00000001
# 8D74_27227  39304000.0
# 8SAT_40279  39693696.892928004       Actual fabs missing in sabdab
# 7L89_23226  50982270.912 non-virus   Actual fabs missing in sabdab
# 7PHR_13427  65548320.76800001 non-virus
# 7RU2_24694  98252976.55398402 non-virus  Extra nano in artifacts, disappears
# 5IZ7_8139  153990656.0 virus
# 3J2W_5577  186064687.68537605 virus
# 3EPD_1562  188014343.31187204 virus
# 3EPC_1570  190158973.06262496 virus
# 3EPF_1563  192319849.93104804 virus
# 8VSD_43494 217288184.38074684 non-virus
# 8DEC_27389 505880543.23200005 virus

if __name__ == "__main__":
    # ags = get_antigens()
    # uniprots = get_all_uniprots(recompute=False)
    selected_pdbs = get_all_pdbs(recompute=False)

    build_pdb_em(selected_pdbs)

    predict_all(selected_pdbs)
    results = pickle.load(open("no_abs_predicted.p", "rb"))
    sys_res = [x > 0 for x in results.values()]
    print(sys_res)
    print(sum(sys_res))
    print(len(sys_res))
