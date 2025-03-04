import os
import sys

from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import pymol2
import scipy
from scipy.spatial.transform import Rotation
import shutil
import time
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.SimpleUnet import SimpleHalfUnetModel
from learning.predict_coords import predict_coords
from prepare_database.get_templates import REF_PATH_FV, REF_PATH_NANO, REF_PATH_FAB


def mwe():
    """Just sanity check for pymol procedures"""
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        pdb_em_path = "../data/pdb_em/"
        pdb = '7WP6'
        mrc = '32676'
        pdb_dir_path = os.path.join(pdb_em_path, f'{pdb}_{mrc}')
        pdb_path = os.path.join(pdb_dir_path, f'{pdb}.cif')
        p.cmd.load(pdb_path, 'in_pdb')
        selections = ['chain H or chain L', 'chain D or chain C', 'chain B or chain I']
        for i, selection in enumerate(selections):
            sel = f'in_pdb and ({selection})'
            p.cmd.extract(f"to_align", sel)
            coords = p.cmd.get_coords("to_align")
            print(pdb, selection)
            print(coords.shape)


def get_pdbsels(csv_in='../data/csvs/sorted_filtered_test.csv',
                out_name=None):
    """
    Get target PDBs within a csv test set
    """
    # group mrc by pdb
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    pdb_selections = defaultdict(list)
    df = df[['pdb', 'mrc', 'resolution', 'antibody_selection']]
    for i, row in df.iterrows():
        pdb, mrc, resolution, selection = row.values
        pdb_selections[(pdb.upper(), mrc, resolution)].append(selection)
    # Remove outlier systems with over 10 abs (actually just one)
    pdb_selections = {key: val for key, val in pdb_selections.items() if len(val) < 10}
    if out_name is not None:
        pickle.dump(pdb_selections, open(out_name, 'wb'))
    return pdb_selections


def get_systems(csv_in='../data/csvs/sorted_filtered_test.csv',
                pdb_em_path="../data/pdb_em/",
                test_path="../data/testset/",
                nano=False):
    """
    The goal is to organize the test set in a clean repo with only necessary files:
    This results in gt_fv_{i}.pdb and gt_nano_{i}.pdb files with extractions
    """
    os.makedirs(test_path, exist_ok=True)
    out_name_pdbsels = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    pdb_selections = get_pdbsels(csv_in=csv_in, out_name=out_name_pdbsels)
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(REF_PATH_FV, "ref_fv")
        p.cmd.load(REF_PATH_NANO, "ref_nano")
        p.cmd.load(REF_PATH_FAB, 'ref_fab')
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            # if pdb != "7YVN":
            #     continue
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)}")
            pdb_dir_path = os.path.join(pdb_em_path, f'{pdb}_{mrc}')
            pdb_path = os.path.join(pdb_dir_path, f'{pdb}.cif')
            em_path = os.path.join(pdb_dir_path, "full_crop_resampled_2.mrc")

            # Make new dir
            new_dir_path = os.path.join(test_path, f'{pdb}_{mrc}')
            os.makedirs(new_dir_path, exist_ok=True)

            # Copy mrc file
            new_em_path = os.path.join(new_dir_path, "full_crop_resampled_2.mrc")
            new_pdb_path = os.path.join(new_dir_path, f"{pdb}.cif")
            shutil.copy(em_path, new_em_path)
            shutil.copy(pdb_path, new_pdb_path)

            # Get gt and rotated pdbs
            p.cmd.load(new_pdb_path, 'in_pdb')
            all_rmsd = []
            for i, selection in enumerate(selections):
                outpath_gt = os.path.join(new_dir_path, f'gt_{"nano_" if nano else ""}{i}.pdb')
                sel = f'in_pdb and ({selection})'
                p.cmd.extract(f"to_align", sel)
                p.cmd.save(outpath_gt, "to_align")
                coords = p.cmd.get_coords("to_align")

                # To get COM consistence, we need to save the Fv part only
                if not nano:
                    residues_to_align = len(p.cmd.get_model("to_align").get_residues())
                    fab = residues_to_align > 300
                    # For fabs, first align the whole fab and then the Fv to its Fab,
                    # this drastically reduces the rmsd
                    if fab:
                        rmsd1 = p.cmd.align(mobile="ref_fab", target="to_align")[0]
                        rmsd2 = p.cmd.align(mobile="ref_fv", target="ref_fab")[0]
                        rmsd = rmsd1 + rmsd2
                        # if rmsd > 3:
                        #     print(pdb, rmsd1, rmsd2, fab, residues_to_align)
                    else:
                        rmsd = p.cmd.align(mobile="ref_fv", target="to_align")[0]
                        # if rmsd > 3:
                        #     print(pdb, rmsd, fab, residues_to_align)
                    outpath_gt_fv = os.path.join(new_dir_path, f'gt_fv_{i}.pdb')
                    p.cmd.save(outpath_gt_fv, "ref_fv")

                # To get COM consistence, we need to save the nano part only (for edge cases like megabodies)
                else:
                    rmsd = p.cmd.align(mobile="ref_nano", target="to_align")[0]
                    outpath_gt_nano = os.path.join(new_dir_path, f'gt_nano_{i}.pdb')
                    p.cmd.save(outpath_gt_nano, "ref_nano")
                p.cmd.delete("to_align")
                all_rmsd.append(rmsd)
            p.cmd.delete("in_pdb")


def make_predictions(nano=False, gpu=0, test_path="../data/testset/", use_mixed_model=True, sorted_split=True,
                     use_pd=True, suffix='', model_path=None, use_uy=False):
    """
    Now let's make predictions for this test set with ns_final model.
    :param nano:
    :param gpu:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))

    model = SimpleHalfUnetModel(classif_nano=use_mixed_model, num_feature_map=32)
    if model_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if use_mixed_model or nano:
            if sorted_split:
                model_path = os.path.join(script_dir, '../saved_models/ns_final_last.pth')
            else:
                model_path = os.path.join(script_dir, '../saved_models/nr_final_last.pth')
        else:
            if sorted_split:
                model_path = os.path.join(script_dir, '../saved_models/fs_final_last.pth')
            else:
                model_path = os.path.join(script_dir, '../saved_models/fr_final_last.pth')
    model.load_state_dict(torch.load(model_path))

    time_init = time.time()
    pred_number = {}
    with torch.no_grad():
        for step, (pdb, mrc, resolution) in enumerate(pdb_selections.keys()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")
            pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
            in_mrc = os.path.join(pdb_dir, "full_crop_resampled_2.mrc")
            out_name = os.path.join(pdb_dir, f'crai_pred{suffix}{"_nano" if nano else ""}.pdb')
            predict_coords(mrc_path=in_mrc, outname=out_name, model=model, device=gpu, split_pred=True,
                           n_objects=10, thresh=0.2, classif_nano=use_mixed_model, use_pd=use_pd, verbose=False,
                           use_uy=use_uy)

            # TO GET num_pred
            transforms = predict_coords(mrc_path=in_mrc, outname=None, model=model, device=gpu, split_pred=True,
                                        n_objects=None, thresh=0.2, classif_nano=use_mixed_model, use_pd=use_pd,
                                        verbose=False, use_uy=use_uy)
            pred_number[pdb] = len(transforms)
    pickle_name = f'num_pred{suffix}{"_nano" if nano else ""}.p'
    pickle.dump(pred_number, open(os.path.join(test_path, pickle_name), 'wb'))


def compute_matching_hungarian(actual_pos, pred_pos, thresh=10):
    dist_matrix = scipy.spatial.distance.cdist(pred_pos, actual_pos)
    # print(actual_pos, pred_pos)
    gt_found = []
    for i in range(1, len(pred_pos) + 1):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix[:i])
        position_dists = dist_matrix[row_ind, col_ind]
        found = np.sum(position_dists < thresh)
        gt_found.append(found)
    # print(gt_found)
    # print(position_dists)
    # print()
    return gt_found


def get_hit_rates(nano=False, test_path="../data/testset/", suffix='', fitmap=False):
    """
    Go over the predictions and computes the hit rates with each number of systems.
    :param nano:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))

    time_init = time.time()
    all_res = {}
    with pymol2.PyMOL() as p:
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")

            pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')

            # First get the list of GT com
            gt_com = []
            for i in range(len(selections)):
                # We use the Fv GT in the case of Fabs
                gt_name = os.path.join(pdb_dir, f'gt_{"nano_" if nano else "fv_"}{i}.pdb')
                p.cmd.load(gt_name, 'gt')
                gt_coords = p.cmd.get_coords('gt')
                com = np.mean(gt_coords, axis=0)
                gt_com.append(com)
                p.cmd.delete('gt')
            max_com = np.max(np.stack(gt_com), axis=0)
            default_coords = tuple(max_com + 1000)

            # Now get the (sorted) list of predicted com
            predicted_com = []
            for i in range(10):
                # for i in range(len(selections)):
                file_name = f'{"fitmap_" if fitmap else ""}crai_pred{suffix}{"_nano" if nano else ""}_{i}.pdb'
                out_name = os.path.join(pdb_dir, file_name)
                if not os.path.exists(out_name):
                    # Not sure why but sometimes fail to produce 10 systems.
                    # Still gets 5-6 for small systems. Maybe the grid is too small.
                    # print(out_name)
                    predicted_com.append(default_coords)
                    continue
                p.cmd.load(out_name, 'crai_pred')
                predictions = p.cmd.get_coords(f'crai_pred')
                com = np.mean(predictions, axis=0)
                predicted_com.append(com)
                p.cmd.delete('crai_pred')

            hits_thresh = compute_matching_hungarian(gt_com, predicted_com)
            gt_hits_thresh = list(range(1, len(gt_com) + 1)) + [len(gt_com)] * (len(predicted_com) - len(gt_com))
            all_res[pdb] = (gt_hits_thresh, hits_thresh, resolution)
    outname = os.path.join(test_path, f'{"fitmap_" if fitmap else ""}all_res{suffix}{"_nano" if nano else ""}.p')
    pickle.dump(all_res, open(outname, 'wb'))


def string_rep(sorted_split=None, nano=None, mixed=None, num=None, dockim=None, fitmap=None):
    s = ""
    if sorted_split is not None:
        s += 'Sorted ' if sorted_split else 'Random '
    if nano is not None:
        s += 'Nano ' if nano else 'Fab '
    if mixed is not None:
        s += 'Mixed ' if mixed else 'NonMixed '
    if num is not None:
        s += 'Num' if num else 'Thresh '
    if dockim is not None:
        s += 'Dockim ' if dockim else ''
    if fitmap is not None:
        s += 'Fitmap ' if fitmap else ''
    return s


def compute_all():
    """
    Get the tables results
    :return:
    """
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sorted_split else ""}filtered_test.csv'
            print('Getting data for ', string_rep(sorted_split=sorted_split, nano=nano))
            get_systems(csv_in=csv_in, nano=nano, test_path=test_path)
            # Now let us get the prediction in all cases
            print('Making predictions for :', string_rep(nano=nano, mixed=True))
            make_predictions(nano=nano, test_path=test_path, use_mixed_model=True, gpu=0,
                             sorted_split=sorted_split)
            get_hit_rates(nano=nano, test_path=test_path)

            # No models are dedicated to nano only, for Fabs, use the fab_only model
            if not nano:
                print('Making predictions for :', string_rep(nano=nano, mixed=False))
                make_predictions(nano=nano, test_path=test_path, use_mixed_model=False, gpu=0,
                                 sorted_split=sorted_split, suffix='_fab')
                get_hit_rates(nano=nano, test_path=test_path, suffix='_fab')


def compute_ablations():
    """
    Get ablation results on the random fab split:
    - no OT model
    - no PD
    - uy model
    :return:
    """
    test_path = f'../data/testset_random'
    # Get the no_ot predictions
    print('Making predictions for no ot')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, '../saved_models/fab_random_normalize_last.pth')
    make_predictions(nano=False, test_path=test_path, gpu=0, use_mixed_model=False, model_path=model_path,
                     suffix='_no_ot')

    # Get the no_PD predictions
    print('Making predictions for no pd')
    make_predictions(nano=False, test_path=test_path, gpu=0, sorted_split=False, use_pd=False, suffix='_no_pd')

    # Get the uy predictions
    print('Making predictions for uy')
    model_path = os.path.join(script_dir, '../saved_models/fr_uy_last.pth')
    make_predictions(nano=False, test_path=test_path, gpu=0, use_mixed_model=False, model_path=model_path, use_uy=True,
                     suffix='_uy')

    print('Getting hit rates')
    get_hit_rates(nano=False, test_path=test_path, suffix='_no_ot')
    get_hit_rates(nano=False, test_path=test_path, suffix='_no_pd')
    get_hit_rates(nano=False, test_path=test_path, suffix='_uy')


if __name__ == '__main__':
    pass
    # TODO : understand why n<10 sometimes
    # mwe()

    # To do one
    # sorted_split = True
    # nano = False
    # csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sorted_split else ""}filtered_test.csv'
    # test_path = f'../data/testset{"" if sorted_split else "_random"}'
    # get_systems(csv_in=csv_in, nano=nano, test_path=test_path)
    # make_predictions(sorted_split=sorted_split, nano=nano, test_path=test_path)
    # get_hit_rates(nano=True, test_path='../data/testset')

    # GET DATA
    compute_all()

    # GET ABLATIONS
    compute_ablations()
