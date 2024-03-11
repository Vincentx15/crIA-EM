import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy
import time

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import string_rep
from utils.object_detection import pdbsel_to_transforms


def match_transforms_to_angles_dist(gt_transforms, pred_transforms):
    # TODO : check RMSD
    gt_translations = [x[1] for x in gt_transforms]
    gt_rotations = [x[2] for x in gt_transforms]
    gt_rz = [rotation.as_matrix()[:, 2] for rotation in gt_rotations]

    pred_translations = [x[1] for x in pred_transforms]
    pred_rotations = [x[2] for x in pred_transforms]
    pred_rz = [rotation.as_matrix()[:, 2] for rotation in pred_rotations]

    dist_matrix = scipy.spatial.distance.cdist(pred_translations, gt_translations)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
    position_dists = dist_matrix[row_ind, col_ind]

    def get_angle(u1, u2):
        # just return the angle between vectors
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        return np.arccos(np.dot(u1, u2)) * 180 / 3.14

    angles = [get_angle(pred_rz[row], gt_rz[col]) for row, col in zip(row_ind, col_ind)]
    return position_dists, angles


def get_angles_dist_dockim(nano=False, test_path="../data/testset/"):
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    outname_mapping = os.path.join(test_path, f'dockim_chain_map{"_nano" if nano else ""}.p')
    all_pdb_chain_mapping = pickle.load(open(outname_mapping, 'rb'))
    all_res = {}
    time_init = time.time()
    for step, ((pdb, mrc, resolution), selections) in enumerate(sorted(pdb_selections.items())):
        if not step % 20:
            print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")
        pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
        # First get GT transforms
        gt_name = os.path.join(pdb_dir, f'{pdb}.cif')
        try:
            gt_transforms = pdbsel_to_transforms(gt_name, antibody_selections=selections, cache=False)

            # The dockim chains are not sorted (because we compute copies first) so 1_0 can be H,I and not C,D.
            n_to_names = {}
            for n_pred, (i, k) in enumerate(sorted(all_pdb_chain_mapping[pdb].keys(), key=lambda x: x[1])):
                names_map = all_pdb_chain_mapping[pdb][(i, k)]
                # print(n_pred, (i, k), names_map)
                n_to_names[n_pred] = names_map[1]
            # Just get the num preds
            num_pred = len(selections)
            if nano:
                pymol_chain_sels = [f"chain {n_to_names[j]}" for j in range(num_pred)]
            else:
                pymol_chain_sels = [f"chain {n_to_names[j][0]} or chain {n_to_names[j][1]}"
                                    for j in range(num_pred)]
            out_name = os.path.join(pdb_dir, f'dockim_pred{"_nano" if nano else ""}_{num_pred - 1}.pdb')
            if not os.path.exists(out_name):
                all_res[pdb] = None
                continue
            pred_transforms = pdbsel_to_transforms(out_name, antibody_selections=pymol_chain_sels, cache=False)
            dists, angles = match_transforms_to_angles_dist(gt_transforms, pred_transforms)
            all_res[pdb] = dists, angles
        except Exception as e:
            print('failed on pdb : ', pdb)
            all_res[pdb] = None
    outname_results = os.path.join(test_path, f'dockim_angledist{"_nano" if nano else ""}.p')
    pickle.dump(all_res, open(outname_results, 'wb'))


def get_angles_dist(nano=False, test_path="../data/testset/", num_setting=False, suffix='', recompute=False):
    """
    Go over the predictions and computes the angles and distances for each system.
    :param nano:
    :param test_path:
    :return:
    """
    outfilename_results = f'angledist{suffix}{"_nano" if nano else ""}{"_num" if num_setting else "_thresh"}.p'
    outname_results = os.path.join(test_path, outfilename_results)
    if not recompute and os.path.exists(outname_results):
        return

    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    time_init = time.time()
    all_res = {}
    if not num_setting:
        num_pred_path = os.path.join(test_path, f'num_pred{suffix}{"_nano" if nano else ""}.p')
        num_pred_all = pickle.load(open(num_pred_path, 'rb'))

    for step, ((pdb, mrc, resolution), selections) in enumerate(sorted(pdb_selections.items())):
        if not step % 20:
            print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")
        pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')

        gt_name = os.path.join(pdb_dir, f'{pdb}.cif')
        gt_transforms = pdbsel_to_transforms(gt_name, antibody_selections=selections, cache=False)

        if num_setting:
            num_pred = len(selections)
        else:
            num_pred = min(num_pred_all[pdb], 9)

        # Now get the (sorted) list of predicted com
        pred_transforms = []
        for i in range(num_pred):
            pred_name = os.path.join(pdb_dir, f'crai_pred{suffix}{"_nano" if nano else ""}_{i}.pdb')
            if not os.path.exists(pred_name):
                continue
            dummy_sel = "polymer.protein and polymer.protein"
            i_th_transform = pdbsel_to_transforms(pred_name, antibody_selections=dummy_sel, cache=False)
            pred_transforms.extend(i_th_transform)
        if len(pred_transforms) == 0:
            all_res[pdb] = None
            continue
        dists, angles = match_transforms_to_angles_dist(gt_transforms, pred_transforms)
        all_res[pdb] = dists, angles
    pickle.dump(all_res, open(outname_results, 'wb'))


def get_angledist_results():
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            # Now let us get the prediction in all cases
            print('Doing ', string_rep(sorted_split=sorted_split, nano=nano))
            get_angles_dist_dockim(nano=nano, test_path=test_path)
            for num_setting in [True, False]:
                get_angles_dist(nano=nano, test_path=test_path, num_setting=num_setting)


def get_distance_one(test_path=None, dockim=False, nano=False, num_setting=False, suffix='', thresh=10):
    if dockim:
        pickle_name_to_get = f'dockim_angledist{"_nano" if nano else ""}.p'
    else:
        pickle_name_to_get = f'angledist{suffix}{"_nano" if nano else ""}{"_num" if num_setting else "_thresh"}.p'
    outname_results = os.path.join(test_path, pickle_name_to_get)
    results = pickle.load(open(outname_results, 'rb'))
    sel_dists = list()
    for pdb, pdb_res in results.items():
        if pdb_res is None:
            continue
        dists, angles = pdb_res
        selector = dists < thresh
        sel_dists.extend(list(dists[selector]))
    print(f'{np.mean(sel_dists):.2f}')


def get_distances_all():
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            print('Doing ', string_rep(sorted_split=sorted_split, nano=nano))
            get_distance_one(nano=nano, test_path=test_path, dockim=True, num_setting=True)
            for num_setting in [True, False]:
                get_distance_one(nano=nano, test_path=test_path, num_setting=num_setting)


def get_angles_one(test_path=None, dockim=False, nano=False, num_setting=False, suffix='', thresh=10):
    if dockim:
        pickle_name_to_get = f'dockim_angledist{"_nano" if nano else ""}.p'
    else:
        pickle_name_to_get = f'angledist{suffix}{"_nano" if nano else ""}{"_num" if num_setting else "_thresh"}.p'
    outname_results = os.path.join(test_path, pickle_name_to_get)
    results = pickle.load(open(outname_results, 'rb'))
    sel_angles = list()
    for pdb, pdb_res in results.items():
        if pdb_res is None:
            continue
        dists, angles = pdb_res
        selector = dists < thresh
        sel_angles.extend(list(np.asarray(angles)[selector]))
    return sel_angles


def plot_dict_hist(angle_resdict):
    """
    Expected to receive keys as legend and values as lists of values
    :return:
    """
    # Rest of the plot decorations
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)

    all_angles = []
    all_names = []
    for label, angles in angle_resdict.items():
        all_angles.extend(angles)
        all_names.extend([label for _ in range(len(angles))])
    all_angles = [angle + 1 for angle in all_angles]
    # all_angles = [int(angle) for angle in all_angles]
    # print(all_angles)
    df = pd.DataFrame({'angles': all_angles,
                       'Model Name': all_names})

    use_log = True
    if use_log:
        bins = np.logspace(1, np.log(180) / np.log(10), 10)
    else:
        bins = np.linspace(0, 180, 10)

    # print(bins)
    hist_kwargs = {
        "multiple": "dodge",
        "stat": "percent",
        "kde": False,
        "bins": bins,
        "common_norm": False,
        # "log_scale": (True, True), # does not work with bins
    }
    sns.histplot(data=df, x='angles', hue='Model Name', shrink=.9, **hist_kwargs)
    if use_log:
        ax = plt.gca()
        ax.set_xticks = bins
        ax.xticks = bins
        ax.set(xscale='log')
        ax.set(yscale='log')
    # sns.histplot(all_angles[0], label=labels[0], **hist_kwargs)
    # sns.histplot(all_angles[1], label=labels[1], **hist_kwargs)
    # sns.histplot(all_angles[2], label=labels[2], **hist_kwargs)
    # plt.hist(all_angles[0], label=labels[0], **hist_kwargs)
    # plt.hist(all_angles[1], label=labels[1], **hist_kwargs)
    # plt.hist(all_angles[2], label=labels[2], **hist_kwargs)
    # bw_method = 1
    # sns.kdeplot(all_angles[0], label=labels[0], clip=(0, 180), bw_method=bw_method)
    # sns.kdeplot(all_angles[1], label=labels[1], clip=(0, 180), bw_method=bw_method)
    # sns.kdeplot(all_angles[2], label=labels[2], clip=(0, 180), bw_method=bw_method)
    # sns.histplot(all_angles, kde=True, label=labels, bins=np.linspace(0, 180, 19))
    plt.grid(True)
    plt.xlabel(rf'Angle difference ($^\circ$)')
    plt.ylabel(rf'Percent')
    plt.savefig(f'../fig_paper/python/angles.pdf')
    plt.show()


def plot_all():
    test_path = f'../data/testset_random'
    results = {}
    get_angles_dist(nano=False, test_path=test_path, num_setting=False)
    results[r'Fab'] = get_angles_one(nano=False, test_path=test_path, num_setting=False)

    get_angles_dist(nano=True, test_path=test_path, num_setting=False)
    results['nAb'] = get_angles_one(nano=True, test_path=test_path, num_setting=False)

    get_angles_dist(nano=False, test_path=test_path, num_setting=False, suffix='_uy')
    results['$\overrightarrow{u_y}$'] = get_angles_one(nano=False, test_path=test_path, num_setting=False, suffix='_uy')
    plot_dict_hist(results)


def plot_ablation():
    test_path = f'../data/testset_random'
    results = {}
    get_angles_dist(nano=False, test_path=test_path, num_setting=False)
    results['normal'] = get_angles_one(nano=False, test_path=test_path, num_setting=False)

    get_angles_dist(nano=False, test_path=test_path, num_setting=False, suffix='_no_ot')
    results['no_ot'] = get_angles_one(nano=False, test_path=test_path, num_setting=False, suffix='_no_ot')

    get_angles_dist(nano=False, test_path=test_path, num_setting=False, suffix='_no_pd')
    results['no_pd'] = get_angles_one(nano=False, test_path=test_path, num_setting=False, suffix='_no_pd')

    get_angles_dist(nano=False, test_path=test_path, num_setting=False, suffix='_uy')
    results['uy'] = get_angles_one(nano=False, test_path=test_path, num_setting=False, suffix='_uy')
    plot_dict_hist(results)


if __name__ == '__main__':
    pass
    # get_angledist_results()
    # get_distances_all()
    plot_all()
    # plot_ablation()
