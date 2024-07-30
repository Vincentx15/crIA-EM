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
from utils.rotation import rotation_to_supervision


def match_transforms_to_angles_dist(gt_transforms, pred_transforms):
    # TODO : check RMSD
    gt_translations = [x[1] for x in gt_transforms]
    gt_rotations = [x[2] for x in gt_transforms]
    gt_rz, gt_theta_u = [], []
    for rotation in gt_rotations:
        rz, theta = rotation_to_supervision(rotation)
        theta_u = [np.cos(theta), np.sin(theta)]
        gt_rz.append(rz)
        gt_theta_u.append(theta_u)

    pred_translations = [x[1] for x in pred_transforms]
    pred_rotations = [x[2] for x in pred_transforms]
    pred_rz, pred_theta_u = [], []
    for rotation in pred_rotations:
        rz, theta = rotation_to_supervision(rotation)
        theta_u = [np.cos(theta), np.sin(theta)]
        pred_rz.append(rz)
        pred_theta_u.append(theta_u)

    dist_matrix = scipy.spatial.distance.cdist(pred_translations, gt_translations)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
    position_dists = dist_matrix[row_ind, col_ind]

    def get_angle(u1, u2):
        # just return the angle between vectors
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        return np.arccos(np.dot(u1, u2)) * 180 / 3.14

    p_angles = [get_angle(pred_rz[row], gt_rz[col]) for row, col in zip(row_ind, col_ind)]
    theta_angles = [get_angle(pred_theta_u[row], gt_theta_u[col]) for row, col in zip(row_ind, col_ind)]
    return position_dists, p_angles, theta_angles


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
            dists, angles_p, angles_theta = match_transforms_to_angles_dist(gt_transforms, pred_transforms)
            all_res[pdb] = dists, angles_p, angles_theta
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
        dists, angles_p, angles_theta = match_transforms_to_angles_dist(gt_transforms, pred_transforms)
        all_res[pdb] = dists, angles_p, angles_theta
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

    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    all_sel_dists = list()
    # useful for resolution plot
    all_pdb_dists = dict()
    for step, ((pdb, mrc, resolution), selections) in enumerate(sorted(pdb_selections.items())):
        gt_num = len(selections)
        pdb_res = results[pdb]
        if pdb_res is None:
            continue
        dists, *_ = pdb_res
        # expand dists with underpreds
        dists = np.array(list(dists) + [10 for _ in range(gt_num - len(dists))])

        sel_dists = dists[dists < thresh]
        # if sum(dists > thresh) > 0:
        #     print(dockim, nano, num_setting, pdb)
        all_sel_dists.extend(sel_dists)
        all_pdb_dists[pdb] = list(dists)
    print(f'{np.mean(all_sel_dists):.2f}')
    return all_sel_dists, all_pdb_dists


def get_distances_all():
    res_dict = {}
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            print('Doing ', string_rep(sorted_split=sorted_split, nano=nano))
            dists = get_distance_one(nano=nano, test_path=test_path, dockim=True, num_setting=True)
            res_dict[(True, sorted_split, nano, True)] = dists
            for num_setting in [True, False]:
                dists = get_distance_one(nano=nano, test_path=test_path, num_setting=num_setting)
                res_dict[(False, sorted_split, nano, num_setting)] = dists
    return res_dict


def scatter(proba, distances, alpha=0.3, noise_strength=0.02, fit=True, display_fit=True, colors=None, label=None):
    # Adding random noise to the data

    proba += noise_strength * np.random.randn(len(proba))
    distances += noise_strength * np.random.randn(len(distances))

    # Rest of the plot decorations
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    # a= r'\texttt{dock\_in\_map} Template'
    plt.rc('grid', color='grey', alpha=0.2)
    plt.grid(True)
    # Plotting the scatter data with transparency
    if colors is None:
        colors = sns.color_palette('colorblind', as_cmap=True)
    plt.scatter(proba, distances, color=colors[0], marker='o', alpha=alpha)

    if fit:
        # Linear fit
        m, b = np.polyfit(proba, distances, 1)
        x = np.linspace(proba.min(), proba.max())
        plt.plot(x, m * x + b, color=colors[1],label=label)
        # plt.plot(all_probas_bench, m * all_probas_bench + b, color='red', label=f'Linear Fit: y={m:.2f}x+{b:.2f}')

        if display_fit:
            # Calculating R^2 score
            predicted = m * proba + b
            from sklearn.metrics import r2_score
            r2 = r2_score(distances, predicted)
            plt.text(0.68, 0.86, rf'$y = {m:.2f} x + {b:.2f}$', transform=plt.gca().transAxes)
            plt.text(0.66, 0.8, rf'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes)


def resolution_plot(sys=False, num_setting=False, dockim=False):
    concatenated_dists = list()
    concatenated_res = list()
    res_dict = get_distances_all()
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
            all_pdb_dists = res_dict[(dockim, sorted_split, nano, num_setting)][1]
            for step, ((pdb, mrc, resolution), selections) in enumerate(sorted(pdb_selections.items())):
                if pdb not in all_pdb_dists:
                    continue
                relevant_dists = [x if x < 10 else 10 for x in all_pdb_dists[pdb]]
                if sys:
                    concatenated_dists.append(np.mean(relevant_dists))
                    concatenated_res.append(resolution)
                else:
                    concatenated_dists.extend(relevant_dists)
                    concatenated_res.extend([resolution for _ in relevant_dists])
    # print(len(concatenated_dists), concatenated_dists)
    # print(len(concatenated_res), concatenated_res)
    concatenated_dists = np.asarray(concatenated_dists)
    print(np.sum(concatenated_dists > 9), len(concatenated_dists))
    scatter(concatenated_res, concatenated_dists, fit=True)
    plt.xlabel(r'Resolution (\AA{})')
    plt.ylabel(r'Distance (\AA{})')
    # plt.legend(loc='lower left')
    plt.savefig(f'../fig_paper/python/resolution{"_num" if num_setting else ""}{"_dockim" if dockim else ""}.pdf')
    plt.show()
    return concatenated_res, concatenated_dists


def resolution_plot_both():
    # res_dockim, dists_dockim = resolution_plot(dockim=True, num_setting=True)
    # res_num, dists_num = resolution_plot(dockim=False, num_setting=True)
    # res_thresh, dists_thresh = resolution_plot(dockim=False, num_setting=False)
    #
    # pickle.dump((res_dockim, dists_dockim, res_num, dists_num, res_thresh, dists_thresh), open('temp.p', 'wb'))
    res_dockim, dists_dockim, res_num, dists_num, res_thresh, dists_thresh = pickle.load(open('temp.p', 'rb'))
    colors = sns.color_palette('Paired', n_colors=6)

    scatter(res_dockim, dists_dockim, colors=['red'] * 2, display_fit=False, label=r'\texttt{dock in map}')
    # scatter(res_num, dists_num, colors=colors[2:], display_fit=False)
    scatter(res_thresh, dists_thresh, colors=['darkviolet'] * 2, display_fit=False, label=r'\texttt{CrAI}')

    plt.xlabel(r'Resolution (\AA{})')
    plt.ylabel(r'Distance (\AA{})')
    plt.legend(loc='center right')
    plt.savefig(f'../fig_paper/python/resolution_both.pdf')
    plt.show()


def get_angles_one(test_path=None, dockim=False, nano=False, num_setting=False, suffix='', thresh=10):
    if dockim:
        pickle_name_to_get = f'dockim_angledist{"_nano" if nano else ""}.p'
    else:
        pickle_name_to_get = f'angledist{suffix}{"_nano" if nano else ""}{"_num" if num_setting else "_thresh"}.p'
    outname_results = os.path.join(test_path, pickle_name_to_get)
    results = pickle.load(open(outname_results, 'rb'))
    sel_angles_p = list()
    sel_angles_theta = list()
    for pdb, pdb_res in results.items():
        if pdb_res is None:
            continue
        dists, angles_p, angles_theta = pdb_res
        selector = dists < thresh
        sel_angles_p.extend(list(np.asarray(angles_p)[selector]))
        sel_angles_theta.extend(list(np.asarray(angles_theta)[selector]))
    print(f"For main angle: {np.mean(sel_angles_p):.2f} +/- {np.std(sel_angles_p):.2f}")
    print(f"For theta angle: {np.mean(sel_angles_theta):.2f} +/- {np.std(sel_angles_theta):.2f}")
    print()
    return sel_angles_p


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
    # get_distances_all()

    # resolution_plot(dockim=True, num_setting=True)
    # resolution_plot(dockim=False, num_setting=True)
    # resolution_plot(dockim=False, num_setting=False)
    resolution_plot_both()

    # get_angledist_results()
    # plot_all()
    # plot_ablation()
