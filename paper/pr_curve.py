import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import string_rep


def compute_hr(nano=False, test_path='../data/testset', num_setting=False, dockim=False, suffix=''):
    """
    Compute the HR metric in the sense of the paper (using the actual number of prediction)
    :param nano:
    :param test_path:
    :return:
    """
    all_res_path = os.path.join(test_path, f'all_res{suffix}{"_dockim" if dockim else ""}{"_nano" if nano else ""}.p')
    all_res = pickle.load(open(all_res_path, 'rb'))

    num_pred_path = os.path.join(test_path, f'num_pred{suffix}{"_nano" if nano else ""}.p')
    num_pred_all = pickle.load(open(num_pred_path, 'rb'))

    all_hr = {}
    overpreds_list = []
    underpreds_list = []
    for pdb, (gt_hits_thresh, hits_thresh, resolution) in sorted(all_res.items()):
        # if pdb not in {'8HJ0', '7YVI', '7XDB', '7YAJ', '8D7E', '8GOC', '8GTP'}:
        #     continue
        if test_path == '../data/testset_random':
            # Systems containing both Fab and nAb in random split
            if pdb in ['7PIJ', '7SK5', '7WPD', '7XOD', '7ZLJ', '8HIK']:
                continue

        if test_path == '../data/testset':
            # Misclassified nano in sorted
            if pdb == '7YC5':
                continue

        gt_hits_thresh = np.array(gt_hits_thresh)
        hits_thresh = np.array(hits_thresh)
        num_gt = np.max(gt_hits_thresh)
        if num_setting:
            num_pred = num_gt
        else:
            num_pred = num_pred_all[pdb]
        overpreds = max(0, num_pred - num_gt)
        found_hits = hits_thresh[min(num_pred, 10) - 1]
        underpreds = num_gt - found_hits
        errors = overpreds + underpreds

        # PRINT
        if overpreds > 0:
            overpreds_list.append((pdb, overpreds))
            # Was this overpred useful ? (if found_hits > hits_thresh[num_gt - 1])
            # Actually useful only twice for fabs and twice for nano
            useful = found_hits > hits_thresh[num_gt - 1]
            print(f'over\t {pdb} num_pred : {num_pred}, num_gt : {num_gt}, found_hits : {found_hits}, '
                  f'hits with gt_num : {hits_thresh[num_gt - 1]}, raw results : {hits_thresh} '
                  f'useful overpred : {useful}')
        if underpreds > 0:
            # Would we find it with more hits ?
            # Not so much with Fabs, some are close but further than 10, others are just missed.
            # 100% yes with nano

            # more_would_help = hits_thresh[-1] > found_hits
            # print(f'under\t {pdb} num_pred : {num_pred}, num_gt : {num_gt}, found_hits : {found_hits}, '
            #       f'hits with gt_num : {hits_thresh[num_gt - 1]}, raw results : {hits_thresh} '
            #       f'more would help : {more_would_help}')
            underpreds_list.append((pdb, underpreds))
        # if overpreds > 0 and underpreds > 0:
        #     print(pdb, 'winner !')
        all_hr[pdb] = (errors, num_gt)
    print('Overpredictions : ', len(overpreds_list), sum([x[1] for x in overpreds_list]), overpreds_list)
    print('Underpredictions : ', len(underpreds_list), sum([x[1] for x in underpreds_list]), underpreds_list)
    failed_sys = [x[0] for x in overpreds_list + underpreds_list]

    hit_rate_sys = np.mean([100 * (1 - errors / num_gt) for errors, num_gt in all_hr.values()])
    hit_rate_ab = 100 * (1 - np.sum([errors for errors, _ in all_hr.values()]) / np.sum(
        [num_gt for _, num_gt in all_hr.values()]))
    print(f"{hit_rate_sys:.1f}")
    print(f"{hit_rate_ab:.1f}")
    return overpreds_list + underpreds_list


def compute_all():
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            print('Results HR for :', string_rep(sorted_split=sorted_split,
                                                 nano=nano,
                                                 dockim=True))
            compute_hr(test_path=test_path, nano=nano, num_setting=True, dockim=True)
            for num_setting in [True, False]:
                print('Results HR for :', string_rep(sorted_split=sorted_split,
                                                     nano=nano,
                                                     num=num_setting))
                compute_hr(test_path=test_path, nano=nano, num_setting=num_setting)
                # no specific nano model
                if not nano:
                    print('non mixed')
                    compute_hr(test_path=test_path, nano=nano, num_setting=num_setting, suffix='_fab')


def compute_ablations():
    test_path = '../data/testset_random'
    print("no_ot")
    compute_hr(test_path=test_path, nano=False, num_setting=True, suffix='_no_ot')
    compute_hr(test_path=test_path, nano=False, num_setting=False, suffix='_no_ot')
    print("no_pd")
    compute_hr(test_path=test_path, nano=False, num_setting=True, suffix='_no_pd')
    compute_hr(test_path=test_path, nano=False, num_setting=False, suffix='_no_pd')
    print("uy")
    compute_hr(test_path=test_path, nano=False, num_setting=True, suffix='_uy')
    compute_hr(test_path=test_path, nano=False, num_setting=False, suffix='_uy')


def get_mean_std(hitlist):
    hitlist = np.stack(hitlist)
    hitlist_mean = np.mean(hitlist, axis=0)
    hitlist_std = np.std(hitlist, axis=0)
    return hitlist_mean, hitlist_std


def plot_in_between(ax, x_axis, mean, std, **kwargs):
    ax.plot(x_axis, mean, **kwargs)
    upper = mean + std
    upper[upper > 1] = 1
    lower = mean - std
    lower[lower < 0] = 0
    ax.fill_between(x_axis, upper, lower, alpha=0.5)


def plot_pr_curve(nano=False, test_path='../data/testset', title=None, savefig=None):
    methods = {
        r'\texttt{CrAI}': os.path.join(test_path, f'all_res{"_nano" if nano else ""}.p'),
        r'\texttt{dock in map}': os.path.join(test_path, f'all_res_dockim{"_nano" if nano else ""}.p'),
    }
    # if not nano:
    #     methods['fab'] = os.path.join(test_path, f'all_res_fab.p')

    all_res_dict = {method: pickle.load(open(method_outpath, 'rb')) for method, method_outpath in methods.items()}
    all_parsed_dict = {method: ([], [], []) for method in all_res_dict.keys()}
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    for step, (pdb, mrc, resolution) in enumerate(pdb_selections.keys()):
        if test_path == '../data/testset':
            # Misclassified nano in sorted
            if pdb == '7YC5':
                continue
        if test_path == '../data/testset_random':
            # Systems containing both Fab and nAb in random split
            if pdb in ['7PIJ', '7SK5', '7WPD', '7XOD', '7ZLJ', '8HIK']:
                continue
        for method, res_dict in all_res_dict.items():
            gt_hits_thresh, hits_thresh, resolution = res_dict[pdb]
            gt_hits_thresh = np.array(gt_hits_thresh)
            hits_thresh = np.array(hits_thresh)
            num_gt = np.max(gt_hits_thresh)
            precision = hits_thresh / gt_hits_thresh
            method_lists = all_parsed_dict[method]
            method_lists[0].append(precision)
            method_lists[1].append(gt_hits_thresh / num_gt)
            method_lists[2].append(hits_thresh / num_gt)
            # if precision[-1] < 0.9:
            #     print(pdb, method, num_gt, hits_thresh)

    plotting_dict = {}
    for method, (all_precisions, all_gt, all_preds) in all_parsed_dict.items():
        all_precisions_mean, all_precisions_std = get_mean_std(all_precisions)
        all_gt_mean, all_gt_std = get_mean_std(all_gt)
        all_preds_mean, all_preds_std = get_mean_std(all_preds)
        plotting_dict[method] = (all_precisions_mean, all_precisions_std, all_gt_mean,
                                 all_gt_std, all_preds_mean, all_preds_std)

    # Setup matplotlib
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.5)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xlabel(r'Num prediction')
    ax.set_ylabel(r'Hits')
    x = range(1, 11)

    for i, (method, results) in enumerate(plotting_dict.items()):
        all_precisions_mean, all_precisions_std, all_gt_mean, all_gt_std, all_preds_mean, all_preds_std = results
        # plot_in_between(ax, x, all_precisions_mean, all_precisions_std, label=method)
        if i == 0:
            plot_in_between(ax, x, all_gt_mean, all_gt_std, label=r'\texttt{Ground Truth}')
        plot_in_between(ax, x, all_preds_mean, all_preds_std, label=method)

    # plt.xticks(['1', '2', '4', '6', '8', '10'])
    plt.xticks([1, 2, 4, 6, 8, 10])
    plt.xlim((1, 10))
    plt.ylim((0., 1.1))
    # plt.ylim((0.5, 1))
    plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


if __name__ == '__main__':
    pass
    # Compute the # systems with both Fabs and nanobodies as they bug the validation a bit
    # for sorted_split in [True, False]:
    #     test_path = f'../data/testset{"" if sorted_split else "_random"}'
    #     systems = []
    #     for nano in [True, False]:
    #         pdbsel = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    #         systems.append(pickle.load(open(pdbsel, 'rb')))
    #     nab_pdb = set([pdb for pdb, _, _ in systems[0].keys()])
    #     fab_pdb = set([pdb for pdb, _, _ in systems[1].keys()])
    #     print('Num in nAbs:', len(nab_pdb),
    #           'Num in Fabs:', len(fab_pdb),
    #           'Num in both:', len(nab_pdb.intersection(fab_pdb)))
    #     print(sorted(nab_pdb.intersection(fab_pdb)))

    # TO COMPUTE ONE
    test_path = f'../data/testset'
    # test_path = f'../data/testset_random'
    compute_hr(test_path=test_path, nano=False, num_setting=False)
    print("nano")
    compute_hr(test_path=test_path, nano=True, num_setting=False)
    # compute_hr(test_path=test_path, nano=True, use_mixed_model=True, num_setting=True, dockim=True)

    # # TO COMPUTE ALL
    # compute_all()

    # # TO COMPUTE ABLATIONS
    # compute_ablations()

    # TO PLOT ALL
    # for sorted_split in [True, False]:
    #     for nano in [True, False]:
    #         test_path = f'../data/testset{"" if sorted_split else "_random"}'
    #         sorted_title = rf'\texttt{{{"sorted" if sorted_split else "random"}}}'
    #         nano_title = f'{"VHH" if nano else "Fabs"}'
    #         title = rf'Hit rates for {sorted_title} {nano_title}'
    #         save_title = f'../fig_paper/python/{"sorted" if sorted_split else "random"}_{"nano" if nano else "fab"}.pdf'
    #         plot_pr_curve(nano=nano, test_path=test_path, title=title, savefig=save_title)
