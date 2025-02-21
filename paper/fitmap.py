import os
import sys

import subprocess
import time
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import string_rep
from paper.predict_test import get_hit_rates


def compute_all_fitmap_time(recompute=True):
    """
    Get the fitmap result on just the right amount of systems
    :return:
    """

    # Iterate through all of our predictions to be fit in map
    chimerax_preamble = ['/usr/bin/chimerax', '--nogui', '--cmd']
    total_preds = 0
    t0 = time.time()
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in (True, False):
            import pickle
            pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
            num_pred_path = os.path.join(test_path, f'num_pred{"_nano" if nano else ""}.p')
            num_pred_all = pickle.load(open(num_pred_path, 'rb'))
            for (pdb, mrc, resolution), selections in tqdm(sorted(pdb_selections.items())):
                pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
                num_pred = num_pred_all[pdb]
                todo = []
                for i in range(num_pred):
                    pred_name = f'crai_pred{"_nano" if nano else ""}_{i}.pdb'
                    pred_path = os.path.join(pdb_dir, pred_name)
                    if not os.path.exists(pred_path):
                        continue
                    outfile = os.path.join(pdb_dir, 'fitmap_' + pred_name)
                    if not os.path.exists(outfile) or recompute:
                        todo.append((pred_path, outfile))
                total_preds += len(todo)
                if len(todo):
                    infile_mrc = os.path.join(pdb_dir, "full_crop_resampled_2.mrc")
                    chimerax_todo = f"open  {infile_mrc}; "
                    for i, (in_path, out_path) in enumerate(todo):
                        chimerax_todo += f"open {in_path} ; fit #{i + 2} inmap #1 ; save {out_path} #{i + 2} ; "
                    chimerax_todo += f'"{chimerax_todo} exit ;"'
                    subprocess.run(chimerax_preamble + [chimerax_todo], capture_output=True)
    print(f'Done {total_preds} in {time.time() - t0:.1f}s')


def compute_all_fitmap_all(recompute=False):
    """
    Get the tables results
    :return:
    """

    # Iterate through all of our predictions to be fit in map
    chimerax_preamble = ['/usr/bin/chimerax', '--nogui', '--cmd']
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for dir_file in tqdm(os.listdir(test_path)):
            if os.path.isdir(dir_file_path := os.path.join(test_path, dir_file)):
                todo = []
                for file in os.listdir(dir_file_path):
                    if file.startswith('crai_pred'):
                        infile = os.path.join(dir_file_path, file)
                        outfile = os.path.join(dir_file_path, 'fitmap_' + file)
                        if not os.path.exists(outfile) or recompute:
                            todo.append((infile, outfile))

                if len(todo):
                    infile_mrc = os.path.join(dir_file_path, "full_crop_resampled_2.mrc")
                    chimerax_todo = f"open  {infile_mrc}; "
                    for i, (in_path, out_path) in enumerate(todo):
                        chimerax_todo += f"open {in_path} ; fit #{i + 2} inmap #1 ; save {out_path} #{i + 2} ; "
                    chimerax_todo += f'"{chimerax_todo} exit ;"'
                    subprocess.run(chimerax_preamble + [chimerax_todo], capture_output=True)


def get_hr_all_fitmap():
    """
    Get the tables results
    :return:
    """
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            print('Getting data for ', string_rep(sorted_split=sorted_split, nano=nano))
            get_hit_rates(nano=nano, test_path=test_path, fitmap=True)
    test_path = f'../data/testset_random'
    get_hit_rates(nano=False, test_path=test_path, suffix='_no_ot', fitmap=True)
    get_hit_rates(nano=False, test_path=test_path, suffix='_no_pd', fitmap=True)
    get_hit_rates(nano=False, test_path=test_path, suffix='_uy', fitmap=True)


if __name__ == '__main__':
    # compute_all_fitmap_all()
    # compute_all_fitmap_time()
    get_hr_all_fitmap()
