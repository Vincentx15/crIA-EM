import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.SimpleUnet import SimpleHalfUnetModel
from utils import mrc_utils
from utils.object_detection import output_to_transforms, transforms_to_pdb


def crop_large_mrc(mrc, margin=12):
    arr = mrc.data
    to_find = arr > 0.05
    res = np.nonzero(to_find)
    all_min_max = []
    for r, shape in zip(res, to_find.shape):
        min_i, max_i = np.min(r), np.max(r)
        min_i, max_i = max(0, min_i - margin), min(max_i + margin, shape)
        all_min_max.append(min_i)
        # The max_i is computed from the end
        all_min_max.append(shape - max_i)
    return all_min_max


def predict_coords(mrc_path, model, resample=True, normalize='max', outname=None, outmrc=None,
                   n_objects=None, thresh=0.5, crop=0, classif_nano=False, default_nano=False, use_pd=False):
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    if resample:
        mrc = mrc.resample()
    mrc = mrc.normalize(normalize_mode=normalize)
    if crop != 0:
        mrc = mrc.crop(*(crop,) * 6)
    else:
        all_min_max = crop_large_mrc(mrc)
        mrc = mrc.crop(*all_min_max)
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    transforms = output_to_transforms(out, mrc, n_objects=n_objects, thresh=thresh,
                                      outmrc=outmrc, classif_nano=classif_nano, default_nano=default_nano,
                                      use_pd=use_pd)
    if outname is not None:
        transforms_to_pdb(transforms=transforms, out_name=outname)
    return transforms


if __name__ == '__main__':
    pass

    datadir_name = "../data/pdb_em"
    # datadir_name = ".."
    # dirname = '7V3L_31683' # present in train set
    # dirname = '7LO8_23464'  # this is test set
    # dirname = '6BF9_7093'  # this is test set
    # dirname = '8DG9_27419'  # this is test set
    # dirname = '7DCC_30635'  # this is test set
    # dirname = '6NQD_0485'  # this is test set
    # dirname = '6VJA_21212'  # this is close Fvs
    # dirname = '7YM8_33924'  # this is test set
    # dirname = '8GNI_34165'  # this is test set
    # dirname = '8HBV_34644'  # this is test set
    dirname = '8HIJ_34818'  # this is test set
    pdb_name, mrc_name = dirname.split("_")
    # mrc_path, small = os.path.join(datadir_name, dirname, "resampled_0_2.mrc"), True
    mrc_path, small = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map"), False
    # mrc_path, small = os.path.join(datadir_name, dirname, "full_crop_resampled_2.mrc"), False

    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # align_output(fake_out, mrc)

    # model_name = 'object_best'
    # model_name = 'object_2_best'
    # model_name = 'object_3_best'
    # model_name = 'crop_95'
    # model_name = 'crop_256'
    # model_name = 'focal_332'
    # model_name = 'less_agg_432'
    # model_name = 'multi_train_339'
    # model_name = 'multi_train_861'
    # model_name = 'big_train_gamma_last'
    model_name = 'ns_final_last'
    model_path = os.path.join('../saved_models', f"{model_name}.pth")
    # model = HalfUnetModel(out_channels_decoder=128,
    #                       num_feature_map=24,
    #                       )
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                classif_nano=True,
                                num_feature_map=32)
    model.load_state_dict(torch.load(model_path))
    dump_name = f"{model_name}_{'small' if small else 'large'}.pdb"
    dump_path = os.path.join(datadir_name, dirname, dump_name)
    out_mrc = dump_path.replace(".pdb", "pred.mrc")
    n_objects = None
    thresh = 0.2
    use_pd = True
    crop = 0
    classif_nano = True
    default_nano = False
    normalize = 'max'
    predict_coords(mrc_path=mrc_path, model=model, outname=dump_path, outmrc=out_mrc, normalize=normalize,
                   n_objects=n_objects, thresh=thresh, crop=crop, classif_nano=classif_nano, default_nano=default_nano,
                   use_pd=use_pd)

    # align nano_2, nano, cycles=0, transform=0, object=aln
    # rms_cur (nano_2 and aln), (nano & aln), matchmaker=-1
