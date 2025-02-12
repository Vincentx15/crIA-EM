from chimerax.core.commands import CmdDesc
from chimerax.core.commands import FloatArg, StringArg, IntArg, BoolArg
from chimerax.core.commands import OpenFileNameArg, SaveFileNameArg, Or
from chimerax.core.commands import run
from chimerax.map import MapArg
from chimerax.map.volume import Volume

import os
import sys

import pathlib
import time
import numpy as np

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir))

import utils_mrc
from SimpleUnet import SimpleHalfUnetModel
from utils_object_detection import output_to_transforms, transforms_to_pdb_biopython


def get_volume_from_path(session, map_path):
    output_run = run(session, f"open {map_path}")
    map_id = output_run[0].id_string
    # #1.1 =>(1,1) which is what's keyed by session._models.
    as_tuple = tuple([int(x) for x in map_id.split('.')])
    queried = session.models._models[as_tuple]
    if not isinstance(queried, Volume):
        raise ValueError(f"Expected the path to refer to map data, got {queried}")
    return queried


def get_mrc_from_input(session, path_or_volume):
    # If it's a path, open it in Chimerax and get its id.
    if not isinstance(path_or_volume, Volume):
        if not isinstance(path_or_volume, str):
            raise ValueError(f"Expected the path to be either a volume or a path, got {path_or_volume}")
        volume = get_volume_from_path(session, path_or_volume)
    else:
        volume = path_or_volume
    queried_data = volume.full_matrix().transpose((2, 1, 0))
    mrc = utils_mrc.MRCGrid(data=queried_data,
                            voxel_size=volume.data.step,
                            origin=volume.data.origin,
                            )
    return volume.id_string, mrc


def clean_mrc(mrc, resample=True, crop=0, normalize='max', min_val=None):
    if resample:
        mrc = mrc.resample()
    mrc = mrc.normalize(normalize_mode=normalize, min_val=min_val)
    if crop != 0:
        mrc = mrc.crop(*(crop,) * 6)
    else:
        mrc = mrc.crop_large_mrc()
    return mrc


def get_outname(session, map_path, outname=None):
    """
    If no outname is given, try to get a default
    :param map_path:
    :param outname:
    :return:
    """
    if outname is not None:
        suffix = pathlib.Path(outname).suffix
        if suffix not in {'.pdb', '.cif'}:
            outname += '.pdb'
        return outname

    if isinstance(map_path, Volume):
        default_outname = "crai_prediction.pdb"
    else:
        default_outname = map_path.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
    if not os.path.exists(default_outname):
        return default_outname
    else:
        print("Default name not available, one could not save the prediction. Please add an outname.")
        return None


def predict_coords(session, mrc, outname=None, outmrc=None, n_objects=None, thresh=0.2, default_nano=False,
                   use_pd=True, split_pred=True):
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    model_path = os.path.join(script_dir, "data/ns_final_last.pth")
    model = SimpleHalfUnetModel(classif_nano=True, num_feature_map=32)
    model.load_state_dict(torch.load(model_path))

    t0 = time.time()
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    print(f'Done prediction in : {time.time() - t0:.2f}s')
    transforms = output_to_transforms(out, mrc, n_objects=n_objects, thresh=thresh, outmrc=outmrc,
                                      classif_nano=True, default_nano=default_nano, use_pd=use_pd)
    outnames = transforms_to_pdb_biopython(transforms=transforms, outname=outname, split_pred=split_pred)
    print('Output saved in ', outnames[0])
    return outnames


def crai(session, density, outName=None, usePD=True, nObjects=None, splitPred=True, fitMap=True, minVal=None):
    """

    :param session:
    :param densityMap:
    :param outname:
    :param test_arg:
    :return:
    """
    # print("Torch version used: ", torch.__version__)
    # print("Torch path used: ", os.path.abspath(torch.__file__))
    t0 = time.time()
    map_id, mrc = get_mrc_from_input(path_or_volume=density, session=session)
    mrc = clean_mrc(mrc, min_val=minVal)
    print(f'Data loaded in : {time.time() - t0:.2f}s')

    outname = get_outname(outname=outName, map_path=density, session=session)
    if outname is None or mrc is None:
        return None
    outnames = predict_coords(mrc=mrc, outname=outname, use_pd=usePD, n_objects=nObjects, session=session,
                              split_pred=splitPred)
    for outname in outnames:
        ab = run(session, f"open {outname}")
        if fitMap:
            run(session, f"fit #{ab[0].id_string} inmap #{map_id}")


crai_desc = CmdDesc(required=[("density", Or(MapArg, OpenFileNameArg))],
                    keyword=[("outName", SaveFileNameArg),
                             ("usePD", BoolArg),
                             ("nObjects", IntArg),
                             ("splitPred", BoolArg),
                             ("fitMap", BoolArg),
                             ("minVal", FloatArg),
                             ], )
