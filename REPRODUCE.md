## Data processing

The steps to gather the database are detailed in prepare_database/README.md.
This will equip you with csvs, structures and maps.
The csvs detail all antibodies examples split following our two possible splits (random and sorted).
The aligned structures and maps are found in data/pdb_em.

Then, the way to load those raw datas in pytorch datasets is detailed in the load_data/ folder.

## Training a model

Once the data steps have been carried out, simply run :
```bash
cd learning/
python train_coords.py --sorted -m example_sorted
```

## Validation of the results


### Producing results files

To make predictions using our different models and ablations, please run:
```bash
cd paper
python predict_test.py
python dockim_predict_test.py
```

This will produce 10 predictions for each systems with several settings, as well as the ablations for the random fab split.
In addition, this creates a pickle file that stores how many systems were output by the thresh model, as well as one
containing lists of len 10 containing the number of GT systems detected if retaining at most k predictions.


### Producing figures

Once such files are produced, you can reproduce the figures of the paper by running 
```bash
python get_distances_angles.py
python pr_curve.py
```

