## New analyse

We dump the predicted pdbs directly to get interpretable results and errors.
We also make up to ten predictions to factor out the performance of the model and of the thresholding methods.
By monitoring failure cases, we remove a few buggy systems and correct the computations.
This is done in `predict_test` and `dockim_predict_test`.

Then, we compute the result of using fitmap in addition to crai in `fitmap.py`.
After running those steps, you should have `test_set/` and `test_set_random/` directories, populated with the cryo-EM
resampled maps, predicted structures by dockim, by crai and by crai+fitmap, along with pickle files holding the number
of antibodies present in the gt and predicted by the automatic method.
The files are a bit different for dockim and crai: for crai, they are sorted and numbered by increasing proba,
so they look like `crai_pred_{0,1,2...9}.pdb`.
For dockim though, since the program docks all systems at once, `dockim_pred_2.pdb` holds 3 antibody structures.

We then parse those results to get values for hit rates in the num and ab settings.
We get slightly different results, that do not indicate that dedicated models are much better.
Moreover, to assess the impact of under/overpredictions, we wonder if we would get all hits using ten predictions.
We screen from most probable to least probable and compute the fraction of detected systems for each number of
predictions.
This is done in `pr_curve.py`.

To assess the quality of the pose beyond F1, we compute distances and angles for our predictions and
for ablations in `get_distance_angles`.

Finally, we make predictions over some systems that do not contain antibodies.
In particular, we fetch all antigen chains uniprots'codes and find other PDBs where they occur.
We filter those PDBs to not be in SabDab or SabDab-nano, resulting in 33 apo systems.
We make predictions over those in the `no_abs.py` script.
