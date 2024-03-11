
## Initial analyse
By running `relog.py`, we obtain a pickle file whose name depends on the exact modalities used to get the results.
For instance, it includes the model used, nano, sorted and a suffix based on the thresholding method used.

Similar files are produced by running `benchmark.py` but this time with the output of dock in map.

Then those files are parsed with `analyse.py`. 
Because we originally wanted to compare our method with dockim, we don't use a GT vector, for instance for the number of abs per systems.
We have it implicitly, because we know that our predictions in the _num_ setting include exactly those numbers.

The computations of hit rates thus loop through this directory and return a certain number of distances to GT.
In the _num_ setting, this is fairly straightforward, and the main subtlety is the grouping or not of systems.

When using a thresh, it's a bit more subtle.
In `relog.py`
- if we underpredicted or if we predicted the right amount, just return the distances of the assigned antibodies
- if we overpredicted, we pad the results with n_over * [20,]

When parsing those results (in `analyse.py`), underpredictions are completed with n_under * [20,]

## New analyse

In the newer analyse, we dump the predicted pdbs directly to get interpretable results and errors.
We also make up to ten predictions to factor out the performance of the model and of the thresholding methods.
We thus reimplement HR computations, parsing the files with PyMol.
By monitoring failure cases, we remove a few buggy systems and correct the computations.
This is done in `predict_test` and `dockim_predict_test`.

We then parse those results to get values for hit rates in the num and ab settings.
We get slightly different results, that do not indicate that dedicated models are much better.
Moreover, to assess the impact of under/overpredictions, we wonder if we would get all hits using ten predictions.
We screen from most probable to least probable and compute the fraction of detected systems for each number of predictions.
This is done in `pr_curve`.

Finally, to assess the quality of the pose beyond hit_rates, we compute distances and angles for our predictions and 
for ablations in `get_distance_angles`.

