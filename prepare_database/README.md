## Data processing pipeline

These are the steps to follow in order to get the list of systems as well as the structures and maps used for learning.
The resulting list of systems is also present in the git.


### Getting the initial data

The data was originally fetched from SabDab in July 2023. 
We manually curated it, as several lines pertain to Fabs instead of nanobodies.
We explain our manual curation steps in the file named `manual_curation.txt`.
The resulting initial files are in `ROOT/data/csvs/fabs.tsv` and `ROOT/data/nano_csvs/nanobodies.tsv`.


### Getting the right pairs and downloading the structure and map files.

The PDB ids and chain selections are retrieved, and stored in `cleaned.csv`.
Then, using the PDB, we find the corresponding cryo-em maps and build a mapping pdb_id : em_id.
We add the mrc column and dump `mapped.csv`

Using this intermediary result, we download all corresponding maps and cifs.
```bash
python download_data.py
```

### Filtering the database

Starting from mapped.csv, the parsed output of SabDab, we add missing resolutions by opening the cif files,
yielding the `resolution.csv` file.
Finally, we process the csv without using validation and docking scores : we simply remove systems with resolution
below 10A or ones with no antibodies or antigen chains, and group pdb_ids together.
We dump `filtered.csv` and split it to obtain `filtered_{train,val,test}.csv`.
```bash
python filter_database.py
```

In a deprecated version, we also tried to use phenix to score our systems and keep only the ones with a good fit. 
However, it crashed on many systems and analysing those results made us realize that these scores were not great even for correct files.
This can be explained by missing B-factors.
Thus, we deprecated these steps.

### Processing the database

Once equipped with those splits we are ready to do machine learning.
However, the map files obtained from the PDB can be enormous (up do 1000³ grid cells) and not centered for viruses :
the pdb only occupies a fraction of the map.
To deal with this we replace the original maps with ones centered around the PDB with a margin of 25A, resampled with a
voxel size of 2 in files named `f"full_crop_resampled_2.mrc"`.
We also provide cropping around the antibodies and their antigen to get even smaller maps for learning.
To do so, we filter out redundant copies of antibodies (for symmetrical systems) and dump cropped mrc files, along with
a csv keeping track of those systems `chunked_{train,val,test}.csv`.

### Template management

The last thing we need for object detection is an antibody template that serves to transform an antibody into a
translation and rotation (using Pymol align).
To get this template, we pick a random Fab system and manually select the Fv residues.
Then we shift the system so that the Fv is centered at the origin and align its main axis with uz vector using PCA.
Given an antibody, pymol align now gives us a translation and rotation to transform our template into it.

## Nanobodies

All of these steps apply to the production of nanobody data.
We start with the .tsv result of cryo-EM systems containing nanobodies
