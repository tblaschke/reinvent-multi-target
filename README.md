# Data set and source code for generative multi-target compound modeling

This repository holds the data sets and the code used for fine-tuning and sampling of multi-target compounds using REINVENT.

The scripts and folders are the following:

1) The extracted PubChem data sets can be found in the data folder.
2) The Python Notebooks and script contain all the code used for transfer learning, sampling, and the analysis of the sampled compounds.
3) `./reinvent` folder: Contains a modified version of REINVENT.
4) `./sampled` folder: Contains all configurations used for sampling of the multi-target compounds. The configuration files are needed to reproduce the sampling results.


## Data Set description

We provide a data set with 2809 multi-target (MT), 61,928  single-target (ST), and 295,395 and inactive (no-target; NT) compounds.
The compounds were extracted from PubChem assays using the following criteria:
1. Assays for individual human targets
2. Only Qualitative activity annotations were considered (‘active’ or ‘inactive’)
3. Inconsistently annotated or revoked assays were disregarded as well as assays imported from other databases (for external assays, negative test results were mostly missing)
4. Assays with an unusually high hit rate > 2% were eliminated
5. Screening compounds with aggregation or other assay interference (artifact) potential were discarded
6. Compounds were categorized into three groups: 
6.1. MT: Screening compounds with activity against five or more targets.
6.2. ST: Activity against only one target and confirmed inactivity against at least four other targets.
6.3. NT: No reported activity, but confirmed inactivity against at least five targets.

The collection of all extracted compounds can be found in `./data/all_extracted_pubchem_mt_st_nt_compounds.tsv.gz`.

`./datapubchem_assay_compounds_processed.tsv` contains all MT, ST, and a random subset of NT compounds. This file was used to perform the analysis if generative multi-target compound modeling was successful.
The SMILES used fine-tuning of REINVENT can be found in  `./data/pubchem_assay_compounds_processed_training.smi`.

## Differences of the included REINVENT and the upstream [REINVENT](https://github.com/MolecularAI/Reinvent)

We had to modify the publically available [REINVENT](https://github.com/MolecularAI/Reinvent) model to be able to perform this analysis.
The following changes were made and will be proposed to the upstream REINVENT repository:

1. Bug fixes in the NLL calculation inside REINVENT. The upstream version calculates a wrong NLL for the longest SMILES in a batch 
2. Added support for random seeds. We had to modify the configuration parser to allow for an additional parameter and added a helper function to set the seed.
3. Make logging of SMILES sampling optional. Sampling many SMILES (200M) was too slow because the tensorboard logger was creating too much disk i/o. 
4. Skip gradient computation when evaluating NLLs. REINVENT was always evaluating the gradient when computing the NLL for SMILES. 

## General Code Usage

The repository includes a Conda `environment.yml` file to create an Anaconda environment with all the required software dependencies.

~~~~bash
git clone https://github.com/tblaschke/reinvent-multi-target
cd reinvent-multi-target
conda env create -f environment.yml
conda activate einvent_shared.v2.1
~~~~
From here start a Jupyter Notebook Server and check out the notebooks and their description. 

## Generated Molecules and Fine-Tuned Models

Due to file size restriction in GitHub we had to deposit the fine-tuned models, the 200M sampled compounds including their fingerprints and descriptors on [Zenodo] (https://zenodo.org/record/4594647/).

You can download the files and unpack them in the this repository by executing:

~~~bash
cd reinvent-multi-target
curl -O https://zenodo.org/record/4594647/files/fine_tuned_models.tar.gz
curl -O https://zenodo.org/record/4594647/files/fingerprints_and_descriptors.tar.gz
curl -O https://zenodo.org/record/4594647/files/sampled_and_processed_multi_target_compounds.tar.gz

tar -xzvf  fine_tuned_models.tar.gz 
tar -xzvf  fingerprints_and_descriptors.tar.gz    
tar -xzvf  sampled_and_processed_multi_target_compounds.tar.gz

rm  fine_tuned_models.tar.gz fingerprints_and_descriptors.tar.gz sampled_and_processed_multi_target_compounds.tar.gz
~~~

## Support

If you have any questions please feel free to open an issue on GitHub or write a mail to thomas.blaschke@uni-bonn.de
