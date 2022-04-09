# Implementation for Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning

Modified by Kyle Williford, Jiehan Zhu - students at Georgia Institute of Technology, Spring 2022, CSE 6250 Big Data for Health Informatics

## Minimum System Requirements
* 4 Core CPU
* 24GB System RAM
* Recent Linux or MacOS (2020 or newer) - no Windows 10 compatibility
* Python 3.9.7
* HDD with 20GB free storage (for extracted datasets use only) or HDD with 100GB free storage if planning to download all MIMIC-III datasets and create a local Postgres database [instructions here if you plan to do this](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/buildmimic/postgres/README.md)
* IDE that can run Jupyter Notebooks, such as VSCode or PyCharm Professional Edition
* Anaconda 3 2021.11 or better
* Recent Nvidia GPU, at least 4GB VRAM (to load med7 model). This model can also be loaded and run with CPU / System RAM only

## Recommended System Requirements
* 8 Core CPU or better
* 32GB System RAM or more
* Recent Nvidia GPU with CUDA and Tensor Core architecture, at least 8GB VRAM
* SSD instead of HDD for faster data I/O

## Python Dependencies
See `environment.yml`

## Data & Pre-Trained Models
* [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/)
* [med7 pre-trained model](https://github.com/kormilitzin/med7)
* [Word2Vec and FastText pre-trained models](https://github.com/kexinhuang12345/clinicalBERT)

## Usage

0. Get credentialed for [MIMIC-III dataset, version 1.4](https://physionet.org/content/mimiciii/1.4/) (this may take several days)

0. Download the required datasets (`ADMISSIONS.csv.gz`, `NOTEEVENTS.csv.gz`, `ICUSTAYS.csv.gz`)

0. Clone the repo  
```
$ git clone https://github.com/KyleWilliford/ConvolutionMedicalNer.git
$ cd ConvolutionMedicalNer
$ conda env create --file=environment.yml
$ conda activate cse6250
$ pip install --upgrade pip
```

0. Extract `ADMISSIONS.csv.gz`, `NOTEEVENTS.csv.gz`, `ICUSTAYS.csv.gz` zips into `data` folder.

0. Download [pre-trained embeddings](https://github.com/kexinhuang12345/clinicalBERT) into `embeddings` folder

0. Run MIMIC-Extract Pipeline as explained in https://github.com/MLforHealth/MIMIC_Extract or use the pre-extracted dataset [here](https://console.cloud.google.com/storage/browser/mimic_extract;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). If you do not have access, return to step 0.  

0. Copy the output file of MIMIC-Extract Pipeline named `all_hourly_data.h5` to `data` folder.

0. Run `01-Extract-Timseries-Features.ipnyb` to extract first 24 hours timeseries features from MIMIC-Extract raw data.

0. Run `02-Select-SubClinicalNotes.ipynb` to select subnotes based on criteria from all MIMIC-III Notes.

0. Run `03-Prprocess-Clinical-Notes.ipnyb` to prepocess clinical notes.

The following steps can be run with `python` instead of a Jupyter Notebook.

7. Run `rewrite/04-Apply-med7-on-Clinical-Notes.py` to extract medical entities. 

0. Run `rewrite/05-Represent-Entities-With-Different-Embeddings.py` to convert medical entities into word representations.

0. Run `rewrite/06-Create-Timeseries-Data.py` to prepare the timeseries data to fed through GRU / LSTM.

0. Run `rewrite/07-Timeseries-Baseline.py` to run timeseries baseline model to predict 4 different clinical tasks.

0. Run `rewrite/08-Multimodal-Baseline.py` to run multimodal baseline to predict 4 different clinical tasks.

0. Run `rewrite/09-Proposed-Model.py` to run proposed model to predict 4 different clinical tasks.

## References

Original project this is forked from https://github.com/tanlab/ConvolutionMedicalNer

Download the MIMIC-III dataset via https://mimic.physionet.org/

MIMIC-Extract implementation: https://github.com/MLforHealth/MIMIC_Extract

med7 implementation: https://github.com/kormilitzin/med7

Download Pre-trained Word2Vec & FastText embeddings: https://github.com/kexinhuang12345/clinicalBERT

Preprocessing Script: https://github.com/kaggarwal/ClinicalNotesICU

