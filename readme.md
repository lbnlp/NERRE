# LLM-NERRE  - Structured Data Extraction

> For the publication "*Structured information extraction from scientific text with large language models*" in Nature Communications by John Dagdelen*, Alexander Dunn*, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin A. Persson, and Anubhav Jain.
> ###### * = Equal contribution

This repository contains code for extracting structured relational data as JSON documents from complex scientific text, with particular application to materials science.
For the Llama-2 fine-tuned models and code, see the supplemetary [nerre-llama](https://github.com/lbnlp/nerre-llama) repo.


## Contents

**General/MOF JSON models** (`general_and_mofs` subdirectory): 
 - Code reproducing the General-JSON and MOF-JSON models:
   - Code for fine-tuning GPT-3 models using the data shown in the paper to obtain similar results.
   - Code for preparing cross-validation splits for all models
   - Code for scoring results for all models
   - Initial inputs (annotations), intermediate files (if applicable), and final outputs for results shown in the paper for all models
   - For Llama-2 fine-tuning code and weights, see the supplementary [nerre-llama](https://github.com/lbnlp/nerre-llama) repo
 - Includes the annotation UI (including optional in-the-loop annotation if you have your own LLM fine tune) for annotating new datasets!

**Doping models** (`doping` subdirectory): 
- Code for reproducing the Doping models:
  - Code for fine-tuning GPT-3 models using the data shown in the paper to obtain similar results, for all three schema
  - Code for preparing train/test splits for all models/schemas
  - Code for scoring results for all models/schemas
  - Initial inputs (annotations), intermediate files (incl. decoded entries), and final outputs for results shown in the paper for all models/schemas
  - For Llama-2 fine-tuning code and weights for all schemas, see the supplementary [nerre-llama](https://github.com/lbnlp/nerre-llama) repo
- Includes annotation CLI for annotating new doping examples from your own data.


### Software requirements

- Python >3.7.3 (Unix, MacOS and Linux)
- Software has been run on Python 3.7.3 on MacOS Ventura 13.4.

Software requirements for specific python packages are given as `requirements.txt` files in each subdirectory (with required versions specified).

### Installing and running the evaluation code

Specific instructions for each task are given in the subdirectories in the `readme.md` files. Running the scripts is done either through the command line (`python <script_name.py> [options]`) or through Jupyter notebook. Running the scripts does not require installation, but does require the packages in the `requirements.txt` files.

Demo and expected output are given in the `readme.md` files in each subdirectory. Expected runtimes are several seconds to several minutes. 
