# LLM-NERRE  - Structured Data Extraction

For the publication "*Structured information extraction from scientific text with large language models*" in Nature Communications by John Dagdelen*, Alexander Dunn*, Nicholas Walker, Sanghoon Lee, Andrew S. Rosen, Gerbrand Ceder, Kristin Persson, and Anubhav Jain.

This repository contains code for extracting structured relational data as JSON documents from complex scientific text, with particular application to materials science.


###### * = Equal contribution


**General/MOF JSON models**: 
 - For code reproducing the General-JSON and MOF-JSON models, see the `general_and_mofs` subdirectory.
 - Includes the annotation UI (including optional in-the-loop annotation if you have your own GPT-3 fine tune) for annotating new datasets!


**Doping models**: 
- For code reproducing the doping models, see the `doping` subdirectory.
- Includes annotation CLI for annotating new doping examples from your own data.


### Software requirements

- Python >3.7.3 (Unix, MacOS and Linux)
- Software has been run on Python 3.7.3 on MacOS Ventura 13.4.

Software requirements for specific python packages are given as `requirements.txt` files in each subdirectory (with required versions specified).

### Installing and running the evaluation code

Specific instructions for each task are given in the subdirectories in the `readme.md` files. Running the scripts is done either through the command line (`python <script_name.py> [options]`) or through Jupyter notebook. Running the scripts does not require installation, but does require the packages in the `requirements.txt` files.

Demo and expected output are given in the `readme.md` files in each subdirectory. Expected runtimes are several seconds to several minutes. 
