# Doping NERRE


1. `step0_annotate.py`: Annotate data using CLI tool (alternatively, annotate however you like). The 193 total abstracts used to form the train and test sets (`data/raw_data.json`) are used as the default.
   1. You can annotate all raw data together as one file, then split on train+val/test later.
   2. The train+val and test sets used in the publication are pre-saved as `data/train.json` and `data/test.json`, respectively.
2. `step1_train`