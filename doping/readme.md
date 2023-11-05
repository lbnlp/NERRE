# Doping NERRE

Annotating, training, and evaluating the LLM-NERRE doping experiments requires a bit more code as entries are broken down on a sentence-by-sentence basis. This directory contains code for annotation, training with GPT-3, and evaluating on the test set. You may also use these scripts to run full-scale inference experiments with your own models. **Note: The GPT-3 models used in the publication require private API access and therefore cannot be called; you can, however, train your own model (~$5 with the data provided here) and obtain similar results to those shown in the paper.** Inputs and outputs used in the paper are provided for all intermediate steps such that each step can otherwise be reproduced without access to the model.

*You should not use the annotation UI for the doping task, you should use the annotation script below.*


There are 3 scripts:

1. `step1_annotate.py`: Annotate data using CLI tool (alternatively, annotate however you like). The 193 total abstracts used to form the train and test sets (`data/raw_data.json`) are used as the default.
   1. You can annotate all raw data together as one file, then split on train+val/test later.
   2. The train+val and test sets used in the publication are pre-saved as `data/train.json` and `data/test.json`, respectively.
2. `step2_train_predict.py`: Train a new model using the annotated data using your schema of choice (Eng, EngExtra, or JSON). Also contains code for inferring entries with GPT-3 using the trained model. All requisite JSONL files are automatically written. Optional sentencewise JSONL files are written as well to facilitate use of this data outside of the code framework here (i.e., directly with openai API.) Optional intermediate files are also written for reproducibility and troubleshooting.
3. `step3_evaluate.py`: Evaluate the trained model on the test set. Works with all three schemas presented in the publication.


The expected score results for each of two models (llama2, gpt3) for each of three schemas (Doping-JSON, Doping-English, DopingExtra-English) as well as MatBERT-Proximity and seq2rel are shown int he `all_doping_results_output_summary.txt` file.

### For any of the above scripts, pass --help in order to see all available options.

### For llama2 fine tuning, see the supplementary repo: [https://github.com/lbnlp/nerre-llama](https://github.com/lbnlp/nerre-llama). 

### Included data files

All data files are included in the `data` subdirectory.

- `raw_data.json`: All raw data for the training/test sets (not including annotations, just titles, abstracts, and dois.)
- `train.json`: The annotated train+val set used in the publication.
- `test.json`: The annotated test set used in the publication.
- `training_*.jsonl`: JSONL files created by `step2_train_predict.py` for training the model.
- `inference_raw_*.json`: Raw JSON string outputs created by `step2_train_predict.py` from inferred entities. The included files correspond to each schema as shown in paper.
- `inference_raw_*.jsonl`: JSONL equivalents of `inference_raw_*.json` files, for convenience in case you want to evaluate them in an alternative manner.
- `inference_decoded_*.json`: The "decoded" versions of the raw inference outputs. These include the raw strings output by the LLM converted into structured entries which can be more easily evaluated.

Scripts use defaults to reproduce data shown in the paper, but most input/output files can be specified as needed to fit your use case (e.g., using a different test set).

## Installing requirements
These scripts were tested with python 3.7.3, though they may work with later versions of python as well.

```bash
$: pip install -r requirements.txt
```
*Note: An LLM used in this work is GPT-3, which is dependent on the OpenAI API. The format of the API may change over time and we do not guarantee the API will continue working as is shown here.*

## Example usage: Train your own model to obtain similar results to those shown in the paper

Train an LLM model using the data in the paper with the Eng schema.
```bash
$: python step2_train_predict.py train --openai_api_key "YOUR-API-KEY" --schema_type "eng"
```

Output:
```text
Doing 'train' operation with schema type 'eng'.
Using training json of REPO_DIR/doping/data/train.json, saving formatted output file to None.
Training JSONL file will be saved to REPO_DIR/doping/data/training_eng_2023-06-13_15.41.37.jsonl
Inference JSONL file will be saved to REPO_DIR/doping/data/inference_raw_eng_2023-06-13_15.41.37.json
loading training set
training set loaded.
file written to REPO_DIR/doping/data/training_eng_2023-06-13_15.41.37.jsonl with 413 sentence samples.
JSONL written to REPO_DIR/doping/data/training_eng_2023-06-13_15.41.37.jsonl.
Upload progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128k/128k [00:00<00:00, 87.0Mit/s]
Uploaded file from REPO_DIR/doping/data/training_eng_2023-06-13_15.41.37.jsonl: file-Dv4w3cI1UsVr3gT0FgGwJAY2
Created fine-tune: ft-VSuVpsDctSnaMUPOSFazU3BW
Streaming events until fine-tuning is complete...

(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[2023-06-13 15:43:26] Created fine-tune: ft-VSuVpsDctSnaMUPOSFazU3BW
[2023-06-13 15:44:59] Fine-tune costs $6.48
[2023-06-13 15:44:59] Fine-tune enqueued. Queue number: 0
Stream interrupted (client disconnected).
To resume the stream, run:

  openai api fine_tunes.follow -i ft-VSuVpsDctSnaMUPOSFazU3BW
```

When the training completes, predict on the test set:

```bash
$: python step2_train_predict.py predict \
  --openai_api_key "YOUR-API-KEY" \
  --schema_type="eng" \
  --inference_model_name "davinci:ft-matscholar-2022-08-30-05-13-11"
```
*Note: You will need to replace the model name with your actual model name*. 

Output:
```text
Doing 'predict' operation with schema type 'eng'.
Using training json of REPO_DIR/doping/data/train.json, saving formatted output file to None.
Training JSONL file will be saved to REPO_DIR/doping/data/training_eng_2023-06-13_15.58.18.jsonl
Inference JSONL file will be saved to REPO_DIR/doping/data/inference_raw_eng_2023-06-13_15.58.18.json
Loaded 31 samples for inference.
Using davinci:ft-matscholar-2022-08-30-05-13-11 for prediction
Texts processed: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:20<00:00,  2.58s/it]
Dumped 31 total to REPO_DIR/doping/data/inference_raw_eng_2023-06-13_15.58.18.json (and raw jsonl to REPO_DIR/doping/data/inference_raw_eng_2023-06-13_15.58.18.jsonl).
  0%|                                                                                                                                                                                                                                 | 0/31 [00:00<?, ?it/s]step2_train_predict.py:212: UserWarning: Line 'Arsenic' is a p-Type dopant gave no parsable data!
  warnings.warn(f"Line {l} gave no parsable data!")
step2_train_predict.py:212: UserWarning: Line 'chemical substitution' is a type of 'doping' gave no parsable data!
  warnings.warn(f"Line {l} gave no parsable data!")
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 18427.36it/s]
Decoded 31 samples to file REPO_DIR/doping/data/inference_decoded_eng_2023-06-13_15.58.18.json
```

Now evaluate the predicted data compared to the true test set annotations, including sequence accuracy similarity according to Jaro-Winkler using the Eng schema:
```bash
$: python step3_score.py eng --pred_file "data/llama2/inference_decoded_eng.json"
```

Output:
```text
Scoring outputs using 
        test file: /Users/ardunn/ardunn/lbl/nlp/ardunn_text_experiments/nerre_official_provenance_repository/doping/data/test.json
        pred file: data/llama2/inference_decoded_eng.json
basemats: prec=0.8383838383838383, recall=0.9325842696629213, f1=0.8829787234042553
dopants: prec=0.8372093023255814, recall=0.8571428571428571, f1=0.8470588235294118
triplets: prec=0.7868852459016393, recall=0.8421052631578947, f1=0.8135593220338982
      metric         entity     score
0  precision       basemats  0.838384
1     recall       basemats  0.932584
2         f1       basemats  0.882979
3  precision        dopants  0.837209
4     recall        dopants  0.857143
5         f1        dopants  0.847059
6  precision  link triplets  0.786885
7     recall  link triplets  0.842105
8         f1  link triplets  0.813559
Total sequences was: 77
Frac. Sequences parsable:  1.0
Avg sequence similarity:  0.9461784105659841
Frac. of sequences exactly correct:  0.6493506493506493
Support was:  {'ents': {'basemats': 60, 'dopants': 76},
 'links_ents': 72,
 'links_words': 114,
 'words': {'basemats': 111, 'dopants': 110}}
```

*Note: Ensure each step of the train/predict/evaluate pipeline is run using the same schema, otherwise you will not get the correct performance.* 

*Note 2: Defaults are provided for each file to reproduce the data shown in the paper, but you can swap out the files as needed to test your own model's performance (say, on an alternative test set) provided they adhere to the correct format.**

## Docstrings for each script

### Annotating a new dataset
```text
$: python step1_annotate.py --help

usage: step1_annotate.py [-h] [--corpus_file CORPUS_FILE]
                         [--output_file OUTPUT_FILE] [--n N]
                         [--randomize RANDOMIZE]

optional arguments:
  -h, --help            show this help message and exit
  --corpus_file CORPUS_FILE
                        Specify the JSON file to read from. Must contain a
                        "doi", "title", and "abstract" fields in each
                        document. Default is only the 162 abstracts used in
                        the publication ("raw_data.json"). In practice, more
                        abstracts should be used.
  --output_file OUTPUT_FILE
                        Specify the output file to dump json annotations into.
  --n N                 Specify the number of documents to annotate.
  --randomize RANDOMIZE
                        If True, gets a random aggregation of n samples from
                        the dataset.

```


### Training and prediction with LLM (GPT-3 only)
```text
$: python step2_train_predict.py --help

usage: step2_train_predict.py [-h] [--openai_api_key OPENAI_API_KEY]
                              [--schema_type {eng,engextra,json}]
                              [--training_json TRAINING_JSON]
                              [--training_jsonl_output TRAINING_JSONL_OUTPUT]
                              [--training_n_epochs TRAINING_N_EPOCHS]
                              [--inference_model_name INFERENCE_MODEL_NAME]
                              [--inference_json INFERENCE_JSON]
                              [--inference_json_raw_output INFERENCE_JSON_RAW_OUTPUT]
                              [--inference_json_final_output INFERENCE_JSON_FINAL_OUTPUT]
                              [--inference_halt_on_error {True,False}]
                              [--inference_save_every_n INFERENCE_SAVE_EVERY_N]
                              {train,predict}

positional arguments:
  {train,predict}       Specify either "train" or "predict".

optional arguments:
  -h, --help            show this help message and exit
  --openai_api_key OPENAI_API_KEY
                        Your OpenAI API key. If not specified, will look for
                        an environment variable OPENAI_API_KEY.
  --schema_type {eng,engextra,json}
                        The type of NERRE schema; choose between eng,
                        engextra, and json. Default is eng. Only used if
                        op_type is train.
  --training_json TRAINING_JSON
                        If training, specify the name of the training JSON
                        file. Should NOT be a JSONL, as an OpenAI-compatible
                        JSONL will be automatically created.
  --training_jsonl_output TRAINING_JSONL_OUTPUT
                        If training, specify the name for the OpenAI-
                        compatible JSONL to be written. Default is
                        automatically written to a timestamped file in the
                        data directory.
  --training_n_epochs TRAINING_N_EPOCHS
                        The number of epochs to train for; this arg is passed
                        to openai cli. For more resolution in training, cancel
                        the training operation and call the openai API
                        directly.
  --inference_model_name INFERENCE_MODEL_NAME
                        Name of the trained model to use if op_type is
                        'predict'
  --inference_json INFERENCE_JSON
                        If predicting, specify the name of the raw inferred
                        JSON file. This file will contain the raw sentence
                        strings returned by GPT. Should NOT be a JSONL, as
                        JSONL will automatically be saved as well. Default is
                        the test set used in the publication, but this can be
                        used to predict on many thousands of samples.
  --inference_json_raw_output INFERENCE_JSON_RAW_OUTPUT
                        If predicting, specify the name for the JSONL file you
                        would like to save the raw predictions to. Default is
                        automatically written to a timestamped file in the
                        data directory.
  --inference_json_final_output INFERENCE_JSON_FINAL_OUTPUT
                        If predicting, specify the name for the decoded (non
                        raw, structured) entries to be saved to. Default is
                        automatically written to a timestamped file in the
                        data directory.
  --inference_halt_on_error {True,False}
                        If predicting, specify whether to halt on an error.
                        Default is True.
  --inference_save_every_n INFERENCE_SAVE_EVERY_N
                        If predicting, specify how often to save the raw
                        predictions to the JSONL file in case of a midstream
                        interruption. Default is 100.

```

### Score predicted entries
```text
$: python step3_score.py --help

usage: step3_score.py [-h] [--test_file TEST_FILE] [--pred_file PRED_FILE]
                      [--plot] [--loud]
                      {eng,engextra,json}

positional arguments:
  {eng,engextra,json}   The schema to use for similarity scores.

optional arguments:
  -h, --help            show this help message and exit
  --test_file TEST_FILE
                        The test file with correct answers.
  --pred_file PRED_FILE
                        The file predicted by LLM-NERRE. Default is using the
                        data from publication using the ENG schema.
  --plot                If flag is present, show a simple bar chart of
                        results.
  --loud                If true, show a summary of each evaluated sentence w/
                        FP and FNs.

```