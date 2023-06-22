# General and MOFs tasks

The code and data in this directory supports the General-JSON and MOF-JSON models presented in the publication. Although we cannot share the private API key used to fine-tune the specific models we use, we provide all training and validation data along with scripts for reproducing our work using your own fine tunes. These scripts can also be used for annotating and executing your own specific text extraction tasks. Typically, a fine-tune with the number of training examples we present (on the order of 100-1,000) will cost about $5-$10 with current OpenAI API pricing (dated Jun 23 2023). 

**Note: though we use OpenAI's GPT-3 as the LLM for our NERRE approach in this code, the OpenAI API calls can be replaced in principle with fine-tuning code for any LLM.**


## Annotating examples
### Annotation UI
See `annotation_ui.ipynb`. This annotation UI will allow you to annotate abstracts. If you already have a model to use for in-the-loop annotation, this UI will allow you to "pre-fill" annotations for increased annotation speed.

### For annotations used in the publication, see the `data` directory.

## Preparing train/val splits:
You can split the data into train/val splits using the following code and then run them through the GPT-3 API with:
```
# Prepare your train and test sets
python prepare_test_train_data.py --dataset_path "data/general_materials_information_annotations.jsonl" --experiment_dir <experiment_dir>
```

Replace the filename `data/general_materials_information_annotations.jsonl` with your own annotations jsonl created with the annotation UI to evaluate your own model.

To reproduce results in the study, use:
- General-JSON: `data/general_materials_information_annotations.jsonl`
- MOF-JSON: `data/mof_annotations.jsonl`

## Training and prediction
```
# Fine-tune a GPT-3 model
bash eval.sh <experiment_dir> <experiment_name>
```

After the fine tune has completed, you can predict using the openai API (thru python) on your own examples. See `$: openai --help` and/or the python OpenAI API documentation for more help.

Alternatively, to reproduce results from the study with your own model using the same data as the study, use the test sets given in the `data` directory as input to a fine-tuned model.

## Results

The `results.py` script scores results jsonl files previously predicted by five-fold cross validation. The output of your training and prediction should create files called `run_i.jsonl` where `i` is 0, 1, 2, 3, or 4. Each line entry should be a 2-key dictionary where `gpt3_completion` values are what the model output and the `completion` values are the human annotations.


For your convenience, we have included the output of the fine-tuned models presented in the publication in compatible format for each of the 5 validation sets in the following directories:
* `data/general_results`
* `data/mof_results`


To evaluate a set of results (e.g., those presented in the paper), simply run:

```bash
$: python results.py --results_dir <path to results directory> --task <name of task>
```

So to score the results of the General-JSON model presented in the paper, do:

```bash
$: python results.py --results_dir data/general_results --task general
```

The output will look like:

```
All Exact match accuracy average: 0.30322580645161284
Jaro-Winkler avg similarity: 0.9423860071801728
...
 'links': {'formula|||acronym': {'f1': 0.537026764823375,
                                 'precision': 0.6347410530645825,
                                 'recall': 0.4704334365325077},
           'formula|||applications': {'f1': 0.49482745608345385,
                                      'precision': 0.5131327294462099,
                                      'recall': 0.4834157326151445},
           'formula|||description': {'f1': 0.34640257881655445,
                                     'precision': 0.40007020092531603,
                                     'recall': 0.3079513754329325},
           'formula|||name': {'f1': 0.6124697854830494,
                              'precision': 0.7153064349174623,
                              'recall': 0.5373956591437371},
           'formula|||structure_or_phase': {'f1': 0.41710498183797995,
                                            'precision': 0.45103549042689794,
                                            'recall': 0.3887694587314113}}}
```

And for MOFs, do:

```bash
$: python results.py --results_dir data/mof_results --task mofs
```


The help string for the `results.py` script is:

```bash
$: python results.py --help
usage: results.py [-h] [--results_dir RESULTS_DIR] [--task {general,mof}]

optional arguments:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR
  --task {general,mof}  Which schema is being used
```


## Manually evaluated examples:
We manually evaluated 50 validation set examples from experiments_general/fold_0/val.jsonl at random. These examples can be found in `data/general_manual_evaluation_set.jsonl`.