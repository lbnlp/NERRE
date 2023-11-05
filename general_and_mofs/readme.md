# General and MOFs tasks

The code and data in this directory supports the General and MOF models presented in the publication. Although we cannot share the private API key used to fine-tune the specific GPT-3 models we use, we provide all training and validation data along with scripts for reproducing our work using your own fine tunes. These scripts can also be used for annotating and executing your own specific text extraction tasks. Typically, a fine-tune with the number of training examples we present (on the order of 100-1,000) will cost about $5-$10 with current OpenAI API pricing (dated Jun 23 2023). 

**Note: though we use OpenAI's GPT-3 as the LLM for our NERRE approach in this code, the OpenAI API calls can be replaced in principle with fine-tuning code for any LLM.**

Alternatively, see [nerre-llama](https://github.com/lbnlp/nerre-llama) for the Llama-2 weights and code.
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

## Training and prediction (with GPT-3)
```
# Fine-tune a GPT-3 model
bash eval.sh <experiment_dir> <experiment_name>
```

After the fine tune has completed, you can predict using the openai API (thru python) or Llama-2 on your own examples. For GPT-3, see `$: openai --help` and/or the python OpenAI API documentation for more help. For Llama-2, see the [nerre-llama](https://github.com/lbnlp/nerre-llama) repo.

Alternatively, to reproduce results from the study with your own model using the same data as the study, use the training and cross-validation sets given in the `data/experiments*` directories as input to a fine-tuned model.

## Results

The `results.py` script scores results jsonl files previously predicted by five-fold cross validation. The output of your training and prediction should create files called `run_i.jsonl` where `i` is 0, 1, 2, 3, or 4. Each line entry should be a 2-key dictionary where `llm_completion` values are what the model output and the `completion` values are the human annotations.


For your convenience, we have included the output of the fine-tuned models presented in the publication in compatible format for each of the 5 validation sets in the directories `./data/predictions*` with a folder for each model (gpt3, llama2) and task (general, mof).


To evaluate a set of results (e.g., those presented in the paper), simply run:

```bash
$: python results.py --results_dir <path to results directory> --task <name of task>
```

So to score the results of the General-JSON model presented in the paper, do:

```bash
$: python results.py --results_dir data/predictions_general_gpt3 --task general
```

The output will look like:

```
Summary: 
--------------------
Support was  {'ents': {'acronym': 41,
          'applications': 305,
          'description': 233,
          'formula': 353,
          'name': 175,
          'structure_or_phase': 276},
 'links_ents': {'formula|||acronym': 8,
                'formula|||applications': 341,
                'formula|||description': 202,
                'formula|||name': 60,
                'formula|||structure_or_phase': 296},
 'links_words': {'formula|||acronym': 8,
                 'formula|||applications': 577,
                 'formula|||description': 281,
                 'formula|||name': 108,
                 'formula|||structure_or_phase': 485},
 'words': {'acronym': 42,
           'applications': 513,
           'description': 314,
           'formula': 480,
           'name': 269,
           'structure_or_phase': 421}}
All Exact match accuracy average: 0.28387096774193543
Jaro-Winkler avg similarity: 0.9173708433537608
Parsable percentage 1.0
{'ents': {'acronym': {'f1': 0.5117944147355913,
                      'precision': 0.687012987012987,
                      'recall': 0.5161904761904761},
          'applications': {'f1': 0.6958042123522645,
                           'precision': 0.7224432773109244,
                           'recall': 0.6767846217393118},
          'description': {'f1': 0.47623485187055065,
                          'precision': 0.4884546158889088,
                          'recall': 0.4801168100311727},
          'formula': {'f1': 0.6851882626794457,
                      'precision': 0.6984549207242706,
                      'recall': 0.6814060140440954},
          'name': {'f1': 0.5443782209136538,
                   'precision': 0.6189565867587485,
                   'recall': 0.5370883190883191},
          'structure_or_phase': {'f1': 0.6002675463753386,
                                 'precision': 0.6535667977215965,
                                 'recall': 0.56337557371129}},
 'links': {'formula|||acronym': {'f1': 0.28571428571428575,
                                 'precision': 0.3333333333333333,
                                 'recall': 0.25},
           'formula|||applications': {'f1': 0.5164802639857736,
                                      'precision': 0.5448693599413548,
                                      'recall': 0.49596815743979095},
           'formula|||description': {'f1': 0.34034851290758406,
                                     'precision': 0.3465689865689866,
                                     'recall': 0.34235957588890564},
           'formula|||name': {'f1': 0.36734627225048005,
                              'precision': 0.46152743652743655,
                              'recall': 0.4173649641344741},
           'formula|||structure_or_phase': {'f1': 0.47018460647519794,
                                            'precision': 0.5513509112004324,
                                            'recall': 0.4315282151935584}}}
```


*note: for some folds of some tasks, the gold annotations do not have any links for some link types (e.g., name-formula for MOF fold 0); precision and recall for these folds is not included in the averages*. 

And for MOFs, do:

```bash
$: python results.py --results_dir data/predictions_mof_gpt3 --task mof
```

And to score the results from Llama-2 benchmark, do:

```bash
$: python results.py --results_dir data/predictions_general_llama2 --task general
$: python results.py --results_dir data/predictions_mof_llama2 --task mof
```

The help string for the `results.py` script is:

```bash
$: python results.py --help
usage: results.py [-h] [--results_dir RESULTS_DIR] [--task {general,mof}] [--loud]

optional arguments:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR
  --task {general,mof}  Which schema is being used
  --loud                If true, show a summary of each evaluated sentence w/ FP and FNs.
```


## Manually evaluated examples:
We manually evaluated 50 validation set examples from experiments_general/fold_0/val.jsonl at random. These examples can be found in `data/general_manual_evaluation_set.jsonl`.
