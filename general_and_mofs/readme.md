## Preparing train/val splits:
You can split the data into train/val splits using the following code and then run them through the GPT-3 API with:
```
python prepare_test_train_data.py --dataset_path "data/general_materials_information_annotations.jsonl" --experiment_dir <experiment_dir>
bash eval.sh <experiment_dir>
```

## Results directories
For your convenience, we have included the output of the fine-tuned model for each of the 5 validation sets in the following directories:
* `data/general_results`
* `data/mof_results`

The `gpt3_completion` values are what the model output and the `completion` values are the human annotations.

## Manually evaluated examples:
We manually evaluated 50 validation set examples from experiments_general/fold_0/val.jsonl at random. These examples can be found in `data/general_manual_evaluation_set.jsonl`.