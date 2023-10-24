import openai
import tqdm
import datetime
import json
import os
from openai.error import RateLimitError
import warnings
import traceback
import sys
import time

halt_on_error = False

experiments_dir = "experiments_mof"


# fold 0: "ddavinci:ft-matscholar:mof-rerun4-2023-07-16-04-25-53"
# fold 1: "davinci:ft-matscholar:mof-rerun3-2023-07-15-12-41-09"
# fold 2: "davinci:ft-matscholar:mof-rerun3-2023-07-15-14-03-22"
# fold 3: "davinci:ft-matscholar:mof-rerun3-2023-07-15-15-28-32"
# fold 4: "davinci:ft-matscholar:mof-rerun3-2023-07-15-16-49-42"


STOP_TOKEN = "\n\nEND\n\n"
def read_jsonl(filename):
    """
    Read a jsonlines file into a list of dictionaries.

    Args:
        filename (str): Path to input file.

    Returns:
        ([dict]): List of dictionaries.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        samples = []
        for l in lines:
            samples.append(json.loads(l))
        return samples


def dump_jsonl(d, filename):
    """
    Dump a list of dictionaries to file as jsonlines.

    Args:
        d ([dict]): List of dictionaries.
        filename (str): Path to output file.

    Returns:
        None
    """
    with open(filename, "w") as f:
        for entry in d:
            f.write(json.dumps(entry) + "\n")



if __name__ == "__main__":
    openai.api_key = "api_key_here"

    # for run in range(5):
    for run in range(5):
        model = f"davinci:ft-matscholar:mof-rerun4-2023-07-16-04-25-53"
        if run == 1:
            model = f"davinci:ft-matscholar:mof-rerun4-2023-07-16-05-59-12"
        elif run == 2:
            model = f"davinci:ft-matscholar:mof-rerun4-2023-07-16-07-19-45"
        elif run == 3:
            model = f"davinci:ft-matscholar:mof-rerun4-2023-07-16-08-29-52"
        elif run == 4:
            model = f"davinci:ft-matscholar:mof-rerun4-2023-07-16-10-10-29"

        print(f"Running fold {run} with model {model}")

        data_inference = read_jsonl(os.path.join(experiments_dir, f"fold_{run}", "val.jsonl"))
        gpt3_predictions = []
        jsonl_data = []
        for d in tqdm.tqdm(data_inference, desc="Texts processed"):
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            prompt = d["prompt"]
            completion = d["completion"]

            has_response = False
            while not has_response:
                try:
                    response = openai.Completion.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=512,
                        n=1,
                        # top_p=1,
                        temperature=0,
                        stop=[STOP_TOKEN],
                        logprobs=5
                    ).choices[0]
                    has_response = True
                except RateLimitError:
                    warnings.warn("Ran into rate limit error, sleeping for 60 seconds and dumping midstream...")
                    # dumpfn(gpt3_predictions, os.path.join(DATADIR, f"midstream_ratelimit_{dt}.json"))
                    time.sleep(60)
                    print("Resuming...")
                    continue
                except BaseException as BE:
                    if halt_on_error:
                        raise BE
                    else:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        warnings.warn(f"Ran into external error: {BE}")
                        traceback.print_exception(exc_type, exc_value,
                                                  exc_traceback,
                                                  limit=2,
                                                  file=sys.stdout)
                        print("Resuming...")
                        break

            llm_completion = response.text if has_response else None
            d["llm_completion"] = llm_completion.replace(STOP_TOKEN, "").strip()
            jsonl_data.append(d)

        fname = f"mof_results/run_{run}.jsonl"
        dump_jsonl(jsonl_data, fname)
        print(f"dumped fold {run} to {fname}")

