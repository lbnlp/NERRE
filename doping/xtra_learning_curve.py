"""
For creating a learning curve graph.
"""
import pprint
import random
import json
import os
import datetime
import time

import openai
from monty.serialization import loadfn
import matplotlib
import matplotlib.pyplot as plt

# from step1_gpt3_doping_v2 import gpt3_infer, read_jsonl, gpt3_decode
from step3_score import evaluate


def determine_finetune_epochs(n):
    if n <= 16:
        return 3
    elif n <= 64:
        return 4
    else:
        return 6


if __name__ == "__main__":

    # BASEDIR = "/Users/ardunn/alex/lbl/projects/matscholar/ardunn_text_experiments/doping_gpt3/gpt3_per_sentence/data/"
    # CHECK_SLEEPTIME = 60
    # MAX_TOTAL_SLEEPTIME = 7200
    # N_LIST = [2, 4, 8, 16, 32, 64, 128, 256]
    # # N_LIST = [2, 4]
    #
    # api_key = "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"
    # openai.api_key = api_key
    # os.environ["OPENAI_API_KEY"] = api_key
    #
    # data_jsonl = read_jsonl(f"{BASEDIR}GPT3_training_file_162_samples_at_2022-08-29_21.59.04.jsonl")
    # print(f"Length of loaded jsonl is {len(data_jsonl)}")
    #
    # samples_to_infer = loadfn(f"{BASEDIR}dopingv7_VALIDATION.json")
    # samples_to_infer = [{"title": d["title"], "abstract": d["text"], "doi": d["doi"]} for d in samples_to_infer]
    # print("Loaded validation file.")
    #
    # for n in N_LIST:
    #     print(f"LOG: WORKING ON GPT3 LEARNING CURVE WITH N={n}")
    #     sample_jsonl = random.sample(data_jsonl, k=n)
    #
    #     output_filename = f"{BASEDIR}GPT3_training_file_learning_curve_n{n}_at_2022.09.30_XX-XX-XX.jsonl"
    #
    #     with open(output_filename, "w") as f:
    #         for s in sample_jsonl:
    #             j = json.dumps(s)
    #             f.write(j)
    #             f.write("\n")
    #
    #     print(f"Successfully wrote {n} samples to {output_filename}.")
    #
    #
    #     n_epochs = determine_finetune_epochs(n)
    #
    #     print("Waiting for finetune to finish, will probably take a while")
    #     wrapper = os.popen(
    #         f"openai api fine_tunes.create -t '{output_filename}' -m 'davinci' --n_epochs={n_epochs}"
    #     )
    #     log_output = wrapper.read().split("\n")
    #
    #     print(f"LOG: LOG OUTPUT FROM OPENAPI WAS: \n {pprint.pformat(log_output)}")
    #
    #     ft_name = None
    #     for l in log_output:
    #         if "ft-" in l:
    #             ft_name = l.split(" ")[-1].strip()
    #             break
    #
    #     if not ft_name:
    #         raise ValueError("No fine tune name!")
    #
    #     print(f"LOG: FOUND FINE-TUNE NAME: '{ft_name}'")
    #
    #
    #     logfile_name = f"/Users/ardunn/Downloads/GPT3_log_learningcurve__apilog_n{n}.txt"
    #     print(f"LOG: MONITORING LOGFILE {logfile_name} FOR SIGNS OF API COMPLETION.")
    #     folllow_cmd = f"openai api fine_tunes.follow -i {ft_name} | tee {logfile_name}"
    #     os.system(folllow_cmd)
    #
    #     current_sleeptime = 0
    #
    #     model_name = None
    #     while current_sleeptime < MAX_TOTAL_SLEEPTIME:
    #         do_break = False
    #
    #         with open(logfile_name,  "r") as f:
    #             for l in f.readlines():
    #                 if "Uploaded model:" in l:
    #                     model_name = l.split(" ")[-1].strip()
    #                     do_break = True
    #                     break
    #                 if "Stream interrupted" in l:
    #                     os.system("pkill -f openai")
    #                     os.remove(logfile_name)
    #                     print(f"LOG: MODEL STREAM GOT INTERRUPTED, RESTARTING IT")
    #                     os.system(folllow_cmd)
    #
    #         if do_break:
    #             print(f"LOG FOUND MODEL NAME '{model_name}' - BREAKING!")
    #             break
    #
    #         print(f"LOG: MODEL IS STILL NOT DONE AT {datetime.datetime.now().isoformat()}... WAITING AN ADDITIONAL {CHECK_SLEEPTIME} SECONDS.")
    #         time.sleep(CHECK_SLEEPTIME)
    #
    #         current_sleeptime += CHECK_SLEEPTIME
    #     else:
    #         raise ValueError(f"No model name found within {MAX_TOTAL_SLEEPTIME} seconds.")
    #
    #     # Infer the validation samples
    #     output_filename = output_filename.replace("training", "inference")
    #     dois_skipped, inferred_filename = gpt3_infer(
    #         data_inference=samples_to_infer,
    #         data_training_dois=[],
    #         model=model_name,
    #         save_every_n=50,
    #         halt_on_error=True,
    #         output_filename=output_filename
    #     )
    #
    #     try:
    #         # Decode the validation samples and store them
    #         output_filename = inferred_filename.replace("inference", "decoded")
    #         output_filename = gpt3_decode(inferred_filename=inferred_filename, output_filename=output_filename)
    #
    #     except:
    #         print("LOG: FAILED GPT3 DECODING FOR WHATEVER REASON!")
    #
    #
    #     print(f"LOG: COMPLETED RUN FOR n={n}! File stored at {output_filename}." + "\n"*5)

    font = {'family': 'Helvetica',
            # 'weight': 'bold',
            'size': 16
            }
    # print(matplotlib.font_manager.findfont("Century Schoolbook", rebuild_if_missing=True))
    # font_dirs = ['/Users/ardunn/Library/Fonts', ]
    # font_files = matplotlib.font_manager.findSystemFonts(fontpaths=font_dirs)
    # font_list = matplotlib.font_manager.createFontList(font_files)

    # f = matplotlib.font_manager.get_font("/Users/ardunn/Library/Fonts/CenturySchoolbookPro.ttf")
    #
    # print(f, type(f))
    #


    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({"text.usetex": False})

    # raise ValueError


    # Making learning curve graph
    DATA_DIR = "data/learning_curve/"
    recalls_per_n = []
    precisions_per_n = []
    f1s_per_n = []
    basemat_recalls_per_n = []
    basemat_precisions_per_n = []
    basemat_f1s_per_n = []
    dopant_recalls_per_n = []
    dopant_precisions_per_n = []
    dopant_f1s_per_n = []

    ns = []

    n_test_dict = {}

    for fname in os.listdir(DATA_DIR):
        if "GPT3_decoded_file_learning_curve" in fname:
            test = loadfn(f"{DATA_DIR}{fname}")
            n = int([word for word in fname.split("_") if word.startswith("n")][0].replace("n", ""))
            n_test_dict[n] = test


    final_test = loadfn(f"data/gpt3/inference_decoded_eng.json")
    gold = loadfn(f"data/test.json")
    n_test_dict[413] = final_test

    n_test_ordered = sorted(n_test_dict.items(), key=lambda x: x[0])

    for n, test in n_test_ordered:
        print(f"Working on n={n}")
        (scores_computed,
            ent_categories,
            sequences_distances,
            sequences_correct,
            sequences_parsable,
            sequences_total,
            support,
            parsability_valid) = evaluate(gold, test)
        lt = scores_computed["link triplets"]
        precisions_per_n.append(lt["precision"])
        recalls_per_n.append(lt["recall"])
        f1s_per_n.append(lt["f1"])

        bt = scores_computed["basemats"]
        basemat_recalls_per_n.append(bt["recall"])
        basemat_precisions_per_n.append(bt["precision"])
        basemat_f1s_per_n.append(bt["f1"])

        dt = scores_computed["dopants"]
        dopant_recalls_per_n.append(dt["recall"])
        dopant_precisions_per_n.append(dt["precision"])
        dopant_f1s_per_n.append(dt["f1"])

        ns.append(n)

    fig, axes = plt.subplots(3, 1, sharex="all")
    scores_organized = {
        "links": [recalls_per_n, precisions_per_n, f1s_per_n],
        "basemats": [basemat_recalls_per_n, basemat_precisions_per_n, basemat_f1s_per_n],
        "dopants": [dopant_recalls_per_n, dopant_precisions_per_n, dopant_f1s_per_n]
    }

    i = 0

    plotting_order = ("recall", "precision", r"F1")
    for etype, scores in scores_organized.items():
        ax = axes[i]
        for j, c in enumerate(("green", "blue", "red")):
            ax.plot([1] + ns, [0] + scores[j], color=c, marker="o", label=plotting_order[j])
        i += 1
        ax.set_title(etype.capitalize() if etype != "basemats" else "Hosts")
        ax.set_xscale("log")
        ax.set_ylim([-0.1, 1.0])

    axes[0].legend()
    axes[1].set_ylabel("Score")
    axes[2].set_xlabel("Training set size (sequence samples)")
    # plt.tight_layout()
    plt.show()

    fig.savefig("learning_curve.svg")
    fig.savefig("learning_curve.png")