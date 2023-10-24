from monty.serialization import loadfn,  dumpfn
import warnings
import json
import pprint

#todo: Note seq2rel results are uncased! So in evaluation you must uncase the validation as well
# todo: in order to get an actual fair comparison


if __name__ == "__main__":
    d = loadfn("seq2rel_testset_predictions.json")
    d_val = loadfn("/Users/ardunn/alex/lbl/projects/matscholar/ardunn_text_experiments/doping_gpt3/gpt3_per_sentence/data/dopingv7_VALIDATION.json")
    dois_ordered = [e["doi"] for e in d_val]

    DOPANT = "@DOPANT@"
    BASEMAT = "@BASEMAT@"
    RELATION = "@DBR@"
    UNKNOWN = "unknown"
    SEPCHAR = "[SEP]"
    SPECIAL = "@"


    entries_amended = {}

    for entry in d:
        entry_amended = {
            "doi": entry["doi"],
            "text": "",
            "doping_sentences": [],

        }

        for i, prediction in enumerate(entry["sentences_incl_title"]):

            # For some reason, the first sentence is always fucked up
            if i == 0:
                continue


            sjson = {
                "dopants": {},
                "basemats": {},
                "dopants2basemats": {},
                "results": [],
                "doping_modifiers": []
            }


            entry_amended["text"] += prediction["sent_text"]
            dopant_counter = 0
            basemat_counter = 0

            relations = prediction["prediction"].split(RELATION)

            for r in relations:
                r = r.strip().replace(SEPCHAR, "")

                n_dopants = r.count(DOPANT)
                n_basemats = r.count(BASEMAT)

                if UNKNOWN in r:
                    pass

                elif n_dopants == 1 and n_basemats == 1:

                    dopant_str, bmat_str = r.split(DOPANT)
                    bmat_str = bmat_str.replace(BASEMAT, "").strip()
                    dopant_str = dopant_str.strip()
                    increase_basemat = False
                    increase_dopant = False

                    did = f"d{dopant_counter}"
                    bid = f"b{basemat_counter}"

                    if bmat_str not in sjson["basemats"].values():
                        sjson["basemats"][f"b{basemat_counter}"] = bmat_str
                        increase_basemat = True
                    else:
                        bid = {v: k for k, v in sjson["basemats"].items()}[bmat_str]

                    if dopant_str not in sjson["dopants"].values():
                        sjson["dopants"][f"d{dopant_counter}"] = dopant_str
                        increase_dopant = True
                    else:
                        did = {v: k for k, v in sjson["dopants"].items()}[dopant_str]


                    if did in sjson["dopants2basemats"]:
                        if bid not in sjson["dopants2basemats"][did]:
                            sjson["dopants2basemats"][did].append(bid)
                    else:
                        sjson["dopants2basemats"][did] = [bid]

                    if increase_dopant:
                        dopant_counter += 1
                    if increase_basemat:
                        basemat_counter += 1

                elif n_dopants == 1 and n_basemats == 0:
                    dopant_str = r.replace(DOPANT, "")
                    if dopant_str not in sjson["dopants"].values():
                        sjson["dopants"][f"d{dopant_counter}"] = dopant_str
                        dopant_counter += 1
                elif n_basemats == 1 and n_dopants == 0:
                    bmat_str = r.replace(BASEMAT, "")
                    if bmat_str not in sjson["basemats"].values():
                        sjson["basemats"][f"b{basemat_counter}"] = bmat_str
                        basemat_counter += 1
                else:
                    # Not parsable
                    warnings.warn(f"RELATION NOT PARSABLE: '{r}'")


            sjson_formatted = {
                "sentence_text": prediction["sent_text"],
                "entity_graph_raw": sjson,
                "llm_completion": json.dumps(sjson)
            }

            entry_amended["doping_sentences"].append(sjson_formatted)
        entries_amended[entry["doi"]] = entry_amended


    output = [entries_amended[doi] for doi in dois_ordered]

    for i, val_entry in enumerate(d_val):
        assert(len(val_entry["doping_sentences"]) == len(output[i]["doping_sentences"]))

    # pprint.pprint(output)
    dumpfn(output, "seq2rel_doping_evaluation_all_lowercase_correct_fields.json")






