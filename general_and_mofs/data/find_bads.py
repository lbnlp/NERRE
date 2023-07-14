import json
import numpy as np

from pprint import pprint

def read_jsonl(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        samples = []
        for l in lines:
            samples.append(json.loads(l))
        return samples
    


# samples = read_jsonl("mof_results/run_0.jsonl")

# samples = read_jsonl("experiments_mof/fold_0/val.jsonl")


# samples = read_jsonl("experiments_general/fold_0/train.jsonl")


samples = read_jsonl("mof_annotations.jsonl")


# bads = []
# actually_wrong = []
# for i,sample in enumerate(samples):

#     print(i)
#     prompt = sample["prompt"]


#     c = json.loads(sample["completion"].split("\n\nEND\n\n")[0])

#     # print("C is ", c)

#     if c:
#         do_print = False

#         for completion in c:

#             for k, v in completion.items():
#                 # if k in ["name_of_mof", "mof_formula"]:
#                 if isinstance(v, list):
#                     # print("Islist", k)
                                
#                     for vv in v:
#                         if vv.lower() not in prompt.lower():
#                             print(i, k)
#                             bads.append(i)
#                             do_print = True
#                 else:
#                     # print("Is value", k)
#                     if v.lower() not in prompt.lower():
#                         print(i, k)
#                         bads.append(i)
#                         do_print = True
#             if do_print:

#                 pprint(prompt)
#                 pprint(c)

#                 # if not(completion["name_of_mof"] and completion["mof_formula"]):
    
#                 #     # actually_wrong.append(i)

#                 # yn = input("Entry is totall wrong? (y/n)")

#                 # if yn.strip().lower() == "y":
#                 #     actually_wrong.append(i)

# unique_bads = np.unique(bads)

# print(unique_bads)
# print(len(unique_bads),"of", len(samples))


# print([f"line entry {aw + 1}" for aw in actually_wrong])
# {"prompt": "A time-dependent density functional theory study on the effect of electronic excited-state hydrogen bonding on luminescent MOFs\nWe have investigated a new silver-based luminescent metal\u2013organic framework (MOF) using density functional theory and time-dependent density functional theory methods. We theoretically demonstrated that the H\u22efO hydrogen bond is strengthened and the Ag\u2013O coordination bond is shortened significantly due to strengthening of the hydrogen bond in the S1 state. When the hydrogen bond is formed, the mechanism of luminescence changes from a ligand-to-metal charge transfer (LMCT) coupled with intraligand charge transfer (LLCT) to LMCT, and the luminescence is found to be enhanced.\n\n###\n\n", "completion": " [{\"name_of_mof\": \"MIL-96\", \"mof_formula\": \"Aluminum 1,3,5-benzenetricarboxylate\", \"mof_description\": [\"\"], \"guest_species\": [\"\"], \"applications\": [\"separation of C5-hydrocarbons\", \"liquid-phase separation of the aliphatic C 5-diolefins, mono-olefins, and paraffins\"]}, {\"name_of_mof\": \"chabazite\", \"mof_formula\": \"\", \"mof_description\": [\"\"], \"guest_species\": [\"\"], \"applications\": [\"separation of C5-hydrocarbons\", \"liquid-phase separation of the aliphatic C 5-diolefins, mono-olefins, and paraffins\"]}, {\"name_of_mof\": \"\", \"mof_formula\": \"[Cu3(BTC)2] (BTC = benzene-1,3,5-tricarboxylate)\", \"mof_description\": [\"\"], \"guest_species\": [\"\"], \"applications\": [\"separation of C5-olefins from paraffins\", \"liquid-phase separation of the aliphatic C 5-diolefins, mono-olefins, and paraffins\"]}]\n\nEND\n\n"}
# {"prompt": "Sorption of methane, hydrogen and carbon dioxide on metal-organic framework, iron terephthalate (MOF-235)\nIron terephthalate, MOF-235, metal-organic framework synthesized hydrothermally and was used for gas adsorption. Resulting sample was characterized by X-ray diffraction (XRD), Brunauer\u2013Emmet\u2013Teller (BET) and FT-IR analysis. Adsorption properties of CH4, H2 and CO2 on MOF-235 were investigated by volumetric measurements. The absolute adsorption capacity was found in the order of CH4 \u226bH2 CO2. The high CH4 adsorption capacity of MOF-235 was attributed to the high pore volume and large number of open metal sites. The high selectivity for CH4 over CO2 (14.7) and H2 (8.3), suggests that MOF-235 is a potential adsorbent for the separation of CH4 from gas mixtures.\n\n###\n\n", "completion": " [{\"name_of_mof\": \"MOF-235\", \"mof_formula\": \"\", \"mof_description\": [\"Iron terephthalate\", \"metal-organic framework synthesized hydrothermally\"], \"guest_species\": [\"CH4\", \"H2\", \"CO2\"], \"applications\": [\"adsorption\", \"separation of CH4 from gas mixtures\"]}]\n\nEND\n\n"}


ixs = [ 11,  21,  25,  29,  38,  42,  43,  56,  76,  81,  85,  92,  95,  97,  98,  101,  102,  110
,  112,  115,  120,  122,  123,  126,  128,  129,  130,  133,  136,  138,  143,  149,  153,  158,  160,  162
,  163,  165,  172,  173,  175,  178,  188,  191,  194,  196,  198,  199,  211,  213,  219,  224,  226,  240
,  241,  242,  244,  249,  252,  255,  257,  262,  263,  264,  266,  270,  272,  280,  289,  294,  299,  301
,  308,  316,  322,  323,  324,  328,  329,  330,  331,  335,  336,  337,  338,  343,  344,  352,  355,  361
,  372,  373,  374,  375,  379,  384,  386,  392,  394,  396,  399,  401,  406,  416,  418,  419,  423,  428
,  429,  431,  437,  441,  445,  446,  455,  466,  480,  485,  487,  500,  501,  503]

wrong_ix = []

for i, ix in enumerate(ixs):
    sample = samples[ix]

    c = json.loads(sample["completion"].split("\n\nEND\n\n")[0])    
    p = sample["prompt"]

    print(f"Sample {i} of {len(ixs)}")
    pprint(p)
    pprint(c)

    yn = input("Entry is totally wrong? (y/n): ")

    if yn.strip().lower() == "y":
        wrong_ix.append(ix)

    print("WRONGIX: ", wrong_ix)


# WRONGIX:  [11, 21, 25, 29, 38, 43, 56, 76, 81, 85, 95, 97, 98, 101, 122, 126, 128, 129, 136, 143, 153, 158, 163, 173, 178, 219, 224, 257, 270, 280, 299, 301, 316, 324, 336, 352, 355, 372, 379, 392, 394, 396, 401, 406, 418, 423, 431, 441, 455, 480, 500, 503]
