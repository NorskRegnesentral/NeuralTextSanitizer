# Test the following pipeline:
# Text, Target, posList -> blacklist
# blacklist + scores [Here I use random number as placeholders] -> decision for each entity
# Here decision 1 = Keep, decision 0 = Mask [Assumption used in generating constraints]

import torch
import time
import json
import numpy as np

from linearProgramming import linearOpt # Should include this first, otherwise will raise Segmentation Fault
from transformers import BertTokenizer, BertModel
from models import MyModel5
from emulate import simGoogle

from transformers import BertTokenizerFast, BertForMaskedLM
from mlmbert import mlmbert

from bert_model import NERModel
from transformers import RobertaTokenizerFast
from detect import detect_pii

from mask_classifier import MaskClassifier


if __name__ == "__main__":
    # Test Data for Step2 + Step3
    # with open("SampleData/sample.json","r") as f:
    #     tmp = json.load(f)
    # sequence = tmp["sequence"]
    # target = tmp["target"]
    # posList = tmp["posList"]
    # posList = [tuple(p) for p in posList] # Ensure posList List of tuple -> otherwise unhashable
    # sequence = "Bodewin Claus Eduard Keitel (German pronunciation: [ˈkaɪ̯tl̩]; 1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office."
    # target = "bodewin keitel"
    # posList = [(0, 27), (29, 35), (51, 61), (63, 67), (70, 74), (82, 88), (89, 96), (104, 116), (131, 135), (143, 164)]  # annotation text_span

    # Test Data for the whole pipeline
    with open("SampleData/sample2.json","r") as f:
        tmp = json.load(f)

    sequence = tmp["text"]
    target = tmp["target"]

    # Step 0: detect entities
    bert = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(bert)
    model = NERModel(model=bert, num_labels=17)
    model.load_state_dict(torch.load("SampleData/3roberta_model.pt", map_location=torch.device('cpu')))

    try:
        # If posList have annotations
        posList = tmp["annotations"]
        posList = [tuple(p) for p in posList] # Ensure posList List of tuple -> otherwise unhashable
    except KeyError:
        # Privacy enhanced NER to detect all PII
        posList = detect_pii(tmp, model, tokenizer) # List of {target: [Spans detected with labels]}
        posList = list(posList.values())[0] # get only the values

        tagList = [i[1] for i in posList]
        posList = [i[0] for i in posList] # text spans

        # posList = sorted(posList, key=lambda x: x[0])
        posList = list(set(posList))
        posList = sorted(posList, key=lambda x: x[0])

        spans = {pos:tag for pos,tag in zip(posList,tagList)}

    # Step 1: get blacklist
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    print("Getting blacklist1 [Language Model]")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    mlmbert_model = mlmbert(device, tokenizer, model, thres = -4, N=1)
    blacklist1, semantic_loss = mlmbert_model.get_blacklist_and_semantic_loss(sequence, posList)
    blacklist1 = [[(t,1) for t in pair] for pair in blacklist1]
    print(blacklist1)
    print(time.time() - start, "s")

    start = time.time()
    print("Loading Model for blacklist2")
    bertModel = BertModel.from_pretrained("bert-large-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    myModel = MyModel5(device, drop=True, pooling="max").to(device)
    myModel.load_state_dict(torch.load('SampleData/model5max.pt', map_location=device))
    print(time.time() - start, "s")

    start = time.time()
    print("Getting blacklist2 [Web Query Based Models]")
    sim = simGoogle(tokenizer, bertModel, myModel, device)
    sim.initialize(sequence, posList, target)
    res = sim.generateBlackList(2)
    blacklist2 = [[(t, 1) for t in pair] for pair in res]  # entity t can not choose option 1 [KEEP]
    sim.clear()
    print(blacklist2)
    print(time.time() - start, "s")

    start = time.time()
    print("Getting blacklist3 [mask classifier]")
    classifier = MaskClassifier.load("SampleData/mask_classifier.dill", device)
    res = classifier.predict_from_labelled_text(sequence, spans, device=device)
    blacklist3 = [[(i,1)] for i,pos in enumerate(posList) if res[pos]>0.5]
    print(blacklist3)
    print(time.time() - start, "s")


    blacklist = blacklist1 + blacklist2 + blacklist3

    # Step 2: get final decision for each entity
    start = time.time()
    print("Getting final decisions")
    n = len(semantic_loss)
    m = 2
    scores = np.array([[sem_loss,0] for sem_loss in semantic_loss]) # should input np.array
    optSolver = linearOpt(n, m, blacklist, scores)
    res = optSolver.solve()
    print(res)
    print(time.time()-start,"s")

    # Save final decisions
    decisions = {
        "opt_decision": [posList[i] for i,d in enumerate(res) if d==1],        # 1 for KEEP
        "ner_decision": posList,                                               # All the entities detected by NER
        "b1": list(set(posList[t[0]] for pair in blacklist1 for t in pair)),   # blacklist1 LM
        "b2": list(set(posList[t[0]] for pair in blacklist2 for t in pair)),   # blacklist2 Web Query Based Models
        "b3": list(set(posList[t[0]] for pair in blacklist3 for t in pair))    # blacklist3 mask_classifier
    }
    # sort the list
    for v in decisions.values():
        v.sort()
    final_decisions = {target:decisions}

    import json
    json.dump(final_decisions, open("final_decision.json","w"))

    # Some tests

    # Print Blacklist
    # print("Blacklist1")
    # for pair in blacklist1:
    #     print(" AND ".join(sequence[posList[p[0]][0]:posList[p[0]][1]] for p in pair))
    # print("Blacklist2")
    # for pair in blacklist2:
    #     print(" AND ".join(sequence[posList[p[0]][0]:posList[p[0]][1]] for p in pair))

    # Print Masking Decisions
    entities = [sequence[pos[0]:pos[1]] for pos in posList]
    print("Final Decision")
    for entity, decision, sem_loss in zip(entities, res, semantic_loss):
        if decision == 1:
            print(entity, '\t', 'Keep', '\t', 0)
        else:
            print(entity, '\t', 'Mask', '\t', sem_loss)
