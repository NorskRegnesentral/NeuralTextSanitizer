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

if __name__ == "__main__":
    with open("SampleData/sample.json","r") as f:
        tmp = json.load(f)
    sequence = tmp["sequence"]
    target = tmp["target"]
    posList = tmp["posList"]
    posList = [tuple(p) for p in posList] # Ensure posList List of tuple -> otherwise unhashable

    # Step 1: get blacklist
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    print("Loading Model")
    bertModel = BertModel.from_pretrained("bert-large-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    myModel = MyModel5(device, drop=True, pooling="max").to(device)
    myModel.load_state_dict(torch.load('SampleData/model5max.pt',map_location=device))
    print(time.time()-start,"s")

    start = time.time()
    print("Getting blacklist1")
    sim = simGoogle(tokenizer, bertModel, myModel, device)
    sim.initialize(sequence,posList,target)
    res = sim.generateBlackList(2)
    blacklist1 = [[(t,1) for t in pair] for pair in res] # entity t can not choose option 1 [KEEP]
    sim.clear()
    print(blacklist1)
    print(time.time()-start,"s")

    start = time.time()
    print("Getting blacklist2 [Using BertForMaskedLM]")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    mlmbert_model = mlmbert(device, tokenizer, model, thres = -20, N=3)
    blacklist2, semantic_loss = mlmbert_model.get_blacklist_and_semantic_loss(sequence, posList)
    blacklist2 = [[(t,1) for t in pair] for pair in blacklist2]
    print(blacklist2)
    print(time.time() - start, "s")
    blacklist = blacklist1 + blacklist2

    # Step 2: get final decision for each entity
    start = time.time()
    print("Getting final decisions")
    n = len(semantic_loss) # 21
    m = 2
    # scores = np.random.randn(n,m) # Use Random Value as place holders
    scores = np.array([[sem_loss,0] for sem_loss in semantic_loss]) # should input np.array
    optSolver = linearOpt(n, m, blacklist, scores)
    res = optSolver.solve()
    print(res)
    print(time.time()-start,"s")

    # Print Masking Decisions
    entities = [tmp["sequence"][pos[0]:pos[1]] for pos in posList]
    for entity, decision, sem_loss in zip(entities, res, semantic_loss):
        if decision == 1:
            print(entity, '\t', 'Keep', '\t', 0)
        else:
            print(entity, '\t', 'Mask', '\t', sem_loss)
