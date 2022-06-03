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

if __name__ == "__main__":
    with open("SampleData/sample.json","r") as f:
        tmp = json.load(f)
    sequence = tmp["sequence"]
    target = tmp["target"]
    posList = tmp["posList"]

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
    print("Getting blacklist")
    sim = simGoogle(tokenizer, bertModel, myModel, device)
    sim.initialize(sequence,posList,target)
    res = sim.generateBlackList(2)
    blacklist = [[(t,1) for t in pair] for pair in res] # entity t can not choose option 1 [KEEP]
    sim.clear()
    print(blacklist)
    print(time.time()-start,"s")

    # Step 2: get final decision for each entity
    start = time.time()
    print("Getting final decisions")
    n = 21
    m = 2
    scores = np.random.randn(n,m) # Use Random Value as place holders
    optSolver = linearOpt(n, m, blacklist, scores, target="min")
    res = optSolver.solve()
    print(res)
    print(time.time()-start,"s")