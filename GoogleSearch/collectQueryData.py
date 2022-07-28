import pickle
import json
import numpy as np
import pandas as pd
from searchGoogle import search, saveRes
from itertools import combinations

def getEntityId(docid, res1Cleaned, res2Cleaned, quotes=True):
    """
    Given document id (docid) in the res1Cleaned and res2Cleaned,
    Get the corresponding query
    Quotes: if put the entities inside the quotes
    """
    selectedIndex = (res2Cleaned["docId"] == docid)

    if quotes:
        target = '"' + res1Cleaned.loc[docid, "target"] + '"'
    else:
        target = res1Cleaned.loc[docid, "target"]

    entityIdList = list(res2Cleaned.loc[selectedIndex]["entityId"].values)
    return target, entityIdList

def getQueries(entityIdList,entityDict,n,sep="AND",quotes=True):
    """
    Get queries composed of n elements in the entityList after the header.
    We can put our target in the header
    """
    sep = " "+sep+" "
    queryList = []
    entityIdCombList = list(combinations(entityIdList, n))
    for comb in entityIdCombList:
        if quotes:
            queryList.append(sep.join([ '"'+entityDict[i]["entity"]+'"' for i in comb]))
        else:
            queryList.append(sep.join([ entityDict[i]["entity"] for i in comb]))
    return entityIdCombList,queryList

def testCombination(entityIdCandidates, target, targetSet, res, queryDict, n, sep="AND", maxStep=100):
    """
    testing n-combinations from entityCandidates
    decision: -1 helping query, 0 not Dangerous, 1 Dangerous, None Not Sure -> Next Round
    Will Update targetSet, res
    """
    entityIdCombList, queryList = getQueries(entityIdCandidates, entityDict, n, sep=sep)

    if (search.counter + len(queryList)) >= maxStep:
        print("Total Budget Reached. Shutting Down...")
        print("Unfinished: Target = ", target, "n = ", n)
        return
    else:
        dangerQueries = []  # entityId Combinations
        for i, q in enumerate(queryList):
            search_results = search(q, queryDict, num_results=50, proxy=proxies, verify=verify)
            saveRes(search_results, res)
            res[q]["entityId"] = entityIdCombList[i]  # Save corresponding entityId, tuple

            urls = res[q]["url"]

            if len(res[q]["Estimated Matches"]) == 0:
                hitCount = 0
            else:
                hitCount = res[q]["Estimated Matches"][-1]  # The latter, the more accurate

            if len(set(urls).intersection(targetSet)) > 0:
                res[q]["decision"] = 1  # 1 for Dangerous, 0 not Dangerous, -1 helping query
                # dangerQueries.append(q)
                dangerQueries.append(entityIdCombList[i])

            else:
                if len(urls) == 0:
                    res[q]["decision"] = 0
                    # Keep it to the next round. -> don't add to danger queries
                else:
                    if hitCount == 0:
                        hitCount = len(urls)  # if len(urls)!=0 and hitCount==0

                    # Possibly dangerous -> calculate the P(t | c)
                    q1 = target + " " + sep + " " + q
                    search_results = search(q1, queryDict, num_results=50, proxy=proxies, verify=verify)
                    saveRes(search_results, res)
                    # == Since we don't have entity_id for target, we don't save its entity_id in the result ==

                    urls1 = res[q1]["url"]

                    if len(res[q1]["Estimated Matches"]) == 0:
                        hitCount1 = 0
                    else:
                        hitCount1 = res[q1]["Estimated Matches"][-1]

                    targetSet = targetSet.union(set(urls1))

                    res[q1]["decision"] = -1  # helping query
                    res[q]["proba"] = hitCount1 / hitCount

        # Check if any queries are dangerous again
        for i, q in enumerate(queryList):
            if len(set(res[q]["url"]).intersection(targetSet)) > 0 and entityIdCombList[i] not in dangerQueries:
                # Dangerous
                res[q]["decision"] = 1  # (2)
                # dangerQueries.append(q)
                dangerQueries.append(entityIdCombList[i])
        return dangerQueries, targetSet


def getSafeEntities(entityIdList, dangerQueries):
    def existsDanger(solution, entityPairs):
        sol = set(solution)
        for ep in entityPairs:
            if set(ep).issubset(sol):
                return True
        return False

    if len(dangerQueries) == 0:
        return entityIdList
    else:
        entityPairs = dangerQueries
        n = len(entityPairs[0])  # n-combination
        if n == 1:
            dangerQueries = [d[0] for d in dangerQueries]  # Flatten, [(2,)] -> [2]
            return [e for e in entityIdList if e not in dangerQueries]
        else:
            # If we don't have many dangerQueries, then we can use adjacency matrix
            res = entityIdList.copy()
            N = len(entityIdList)
            A = np.zeros((N, N))
            id2entity = {i: e for i, e in enumerate(entityIdList)}
            entity2id = {e: i for i, e in id2entity.items()}
            for ep in entityPairs:
                for comb in combinations(ep, 2):
                    id1 = entity2id[comb[0]]
                    id2 = entity2id[comb[1]]
                    A[id1][id2] += 1
                    A[id2][id1] += 1
            while existsDanger(res, entityPairs):
                # Remove nodes with highest degree
                removeId = np.argmax(np.sum(A, axis=0))
                res.remove(id2entity[removeId])  # remove
                A[removeId, :] *= 0
                A[:, removeId] *= 0
            return res


def saveFinalRes(finalRes, finalHistory, queryDict, filepath="outputs/"):
    with open(filepath + "finalRes.json", "w") as f:
        json.dump(finalRes, f)
    with open(filepath + "finalHistory.json", "w") as f:
        json.dump(finalHistory, f)
    with open(filepath + "queryDict.json", "w") as f:
        json.dump(queryDict, f)

if __name__=="__main__":
    search.counter = 0
    proxies = {
        "http": "...",
        "https": "...",
    }
    verify = '...'
    sep = "AND"

    # 1. Read Input Data
    res1Cleaned = pd.read_csv("inputs/train1wiki.csv")
    res2Cleaned = pd.read_csv("inputs/train2wiki.csv")
    with open("inputs/entityDict.pkl", "rb") as f:
        entityDict = pickle.load(f)
    with open("inputs/docIdx.pkl", "rb") as f:
        docIdx = pickle.load(f)

    # 2. Read Previous Resutls (if exists)
    try:
        with open("outputs/finalRes.json", "r") as f:
            finalRes = json.load(f)
        with open("outputs/finalHistory.json", "r") as f:
            finalHistory = json.load(f)
        with open("outputs/queryDict.json", "r") as f:
            queryDict = json.load(f)
        print("Previous Results Found.")
    except:
        finalRes = {}
        finalHistory = {}
        queryDict = {}
        print("Previous Results Not Found.")

    # 3. Collect Data
    N = 2           # Consider 1,...,N-entity pairs
    maxStep = 5000  # maximum search times, the real search times might exceeds this a little.

    for docId in docIdx:
        print(f"Searching for document {docId}, current query num {search.counter}")
        target, entityIdList = getEntityId(docId, res1Cleaned, res2Cleaned, quotes=True)

        res = {}

        # Target
        if finalRes.get(str(docId)) is None or finalRes[str(docId)].get(target) is None:
            print("Search target in document ", docId, "target", target)
            search_results = search(target, queryDict, num_results=50, proxy=proxies, verify=verify)
            saveRes(search_results, res)
            res[target]["decision"] = -2  # mark this query as Target
        else:
            res = finalRes[str(docId)]

        if finalHistory.get(str(docId)) is None:
            history = {"safeEntity": [], "targetSet": [], "dangerQueries": []}
            k = 1  # start from k-combination

            targetSet = set(res[target]["url"])
            safeEntityIdList = entityIdList

            history["safeEntity"].append(safeEntityIdList.copy())
            history["targetSet"].append(list(targetSet.copy()))
            history["dangerQueries"].append([])

        else:
            history = finalHistory.get(str(docId))
            k = len(history["safeEntity"])  # start from k-combination
            if k > N:
                print(f"Previous Search Results Found, Continue...")
                continue
            print(f"History Found ! Start from {k}-combinations.")
            targetSet = set(history["targetSet"][-1])
            safeEntityIdList = history["safeEntity"][-1]

        # N-combinations
        for n in range(1, N + 1):
            if n < k:
                # Skip until n = k
                continue
            print("Test", n, "combination for document", docId)
            print("Current search times", search.counter)
            print("Current safe entity number", len(safeEntityIdList))
            dangerQueries, targetSet = testCombination(safeEntityIdList, target, targetSet, res, queryDict,
                                                       n, sep=sep, maxStep=maxStep)
            if dangerQueries is None:
                break

            safeEntityIdList = getSafeEntities(safeEntityIdList, dangerQueries)

            history["safeEntity"].append(safeEntityIdList.copy())
            history["targetSet"].append(list(targetSet.copy()))
            history["dangerQueries"].append(dangerQueries.copy())

            # For data safety, save results once a certain combination is done.
            finalHistory[str(docId)] = history.copy()
            finalRes[str(docId)] = res.copy()
            saveFinalRes(finalRes, finalHistory, queryDict)
        print("Document", docId, "Done!")

