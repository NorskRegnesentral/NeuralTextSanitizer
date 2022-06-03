import time
from bs4 import BeautifulSoup  # bs==4.6.3
import requests
from requests import get
import re

def _req(term, num, start, lang, proxies, verify=None):
    """
    :param term: The query
    :param num: number of urls that we want to return, but not guaranteed
    :param start: The starting point of counting the urls
    :param lang: language used in search
    :param proxies: Use proxies, format:{"http":..., "https":...}
    :param verify: verify document for the proxies
    :return: If succeeds, search results
    """
    usr_agent = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    try:
        resp = get(
            url="https://www.google.com/search",
            headers=usr_agent,
            params=dict(
                q=term,
                num=num,
                hl=lang,
                start=start,
                filter=1,  # 0: don't filter out repeated data
            ),
            proxies=proxies,
            verify=verify,  # path for the verify document
            timeout=60
        )
        resp.raise_for_status()
        return True, resp
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)  # Http Error: 429 Client Error
        return False, None
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        return False, None
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        return False, None
    except requests.exceptions.RequestException as err:
        print("Oops: Something Else", err)
        return False, None

def search(term, queryDict, num_results=10, lang="en", proxy=None, verify=None):
    """
    Get num_results, if total estimate results is < num_results, get less results
    else: at least get num_results

    queryDict: saved the query we have already searched, e.g. Turkish appears in several documents -> only search 1 time

    res["name"]: title.text
    res["Estimated Matches"]: estCnt
    res["snippetmatch"]: keyword
    res["displaySnippet"]: description.text
    """
    escaped_term = term

    # If we have already searched the term
    if escaped_term in queryDict.keys():
        return {escaped_term: queryDict[escaped_term]}

    res = {escaped_term: {"url": [], "name": [], "Estimated Matches": [], "snippetmatch": [], "displaySnippet": []}}

    verify = verify
    proxies = proxy  # {"http": ..., "https": ...}

    start = 0  # start from the first link

    # do:
    #     search
    #     num_results = min(estCnt, num_results) # update the num of results we want to get
    # while start < num_results

    condition = True
    stop = False
    while condition:
        flag = False
        while flag is False:
            # assert search.counter < 5200, "To Much Search!"
            flag, resp = _req(escaped_term, num_results, start, lang, proxies, verify)  # Send request
            search.counter += 1  # request count
            if flag is False:
                time.sleep(0.1)  # sleep before continuing
                print("Retry the query :", escaped_term)

        # Since flag = Ture -> resp not None -> Parse
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Case 1: "No results found for ..., Switch to without quotes"
        tmp = soup.find_all('div', attrs={'role': 'heading'})
        if tmp is not None and len(tmp) > 0 and "No results found for" in tmp[0].text:
            print("query = ", escaped_term, "start = ", start, tmp[0].text)
            queryDict[escaped_term] = res[escaped_term]
            return res

        # Case 2: Your search - ... - did not match any documents. [Also Including: Did you mean]
        tmp = soup.find_all('p', attrs={'role': 'heading'})
        for item in tmp:
            if "did not match any documents" in item.text:
                print("query = ", escaped_term, "start = ", start, item.text)
                queryDict[escaped_term] = res[escaped_term]
                return res

        tmp = soup.find("div", {"id": "result-stats"})             # Estimated Count
        result_block = soup.find_all('div', attrs={'class': 'g'})
        if (tmp is None or len(tmp) == 0) and result_block is None:
            # Case 3: Type 1 error, no estimated matches and no result urls found.
            # Probably due to the fact that start is too big
            filename = f"{start}-{escaped_term}-1.html".replace('/','-') # "7971/07" will cause save error
            with open("htmlError/Type1/"+filename, "w") as f:
                f.write(str(soup))
            print("query = ", escaped_term, "start = ", start,
                  "Type 1 error occurs, refer to ", "htmlError/Type1/"+filename)
            if start > 0:
                print("Probably due to big start")

            queryDict[escaped_term] = res[escaped_term]
            return res
        else:
            # (tmp is None or len(tmp) == 0) = False  or result_block is None = False
            if tmp is None or len(tmp) == 0:
                # Case 5.1: No estCnt, but have urls
                # In this case, normally the query is a general term like "Turkish Government"
                if result_block is not None:
                    estCnt = num_results  # We do not change the number of results we want to get.
                else:
                    # Case 5.2: a priori can not happen
                    filename = f"{start}-{escaped_term}-2.html".replace('/', '-')
                    with open("htmlError/Type2/" + filename, "w") as f:
                        f.write(str(soup))
                    print("query = ", escaped_term, "start = ", start,
                          "Type 2 error occurs, refer to ", "htmlError/Type2/" + filename)

                    queryDict[escaped_term] = res[escaped_term]
                    return res
            else:
                # Have estimated results, with or without urls.
                estCnt = re.search(r'([0-9]*) result', tmp.text.replace(',', ''))
                # Case 4: Correctly found estimated results
                if estCnt is not None:
                    estCnt = int(estCnt.group(1))
                else:
                    # Case 7: Cannot find estimated matches and no urls, probably due to regex error
                    filename = f"{start}-{escaped_term}-3.html".replace('/', '-')
                    with open("htmlError/Type3/" + filename, "w") as f:
                        f.write(str(soup))
                    print("query = ", escaped_term, "start = ", start,
                          "Type 3 error occurs, refer to ", "htmlError/Type3/" + filename)
                    queryDict[escaped_term] = res[escaped_term]
                    return res

        # Update num_results
        num_results = min(estCnt, num_results)

        # Gather search results
        for result in result_block:
            # Find link, title, description
            link = result.find('a', href=True)
            title = result.find('h3')
            description_box = result.find('div', {'style': '-webkit-line-clamp:2'})
            if description_box:
                description = description_box
                tmp = description.find_all('em')
                if tmp is not None:
                    keyword = [i.text for i in description.find_all('em')]
                else:
                    keyword = []

                if link and title and description:
                    # A valid result must have a link, a title and a description.
                    start += 1
                    res[escaped_term]["url"].append(link['href'])
                    res[escaped_term]["name"].append(title.text)
                    res[escaped_term]["Estimated Matches"].append(estCnt)
                    res[escaped_term]["snippetmatch"].append(keyword)
                    res[escaped_term]["displaySnippet"].append(description.text)

        # Case 6:  It looks like there aren't many great matches for your search
        # In this case, we will not continue the search, since num_results may not be accurate, ex, 2 results
        # But in fact we could only get 1 result, i.e. start = 1
        tmp = soup.find_all('div', attrs={'role': 'heading', 'class': 'v3jTId'})
        for item in tmp:
            if "It looks like there" in item.text:
                print("query = ", escaped_term, "start = ", start, "num_results = ", num_results, item.text)
                stop = True

        if start < num_results and not stop:
            # sleep before continue
            time.sleep(0.1)
        else:
            condition = False

    queryDict[escaped_term] = res[escaped_term]
    return res

def saveRes(search_results, res):
    query = list(search_results.keys())[0]
    urls = search_results[query].get("url", [])
    names = search_results[query].get("name", [])
    estimatedMatches = search_results[query].get("Estimated Matches", [])
    snippetMatch = search_results[query].get("snippetmatch", [])
    displaySnippet = search_results[query].get("displaySnippet", [])

    if res.get(query) is None:
        res[query] = {"url": urls,
                      "name": names,
                      "Estimated Matches": estimatedMatches,
                      "snippetmatch": snippetMatch,
                      "displaySnippet": displaySnippet}
    else:
        print("Duplicated Results, probably due to the fact that query = target = ",query)

