# import torch
import os
import pandas as pd
from gem_metrics.tokenize import default_tokenize_func
import json
from pycountry import languages
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import re
import string
import sys
sys.path.append("../")
PUNCTUATION = set(string.punctuation)
print(PUNCTUATION)
from PIL import Image
import sys
import pandas as pd
import json
import gem_metrics
import time
import sys
sys.path.append("../")
from evalAll import  *
"""set up """

def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return  score
def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    return score
def get_metrics(list_of_predictions,list_of_references):
    preds = gem_metrics.texts.Predictions(list_of_predictions)
    refs = gem_metrics.texts.References(list_of_references)  # input may be list of lists for multi-ref
    result = gem_metrics.compute(preds, refs, metrics_list=['bertscore']) 
    # result = gem_metrics.compute(preds, refs, metrics_list=['bleu','rouge','bertscore','cider']) #'bleurt' # add list of desired metrics here
    return result
    

def coco_metrics(o_predictions,o_references,metric):
    token_func = default_tokenize_func(languages.get(alpha_2="en"))
    o_predictions = [token_func(pred) for pred in o_predictions]
    o_references= [token_func(ref) for ref in o_references]
    o_predictions =[[w.lower() for w in ref] for ref in o_predictions]
    o_references = [[w.lower() for w in ref] for ref in o_references]

    tmp_predictions=[]
    for ref in o_predictions:
        tmp_ref = []
        for w in ref:
            if w not in PUNCTUATION:
                tmp_ref.append(w)
        tmpstring =  " ".join(tmp_ref)
        tmp_predictions.append(tmpstring)

    tmp_references=[]
    for ref in o_references:
        tmp_ref = []
        for w in ref:
            if w not in PUNCTUATION:
                tmp_ref.append(w)
        tmpstring =  " ".join(tmp_ref)
        tmp_references.append(tmpstring)

    new_cap = {}
    for i in range(0,len(tmp_predictions)):
        new_cap[i] = [tmp_predictions[i]]

    new_ref = {}
    for i in range(0,len(tmp_references)):
        new_ref[i] = [tmp_references[i]]

    gts = new_ref
    res = new_cap
    if metric=="rougel":
        return rouge(gts, res)
    if metric =="meteor":
        return meteor(gts, res)

def main(data_path,whichmetric):
    savepath =data_path.replace('.json','-'+whichmetric+'.json')
    '---------------If has processed--------------------'
    if os.path.exists(savepath):
        with open(savepath, 'r') as f:
            new_json = json.load(f)
    else:
        new_json = {}
    processed_rows = set(new_json.keys())
    '---------------load prediction--------------------'
    with open(data_path,"r") as file:
        json_data = json.load(file)
        
    # count =0 
    # Iterate over the keys and access the values
    for key, value in json_data.items():
        if key in processed_rows:
            print(key,"pass")
            pass
        else:
            print(key)
            o_prediction=[value["pred"]]
            o_reference=[value["gt"]]

            if "bert" in whichmetric:
                o_metric = (get_metrics(o_prediction,o_reference))
                new_json[key]=o_metric["bertscore"]
            else:
                o_metric = (coco_metrics(o_prediction,o_reference,whichmetric))
                new_json[key]=o_metric
        #     count+=1
        # if (count + 1) % 10 == 0:
    with open(savepath, "w") as o_f:
        json.dump(new_json, o_f)


if __name__ == "__main__":
    main("test.json","meteor")# metric = "bert",'rougel','meteor'