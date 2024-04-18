# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from typing import List, Optional
import json
import fire
import re
import argparse
from llama import Llama, Dialog
 
 

PROMPT_EVAL ="""
You are a helpful AI assistant. You will be presented with a REFERENCE ANSWER and a PREDICTED ANSWER. Your task is to rate the correctness of the PREDICTED ANSWER. 

Chose one of the following rating: 0 (Totally Wrong), 1 (Partially Correct), or 2(Totally Correct).
Just complete the last space of the correctness score.
"""
 

def try_parse_score(s):
    try:
        text_lower = s.lower()
        # Regular expression to find patterns that start with 'score:' followed by any number of spaces and then a digit or more
        pattern_score = r"score:\s*(-?\d+)"
        # Search for the numeric score pattern in the text
        match_score = re.search(pattern_score, text_lower)
        if match_score:
            score = match_score.group(1)
        else:
            if "totally wrong" in text_lower or "incorrect" in text_lower:
                score = 0
            elif "partially correct" in text_lower:
                score = 1
            elif "correct" in text_lower:
                score = 2
            else:
                score = "-1"
    except:
        score = "-1"
    return score
            
def user_input_eval(question, context, gt, pred):
    return f"Ground truth: {gt}\n\nPrediction: {pred}\n\nScore: "

def main(
        data_path:str,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4098,
        max_batch_size: int = 24,
        max_gen_len: Optional[int] = None,):
        
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
 
        
    save_path =data_path.replace('.json','-llava-score.json')
    if os.path.exists(save_path):
        answers = json.load(open(save_path))
    else:
        answers = {}
    processed_rows = set(answers.keys())



    '----------------start evaluation---------'
    with open(data_path, 'r') as file:
        j = json.load(file)
    count =0 

    for k, v in j.items():
        if k in processed_rows:
            pass
        else:
            q= v["title"]
            c = v["main_text"]
            gt = v['gt']
            pred = v['pred']
            user_message = user_input_eval("", "", gt,pred)
            prompt =PROMPT_EVAL
            new_mes=[{"role": "system", "content":prompt },{"role": "user", "content": user_message}]
            dialogs: List[Dialog] = [
                new_mes,
                # new_mes2
            ]
            if len(user_message.split())>4095:
                print("larger than max seq len, pass this visual question")
                pass
            else:
                try:
                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    for dialog, result in zip(dialogs, results):
                        score= (try_parse_score(result['generation']['content']))
                    
                    if score!="-1":
                        # v.update({result_key: score})
                        answers[k]=score
                        count+=1
                    print(score)
                except:
                    print(f"fail to proceed {k}")
        # if (count + 1) % 10 == 0:   
    with open(save_path,"w") as o_f:
        json.dump(answers,o_f)
if __name__ == "__main__":
    main("test.json","llama-2-13b-chat/","tokenizer.model",max_seq_len=4096,max_batch_size=16)
