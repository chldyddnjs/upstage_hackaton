import os
import argparse
import json
from DocSearch.BaseModel import bge
from datasets import load_dataset
from multiprocessing import Pool,cpu_count
from functools import partial
from tqdm import tqdm

def one_hot(scores:list):
        MAX = 0
        MAX_i = 0
        for i,score in enumerate(scores):
            if MAX < score:
                MAX = score
                MAX_i = i
        return MAX_i,MAX       

def search(document,question,threshold):

    res = []
    sentence_pairs = [question,document]
    model = bge("BAAI/bge-m3").model
    scores = model.compute_score(
        sentence_pairs=sentence_pairs,
        max_passage_length=128,
        weights_for_different_modes=[0.4, 0.2, 0.4]
    )
    return dict(scores=scores,documents=sentence_pairs)
    # idx, _max = one_hot(scores['colbert'])
    # if threshold < _max:
    #     res.append((idx, _max))
    #     data = dict(similarity=scores['colbert'], documents=documents, question=question_list)
    #     if args.debug:
    #         print(json.dumps(data, indent=4, ensure_ascii=False))

def search_parallel(args):
    engine = partial(search,question=args.question,threshold=args.threshold)
    with Pool(processes=3) as pool:
        results = pool.map(engine,tqdm(args.ds))
    
    with open(args.output_path,'a') as f:
        json.dump(results,f,indent=4,ensure_ascii=False)

    
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--question",type=str,required=True)
    parser.add_argument("-m","--model_name",type=str,default="BAAI/bge-m3")
    parser.add_argument("-o","--output_path",type=str,default="output/qa_score.jsonl")
    parser.add_argument("-d","--data_files",type=str,default="legal.jsonl")
    parser.add_argument("-t","--threshold",type=float,default="0.3")
    return parser.parse_args()

def main(args):
    ds = load_dataset('json',data_files=args.data_files,split="train",num_proc=os.cpu_count()-2)
    args.ds = ds['title']
    # print(args.ds)
    # args.model = bge(args.model_name)
    search_parallel(args)

if __name__=="__main__":
    args = setup_args()
    main(args)
