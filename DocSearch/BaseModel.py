import os
import json
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
class bge:
    def __init__(self,model_name:str,debug:bool=False):
        super().__init__()
        
        self.model_name = model_name
        self.model = BGEM3FlagModel(self.model_name,use_fp16=True)
        self.debug = debug

    def one_hot(self,scores:list):
        MAX = 0
        MAX_i = 0
        for i,score in enumerate(scores):
            if MAX < score:
                MAX = score
                MAX_i = i
        return MAX_i,MAX       

    def top_k(self,scores,k:int=3):
        
        sorted_scores = sorted(scores.items(),lambda x:x[1])
        return sorted_scores[:k]