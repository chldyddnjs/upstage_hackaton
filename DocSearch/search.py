from DocSearch.BaseModel import bge
from datasets import load_dataset
import os

model_name = "BAAI/bge-m3"
model = bge(model_name)

if __name__=="__main__":
    data_path="legal.jsonl"
    output_path="output/qa_score.jsonl"
    model.search(data_path,output_path,0.3)
