import warnings
import os
import torch
import pandas as pd
import numpy as np
from utils import setup_args
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

"""
I applied the pooling method for each model(e.g. sentence transformer, bge , etc ...)
https://www.linkedin.com/posts/prithivirajdamodaran_cls-vs-mean-pooling-is-case-by-case-activity-7127905094477496320-orXv/

"""

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(model,pool_type,tokenizer,texts,device):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)

    pooling_funcs = {
        "mean": lambda: mean_pooling(model_output, encoded_input["attention_mask"]),
        "cls": lambda: cls_pooling(model_output)
    }
    
    pooling_func = pooling_funcs.get(pool_type, None)
    
    return pooling_func()

def doc_retriver(args):
    
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset('json',data_files=args.data_files,split="train",num_proc=os.cpu_count()-1)
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    model = AutoModel.from_pretrained(args.name_or_path)
    path = f"output/embeddings_{args.name_or_path}_{args.field}_{args.pool_type}"

    if not os.path.exists(path):
        print("Creating embeddings ...")
        embeddings_ds = ds.map(
            lambda x: {"embeddings": get_embeddings(model,args.pool_type,tokenizer,x[args.field],device).detach().cpu().numpy()[0]}
        )
        embeddings_ds.save_to_disk(path)
        embeddings_ds.add_faiss_index(column="embeddings")
        embeddings_ds.save_faiss_index('embeddings',f'{path}_index.faiss')

    else:
        print("Loading embeddings ...")
        embeddings_ds = load_from_disk(path)
        embeddings_ds.load_faiss_index('embeddings',f'{path}_index.faiss')
    
    question_embedding = get_embeddings(model,args.pool_type,tokenizer,[args.question],device).cpu().detach().numpy()

    texts = []
    
    if args.similarity == "cosin":

        scores = np.array(question_embedding) @ np.array(embeddings_ds['embeddings']).T
        sorted_scores = np.sort(scores).reshape(-1)
        top_k_scores = sorted_scores[-args.top_k:].reshape(-1)
        indices = np.where(scores[0] >= top_k_scores[0])

        print("Similarity -> cosin")
        for i in indices[0]:
            print("TITLE : ",embeddings_ds['title'][int(i)])
            print("SCORE : ",scores[0][i])
            print("=" * 50)
            print()

        filtered_ds = ds.select(list(indices[0]))
        print(filtered_ds)
        texts.append(dict(title=filtered_ds['title'],text=filtered_ds['text']))

    if args.similarity == "nearest":
        print("Nearest -> distance")
        scores, samples = embeddings_ds.get_nearest_examples("embeddings", question_embedding, k=args.top_k)


        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=False, inplace=True)

        for _, row in samples_df.iterrows():
            print(f"TITLE: {row.title}")
            print(f"SCORE: {row.scores}")
            print("=" * 50)
            print()
            
        filtered_ds = ds.filter(lambda example: example['title'] in samples['title'])
        texts.append(dict(title=filtered_ds['title'],text=filtered_ds['text']))
    return texts
if __name__=="__main__":
    args = setup_args()
    doc_retriver(args)
