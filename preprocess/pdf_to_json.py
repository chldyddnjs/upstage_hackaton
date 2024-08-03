import os
import PyPDF2
import argparse
import json
import re
from tqdm import tqdm
import multiprocessing as mp

def clean(text):
    pattern = r"[^ㄱ-ㅎ가-힣0-9\n\.]"
    clean_text = re.sub(pattern," ",text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'삭제 \d\. \d\. \d\.', '',clean_text)
    return clean_text

def extract_text_from_pdf(pdf_file_path):

    try:
        title = clean(pdf_file_path.split(".pdf")[-2].split("/")[-1])
    except IndexError as e:
        print(e,pdf_file_path)
        return None

    with open(pdf_file_path, "rb") as file:
        
        reader = PyPDF2.PdfReader(file)

        full_text = ""
        
        for page in reader.pages:
            full_text += page.extract_text()

        text = '\n'.join(full_text.split("\n")[3:4] + full_text.split("\n")[6:])

    return  dict(title=clean(title).strip(), text=clean(text).strip())
    

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--roots', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-o','--output_file',default="legal.json",type=str)
    return parser.parse_args()

def main(args):
    #["11111","22222","33333","4444","55555","66666"]
    roots = args.roots
    for root in roots:
        paths = os.listdir(f"./{root}/")
        for path in tqdm(paths):
            data = extract_text_from_pdf(f"./{root}/"+path)
            if data is not None:
                with open(args.output_file,"a") as f:
                    json.dump(data,f,ensure_ascii=False)

if __name__=="__main__":
    args = setup_args()
    main(args)
    # extract_text_from_pdf("./data/11111/「대한민국 법원의 날」제정에 관한 규칙(대법원규칙)(제02605호)(20150629).pdf")
    