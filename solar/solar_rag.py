import os
import openai
from dotenv import load_dotenv
from utils import setup_args
from DocSearch.search import doc_retriver
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.predibase import PredibaseLLM
from llama_index.core import Settings
import time

load_dotenv(verbose=True)
os.environ["PREDIBASE_API_TOKEN"] = os.getenv('PREDIBASE_API_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["CHUNK_SIZE"] = os.getenv('CHUNK_SIZE')
openai.api_key = os.environ["OPENAI_API_KEY"]

def create_docments(texts):
    documents = [Document(title=text['title'][0],text=text['text'][0]) for text in texts]
    return  documents

def load_solar_mini():
    return PredibaseLLM(
        model_name="solar-1-mini-chat-240612", 
        temperature=0.1,
        max_new_tokens=1024)

def load_embed_model_hf(name_or_path):
    return HuggingFaceEmbedding(name_or_path)

def run(args):

    if not os.path.exists("./storage"):
        print('Loading Documents ...')
        #찾은 법률문서
        texts = doc_retriver(args)
        documents = create_docments(texts)
        #여기까지 확인
        print("Creating Index ... ")
        splitter = SentenceSplitter(chunk_size=args.chunk_size,chunk_overlap=args.chunk_overlap)
        index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter],embed_model=args.embed_model
        )
        # index.storage_context.persist('./storage')
    else:
        print("Loading Index ...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage/")
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(llm=args.llm,similarity_top_k=args.top_k)
    retriever = index.as_retriever(
        # vector_store_query_mode="mmr",
        similarity_top_k=args.top_k,
        # vector_store_kwargs={"mmr_threshold": 0.5},
    )

    template = f"""
    questions: {args.question}\n\n답변은 다음의 형식을 따라야 합니다: (1) 사실관계 - (2) 관련 법률의 일반 내용 – (3) 사실관계의 적용 – (4) 결론.\n답변은 반드시 legal material에 있는 내용을 기반해서 해야 합니다."

    """

    nodes = retriever.retrieve(
        template
    )
    response = query_engine.query(template)

    for n in nodes:
        print(n)

    print(response)

if __name__=="__main__":
    args = setup_args()
    args.llm = load_solar_mini()
    args.embed_model = load_embed_model_hf(args.name_or_path)
    run(args)