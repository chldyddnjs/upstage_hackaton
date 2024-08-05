import os
from dotenv import load_dotenv
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.predibase import PredibaseLLM
from llama_index.core import Settings
import time

load_dotenv(verbose=True)
os.environ["PREDIBASE_API_TOKEN"] = os.getenv('PREDIBASE_API_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]


# 'storage' 폴더가 없으면
if not os.path.exists("./storage"):
    # 문서를 로드하고 색인을 생성합니다.
    documents = SimpleDirectoryReader("./data").load_data()
    # documents = os.listdir("./data/11111/")
    
    splitter = SentenceSplitter(chunk_size=1024)
    index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter],
    )
    # 나중에 사용할 수 있도록 저장
    # index.storage_context.persist('./storage')
else:
    # 저장된 인덱스 로드
    storage_context = StorageContext.from_defaults(persist_dir="./storage/")
    index = load_index_from_storage(storage_context)

Settings.embed_model = resolve_embed_model("local:BAAI/bge-m3")
Settings.chunk_size = 512

predibase_llm = PredibaseLLM(
    model_name="solar-1-mini-chat-240612", 
    temperature=0.1, 
    max_new_tokens=512)

query_engine = index.as_query_engine(llm=predibase_llm,similarity_top_k=3)
retriever = index.as_retriever(
    # vector_store_query_mode="mmr",
    similarity_top_k=3,
    # vector_store_kwargs={"mmr_threshold": 0.5},
)


question = "착오송금으로 하동군법원에 부당이득금 반환 소송을 진행하여 2021년 2월 2일 지급판결을 받았습니다. 그런데 피고는 주민등록 말소 상태이며, 연락이 되지 않는 상황입니다. 현재 착오송금된 계좌에서 어떻게 반환을 받을 수 있는지 상당을 청합니다."
template = f"""
{question}
"""

nodes = retriever.retrieve(
    question
)
response = query_engine.query(template)

for n in nodes:
    print(n)

print(response)
