import os
import re
import openai
import time
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings,
    PropertyGraphIndex,
    ServiceContext,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import KnowledgeGraphIndex
from pyvis.network import Network
from llama_index.llms.predibase import PredibaseLLM
from GraphRAG import (
    GraphRAGStore,
    GraphRAGExtractor,
    GraphRAGQueryEngine
)
load_dotenv(verbose=True)
os.environ["PREDIBASE_API_TOKEN"] = os.getenv('PREDIBASE_API_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

llm = PredibaseLLM(
    model_name="solar-1-mini-chat-240612", 
    temperature=0.1, 
    max_new_tokens=512)

Settings.llm = llm
Settings.embed_model = resolve_embed_model("local:BAAI/bge-m3")
Settings.chunk_size = 512

documents = SimpleDirectoryReader("./data").load_data()

#setup the storage context
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

#Construct the Knowlege Graph Undex
index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    max_triplets_per_chunk=3,
    storage_context=storage_context,
    include_embeddings=True)

query = "What is ESOP?"
query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=5,)
#
message_template =f"""법원보관금취급규칙이란?"""
#
response = query_engine.query(message_template)
#
print(response)
