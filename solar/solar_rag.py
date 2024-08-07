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

load_dotenv(verbose=True)
os.environ["PREDIBASE_API_TOKEN"] = os.getenv('PREDIBASE_API_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["CHUNK_SIZE"] = os.getenv('CHUNK_SIZE')
openai.api_key = os.environ["OPENAI_API_KEY"]

def create_docments(texts):
    documents = [Document(title=text['title'][0],text=text['text'][0]) for text in texts]
    return  documents

def main(args):

    Settings.embed_model = HuggingFaceEmbedding(args.name_or_path)
    Settings.chunk_size = args.chunk_size
    Settings.chunk_overlap = args.chunk_overlap

    
    if not os.path.exists("./storage"):
        print('Loading Documents ...')
        #찾은 법률문서
        texts = doc_retriver(args)
        documents = create_docments(texts)
        #여기까지 확인
        print("Creating Index ... ")
        splitter = SentenceSplitter()
        index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter],
        )
        index.storage_context.persist('./storage')
    else:
        print("Loading Index ...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage/")
        index = load_index_from_storage(storage_context)

    predibase_llm = PredibaseLLM(
        model_name="solar-1-mini-chat-240612", 
        temperature=0.1, 
        max_new_tokens=1024)

    query_engine = index.as_query_engine(llm=predibase_llm,similarity_top_k=args.top_k)
    retriever = index.as_retriever(
        # vector_store_query_mode="mmr",
        similarity_top_k=args.top_k,
        # vector_store_kwargs={"mmr_threshold": 0.5},
    )

    question = """안녕하세요. 저는 주택임대(소형아파트)하고 암차인으로 부터 소액의 월세를 받고 있습니다. 
    만약에 계약이 만료되기전 임차인이 갱신요구권을 사용하겠다면 예외조건에 해당이 안되며 임대인은 요청을 받아들여야 되는걸로 알고 있습니다. 
    그런데 만약 기존 임대차 계약조건이 아니고 암처안 오히려 월세를 낮추어서 갱신요구를 하는 경우 이런 경우는 임대인이 거절하고 계약해지 하면 되나요? 
    일반적으로 임대인이 인상을 요구하다 보니 5% 상향한도가 있고 하는데 반대로 임차인이 낮추어 요구하면은요?
    주택임대차 갱신요구는 임대인의 무리한 인상요구에 임차인을 보호하는 취지로 알고 있는데 기본적으로 기존 임대차유지조건이 유지되는 경우(5%인상범위안)에 한하는게 맞아 인하의 경우에는 임대인이 거절하고 계약을 해지할 수 있는것 아닌가요?
    당연히 거절이 되어서인지 인터넷에는 이러한 내용은 아무리 찾아도 없고 공인중개사에 문의하니 당연히 거절대상이라고 하네요.
    """
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
    main(args)