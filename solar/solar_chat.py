import os
from dotenv import load_dotenv
from llama_index.llms.predibase import PredibaseLLM

load_dotenv(verbose=True)

os.environ["PREDIBASE_API_TOKEN"] = os.getenv('PREDIBASE_API_TOKEN')

# Predibase-hosted fine-tuned adapter example
predibase_llm = PredibaseLLM(
    model_name="solar-1-mini-chat-240612", 
    temperature=0.1, 
    max_new_tokens=512)
# The `model_name` parameter is the Predibase "serverless" base_model ID
# (see https://docs.predibase.com/user-guide/inference/models for the catalog).
# You can also optionally specify a fine-tuned adapter that's hosted on Predibase or HuggingFace
# In the case of Predibase-hosted adapters, you must also specify the adapter_version
question = "무공훈장 수여를 받으려면 어떻게 해야하지?"
template = f"""

질문: {question}

아래와 같은 구조로 질문에 대한 답변을 한다.

- (1) 사실관계(법 적용과 관련 된 내용을 담은 사건개요 간략히 설명) :
- (2) 관련 법률의 일반 내용 :
- (3) 사실관계의 적용 :
- (4) 결론(요약) :
"""

result = predibase_llm.complete(template)
print(result)