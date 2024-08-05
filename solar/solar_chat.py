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
question = "건강기능식품을 먹고 부작용이 일어났어요. 보상을 받고 싶은데 어떻게 해야할까요?, 법적인 조언을 주세요."

template = f"""
건강기능식품에 관한 법률 시행령 ( 약칭: 건강기능식품법 시행령 

제1조(목적) 이 영은 「건강기능식품에 관한 법률」에서 위임된 사항과 그 시행에 관하여 필요한 사항을 규정함을 목적
으로 한다

질문: {question}
"""

result = predibase_llm.complete(template)
print(result)