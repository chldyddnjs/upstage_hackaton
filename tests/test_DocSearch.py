from codes.upstage_hackaton.DocSearch.bge_m3 import bge
from datasets import load_dataset
import os
import json

def one_hot(scores:list):
    MAX = 0
    MAX_i = 0
    for i,score in enumerate(scores):
        if MAX < score:
            MAX = score
            MAX_i = i
    return MAX_i,MAX       

def test_docsearch():
    output_path = "output/qa_score.jsonl"
    data_files = 'legal.jsonl'
    dataset = load_dataset('json',data_files=data_files,split="train",num_proc=os.cpu_count()-2)
    model = bge('BAAI/bge-m3').model
    

    questions = ["단기 3주 주택 임대차 계약을 체결(구두)하고 3주 기간 임차료 전액과 소액의 보증금을 선불로 받았습니다. 임차인이 2주일 거주 후 일방적으로 중도해지를 단행하고(퇴실) 정산을 요구하고 있습니다. 임차인이 주장하는 정산 기준은 공정거래위원회 고시 '소비자분쟁해결 기준'에 명시된 고시원 이용중 소비자의 귀책사유로 중도에 계약해지할 경우에 해당하며 고시원운영영업의 품목별 해결 기준을 적용하여야 한다고 주장하고 있습니다. 참고로 단기 임대차 계약을 체결한 주택은 고시원이 아니고 주택 임대를 하고있습니다. 고시원이 아닌 임대 주택에 단기 임대차 계약을 맺은 임차인이 일방적으로 중도 해지하는 경우에 주택임대차보호법을 적용하지 않고 고시원운영업의 품목별 해결 기준을 의무적으로 적용하여야 하는지 궁금합니다",
                "체분된 퇴직금이 있었고 노동부 감독관에게 체불임금 확인서(사업주확인서 포함)를 받았습니다 소액체당금 신청하려는데 사업주 확인서에 사업자 등록번호가 기재되어있지 않고 감독관도 조사하였으나 알수가 없다고 하네요. 그런 경우 소액체당금 신청이 되나요?",
                "제 처의 생모(저의 장모님)는 호적상으로는 아무 관련이 없는 타인입니다. 혼외장라서 제 처는 친부와 친부의 본부인을 부모로하여 입적된 상태이고, 처의 생모는 저와 제처가 모시고 있는데 처의 생모는 호적상 무연고자입니다. 최근 병세가 악화되어 곧 수술을 앞두고 있어서 수술동의서 및 장례절차 문제가 시급한 상황이 되었습니다. 제 처의 친부와 호적상 모친은 모두 사망하셨습니다. 이런 경우에 생모의 대리인이나 보호자 역할을 할 수 있는 방법이 있을까해서 문의를 드립니다",
                "안녕하세요, 2년 하고도 2개월전 볼리비아 여성과 혼인신고를 하였습니다. 결혼하기 전에 서울에서 만났었는데, 혼인 신고를하고 비자를 발급받은 뒤 한국에 데려오기 위한 준비를 하고 있었습니다. 허나 혼인신고 후 몇달 뒤 아내는 금전을 요구하다가 다른 만나는 사람이 있다며 혼인 후 6개월정도 뒤에 연락이 두절되었습니다. 혼인무효를 하기엔 너무 늦었다 들었고 이혼을 해야한다 했는데 변호사 고용이 금전적으로 어려워 아무것도 할 수 없었습니다. 근애네 법률구조공단에 법적 자문과 지원을 받을 수 있다는 소식을 듣고 상담을 신청하기 위해 연락을 드렸습니다."
                "저는 같은 직장에서 연인 사이었고 지금은 헤어졌지만 같은 직장에서 일하는 전 남자친구가 있었습니다. 전 남자친구가 본인의 친한형과 저와의 성관계로 인해 헤어졌다는 허위 소문을 퍼뜨림으로 인해 억울하고 정신적으로 너무 괴롭다 보니 제가 직장을 그만두게 생겼어요. 정말 헤어진 이유는 전 남자 친구가 놀음을 하고 놀다가 만난 다른 여자랑 잠자리를 가지고 헤어졌음에도 불구하고 그런 소문을 퍼뜨리면서 너무 정신적인 피래가 크고 직장을 다니기도 힘들만큼 성적 수치심도 느껴지고 그만둬야할 상황까지 이르렀는데 왜 제가 사실도 아닌일에 그래야하는 지 너무 억울해요. 고소가 가능한지 궁금합니다. 제발 도와주세요 저 정말 죽고 싶어요.",
                ]
    
    for i in range(0,len(dataset['title']),10):
             
        question = [questions[0]]
        documents = dataset['title'][i:i+10]
        sentence_pairs = [[i,j] for i in question for j in documents]
        
        
        result = model.compute_score(
            sentence_pairs=sentence_pairs, 
            max_passage_length=128, # a smaller max length leads to a lower latency
            weights_for_different_modes=[0.4, 0.2, 0.4]
            )# weights_for_different_modes(w) is used to do weighted sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
        
        data = dict(similarity=result['colbert'],documents=documents,question=question)
        with open(output_path,'a') as f:
            json.dump(data,f,indent=4,ensure_ascii=False)
if __name__=="__main__":
    test_docsearch()