from utils import setup_args, test_questions
from solar import solar_rag

def test_solar_rag():
    import time

    args = setup_args()
    for test_question in test_questions:
        args.question=test_question
        args.llm = solar_rag.load_solar_mini()
        args.embed_model = solar_rag.load_embed_model_hf(args.name_or_path)
        
        s = time.time()
        solar_rag.run(args)
        e = time.time()
        print("="*50)
        print("Runing Time : ",e-s)
        print("="*50)
    
if __name__=="__main__":
    
    test_solar_rag()

