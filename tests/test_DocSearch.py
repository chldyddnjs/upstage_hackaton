from ..DocSearch.search import doc_retriver
from ..utils import setup_args, test_questions



def test_doc_retriver():
    args = setup_args()
    for test_question in test_questions:
        args.question = test_question
        doc_retriver(args)

if __name__=="__main__":
    test_doc_retriver()