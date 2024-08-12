import sys

sys.path.insert(1, '/wd/workspaces/Sparkathon/models')

from models.bert import get_embeddings


def test_build_bert_model():
    sample_text = "This is a test sentence."
    output = get_embeddings(sample_text)
    assert output is not None
    print(output)
