import sys

sys.path.insert(1, '/home/akilan/Sparkathon/models')

from bert import build_bert_model


def test_build_bert_model():
    model = build_bert_model()
    assert model is not None
    sample_text = "This is a test sentence."
    output = model.predict([sample_text])
    assert output is not None
    print(output)
