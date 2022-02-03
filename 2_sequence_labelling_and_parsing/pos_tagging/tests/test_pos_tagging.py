import pytest
from pprint import pprint
import sys

sys.path.append('..')
import pos_tagging


def test_predict():
    test = [
        ('<s>', '<s>'),
        ('natural', 'jj'),
        ('language', 'nn'),
        ('processing', 'nn'),
        ('is', 'vbz'),
        ('a', 'dt'),
        ('field', 'nn'),
        ('.', '.'),
        ('</s>', '</s>'),
    ]
    probs = pos_tagging.load_probs('models/probs.json')
    pred = pos_tagging.predict_one(test, probs)
    pprint(f'{pred = }')
    assert 1 == 1
