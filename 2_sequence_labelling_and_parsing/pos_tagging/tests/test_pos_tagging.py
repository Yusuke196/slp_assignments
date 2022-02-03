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
    pred = pos_tagging.predict_one(test)
    pprint(f'{pred = }')
    assert 1 == 1
