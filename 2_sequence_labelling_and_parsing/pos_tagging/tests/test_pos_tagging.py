import pytest
from pprint import pprint
import sys

sys.path.append('..')
import pos_tagging as pt


def test_calc_probs():
    train = [
        [
            ('<s>', '<s>'),
            ('See', 'vbp'),
            ('the', 'dt'),
            ('article', 'nn'),
            ('on', 'in'),
            ('pattern', 'nn'),
            ('recognition', 'nn'),
            ('.', '.'),
            ('</s>', '</s>'),
        ]
    ]
    probs = pt.calc_probs(pt.calc_cnts(train), save_path='tests/probs.json')
    tra_prob, emi_prob, utag_prob = probs
    lambd = 0.7
    n = 9

    assert tra_prob['<s>']['vbp'] == 1
    assert tra_prob['nn']['in'] == 1 / 3
    assert emi_prob['nn']['article'] == lambd * 1 / 3 + (1 - lambd) / n
    assert utag_prob['<s>'] == 1 / 9
    assert utag_prob['nn'] == 1 / 3


def test_predict_one():
    test = [
        ('<s>', '<s>'),
        ('See', 'vbp'),
        ('the', 'dt'),
        ('article', 'nn'),
        ('on', 'in'),
        ('natural', 'jj'),
        ('language', 'nn'),
        ('processing', 'nn'),
        ('.', '.'),
        ('</s>', '</s>'),
    ]
    probs = pt.load_probs('tests/probs.json')
    pred = pt.predict_one(test, probs)
    ans = [tpl[1] for tpl in test]
    print('')
    pprint(f'{ans = }')
    pprint(f'{pred = }')
