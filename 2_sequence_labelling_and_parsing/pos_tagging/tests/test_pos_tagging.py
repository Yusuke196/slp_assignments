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
        ]
    ]
    lambd = 0.9
    probs = pt.calc_probs(pt.calc_cnts(train), emi_lambd=lambd, save_path='tests/probs.json')
    tra_prob, emi_prob = probs
    n = 8

    assert tra_prob['<s>']['vbp'] == 1
    assert tra_prob['nn']['in'] == 1 / 3
    assert emi_prob['nn']['article'] == lambd * 1 / 3 + (1 - lambd) / n


def test_predict_one():
    test = [
        ('<s>', '<s>'),
        ('See', 'vbp'),
        ('the', 'dt'),
        ('article', 'nn'),
        ('on', 'in'),
        ('natural', 'jj'),  # これをjjまたはnn
        ('language', 'nn'),  # これをnn
        ('processing', 'nn'),  # これをnnと予測できるとよさそう
        ('.', '.'),
    ]
    probs = pt.load_probs('tests/probs.json')
    pred = pt.predict_one(test, probs)
    ans = [tpl[1] for tpl in test]
    print('')
    pprint(f'{ans = }')
    pprint(f'{pred = }')
