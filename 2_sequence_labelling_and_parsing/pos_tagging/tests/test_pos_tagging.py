import pytest
import numpy as np
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
    lambd = 0.95
    probs = pt.fit(train, emi_lambd=lambd, save_path='tests/probs.json')
    tra_prob, emi_prob = probs

    assert tra_prob['<s>']['vbp'] == 1
    assert tra_prob['nn']['in'] == 1 / 3
    assert emi_prob['nn']['article'] == lambd * 1 / 3 + (1 - lambd) / (len(emi_prob['nn']))


@pytest.fixture()
def test():
    yield [
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


def test_forward(test):
    probs = pt.load_probs('tests/probs.json')
    best_score, prev_tags_for_best = pt._forward(test, probs)
    # P_emi(<s>|<s> as pos) = 0.975
    assert best_score[0]['<s>'] == np.log(0.975)
    # P_tra(vbp|<s>) = 1, P_emi(See|vbp) = 0.975
    assert best_score[1]['vbp'] == np.log(0.975) + pt._log(1) + pt._log(0.975)
    assert prev_tags_for_best[1]['vbp'] == '<s>'


def test_predict_one(test):
    probs = pt.load_probs('tests/probs.json')
    pred = pt.predict_one(test, probs)
    ans = [tpl[1] for tpl in test]
    print('')
    print(f'{ans = }')
    print(f'{pred = }')
