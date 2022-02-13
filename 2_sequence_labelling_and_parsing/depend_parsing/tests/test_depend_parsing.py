# TODO 本来はmoduleに合わせて分けたほうがよい
import pytest
from pprint import pprint
import sys

sys.path.append('..')
import predict as prd
import train as trn
import preprocess as pp


@pytest.fixture()
def train():
    yield trn.load_data('data/mstparser-en-train.dep')


def test_load(train):
    assert set(train[0][0].keys()) == set(['id', 'token', 'pos', 'head'])


# def test_preprocess(train):
#     preprocessed = pp.preprocess(train)
#     X, y = preprocessed


def test_extract_feats():
    root = {'token': 'ROOT', 'pos': 'ROOT'}
    feats = pp.extract_feats(st_tail_two=[root, root], buf_fir={'token': 'no', 'pos': 'UH'})
    assert set(feats.keys()) == set(
        ['st_sec', 'st_sec_pos', 'st_last', 'st_last_pos', 'buf_fir', 'buf_fir_pos']
    )


@pytest.fixture()
def feats_all():
    yield {
        'st_sec': 'ROOT',
        'st_sec_pos': 'ROOT',
        'st_last': 'ROOT',
        'st_last_pos': 'ROOT',
        'buf_fir': 'no',
        'buf_fir_pos': 'UH',
    }


def test_predict_one():
    st_tail_two = [{'token': 'ROOT', 'pos': 'ROOT'}] * 2
    buf_head = {'token': 'no', 'pos': 'UH'}
    feat_of_clf = 'st_last_pos'
    feats = pp.extract_feats(st_tail_two, buf_head, feat_of_clf)
    clf = prd.load_svc(feat_of_clf)
    pred = prd._predict_one(feats, clf)
    assert pred == 'Shift'


@pytest.fixture()
def sent():
    yield [
        {'id': 1, 'token': 'no', 'pos': 'UH'},
        {'id': 2, 'token': ',', 'pos': ','},
        {'id': 3, 'token': 'it', 'pos': 'PRP'},
        {'id': 4, 'token': 'was', 'pos': 'VBD'},
        {'id': 5, 'token': "n't", 'pos': 'RB'},
        {'id': 6, 'token': 'black', 'pos': 'JJ'},
        {'id': 7, 'token': 'monday', 'pos': 'NNP'},
        {'id': 8, 'token': '.', 'pos': '.'},
    ]


def test_predict_sent_stlastpos(sent):
    # st_last_posだけ見る場合、すべてShiftと予測するのがベストということになる。これだと役に立たない
    feat_of_clf = 'st_last_pos'
    _test_predict_sent(feat_of_clf, sent)


def test_predict_sent_allpos(sent):
    feat_of_clf = 'all_pos'
    _test_predict_sent(feat_of_clf, sent)


def test_predict_sent_allfeat(sent):
    feat_of_clf = 'all_feat'
    _test_predict_sent(feat_of_clf, sent)


def _test_predict_sent(feat_of_clf, sent):
    clf = prd.load_svc(feat_of_clf)
    pred_heads = prd._predict_sent(sent, clf, feat_of_clf)
    # print(f'{pred_heads = }')
    assert all(isinstance(num, int) for num in pred_heads[1:])


@pytest.fixture()
def test_head():
    test = trn.load_data('data/mstparser-en-test.dep')
    yield test[:3]


def test_predict_allfeat(test_head):
    feat_of_clf = 'all_feat'
    clf = prd.load_svc(feat_of_clf)
    lst_pred_heads = prd.predict(test_head, clf, feat_of_clf, write=True, is_test=True)
    print(f'{lst_pred_heads = }')
    assert len(lst_pred_heads) == 3
