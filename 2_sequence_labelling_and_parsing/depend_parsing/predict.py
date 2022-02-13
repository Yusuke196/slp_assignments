import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from preprocess import extract_feats, get_buffer_head
from train import load_data

def load_svc(feat_of_clf: str):
    with open(f'models/svc_{feat_of_clf}.pickle', 'rb') as file:
        return pickle.load(file)


def predict(test: list[dict], clf: SVC, feat_of_clf: str, write=False) -> list[list]:
    res = []
    for sent in test:
        pred_heads = _predict_sent(sent, clf, feat_of_clf)
        res.append(pred_heads)
    if write:
        write_res(res, feat_of_clf)
    return res


def write_res(res: list[list], feat_of_clf: str):
    output = ''
    for pred_heads in res:
        for idx, pred_head in enumerate(pred_heads[1:], 1):
            # 予測すべき対象はpred_headだけ。究極的にはそれだけあればよい
            output += f'{idx}\t_\t_\t_\t_\t_\t{pred_head}\t_\n'
        output += '\n'

    with open(f'eval/pred_{feat_of_clf}.txt', 'w') as file:
        # 最後の'\n'は評価時にエラーの元になるので[:-1]で弾く
        file.write(output[:-1])


def _predict_sent(sent: list[dict], clf, feat_of_clf: str) -> list[int]:
    stack = [{'id': 0, 'token': 'ROOT', 'pos': 'ROOT', 'head': -1}] * 2
    buffer = sent
    pred_heads = [np.nan] * (len(sent) + 1)

    while buffer != [] or len(stack) > 2:
        feats = extract_feats(stack[-2:], get_buffer_head(buffer), feat_of_clf)
        pred_action = _predict_one(feats, clf)
        # print('')
        # print(f'{stack = }')
        # print(f'{buffer = }')
        # print(f'{feats = }')
        # print(f'{pred_action = }')
        left = stack[-2]
        right = stack[-1]
        # len(stack) == 2という条件は、stackにROOTしかない状態でShift以外が起きるのを防ぐ
        if len(stack) == 2 or pred_action == 'Shift':
            if len(buffer) > 0:
                stack.append(buffer.pop(0))
            else:
                # なんとなく、Reduce Rの方がReduce Lより一般的な気がしたので、Rの操作をする。TODO 要検討
                # print('Applied Action: Reduce R')
                pred_heads[right['id']] = left['id']
                stack.pop(-1)
        else:
            if pred_action == 'Reduce L':
                pred_heads[left['id']] = right['id']
                stack.pop(-2)
                # if len(stack) == 2:  # 必要なケースが出てくるかも
                #     stack.append(buffer.pop(0))
            elif pred_action == 'Reduce R':
                pred_heads[right['id']] = left['id']
                stack.pop(-1)
                # if len(stack) == 2:
                #     stack.append(buffer.pop(0))

    return pred_heads


def _predict_one(feats: dict, clf) -> str:
    # encoded = np.zeros(clf.shape_fit_[1])
    # for feat_name, feat in feats.items():
    #     idx_to_be_1 = np.where(clf.feature_names_in_ == f'{feat_name}_{feat}')
    #     encoded[idx_to_be_1] = 1
    # return clf.predict(np.expand_dims(encoded, 0))

    # 逐次的に、かつtrainのfeatureと同じになるようにone hot encodeする（testデータの性質上、最初からまとめてencodingはおそらくできない）
    encoded = pd.DataFrame(columns=clf.feature_names_in_)
    # 0と指定している行の名は、0でなくても構わない
    encoded.loc[0] = np.zeros(len(clf.feature_names_in_))
    for feat_name, feat in feats.items():
        feat_name_feat = f'{feat_name}_{feat}'
        if feat_name_feat in clf.feature_names_in_:
            encoded.loc[0, f'{feat_name}_{feat}'] = 1
    pred = clf.predict(encoded)[0]

    # try:
    #     # predictの返り値は要素一つのarrayなので、中身を取り出してから返す
    #     pred = clf.predict(encoded)[0]
    # except ValueError as e:
    #     # featが未知のものだった場合、こちらの処理が実行される
    #     print('ValueError:', e)
    #     pred = 'Shift'  # ここは恣意的に決めないといけない
    return pred


if __name__ == '__main__':
    test = load_data('data/mstparser-en-test.dep')
    feat_of_clf = 'all_feat'
    clf = load_svc(feat_of_clf)
    predict(test, clf, feat_of_clf, write=True)
