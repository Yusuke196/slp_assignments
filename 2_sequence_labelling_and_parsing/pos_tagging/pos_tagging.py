import json
import numpy as np
from pprint import pprint


def load(path: str) -> list[list]:
    with open(path) as file:
        res = []
        for l in file:
            tokens = [tuple(token.split('_')) for token in l.lower().split()]
            tokens.insert(0, ('<s>', '<s>'))
            res.append(tokens)
    return res


def fit(train: list[list], emi_lambd: float, save_path: str = '') -> list[dict]:
    cnts = _calc_cnts(train)
    probs = _calc_probs(cnts, emi_lambd=emi_lambd, save_path=save_path)
    return probs


# transition_probつまりP(t_i|t_{i-1})と、emission_probつまりP(o_i|t_i)を求めるための準備
def _calc_cnts(train: list[list]) -> tuple[dict]:
    transition_cnt = {}
    emission_cnt = {}
    for tpls in train:
        # 高速化の余地があるが、可読性を損ないそうなので差し当たりは現状維持
        for i in range(1, len(tpls)):
            prev_tag = tpls[i - 1][1]
            tag = tpls[i][1]
            transition_cnt.setdefault(prev_tag, {})
            transition_cnt[prev_tag].setdefault(tag, 0)
            transition_cnt[prev_tag][tag] += 1
        for tpl in tpls:
            token = tpl[0]
            tag = tpl[1]
            emission_cnt.setdefault(tag, {})
            emission_cnt[tag].setdefault(token, 0)
            emission_cnt[tag][token] += 1
    return transition_cnt, emission_cnt


# 二次元の辞書の深い方の次元にある数値の列を対象に、確率への変換操作を繰り返す
def _calc_probs(cnts: tuple[dict], emi_lambd: float, save_path: str = '') -> list[dict]:
    tra_cnt, emi_cnt = cnts
    # interpolationで使用するNを、ここではtrainに含まれるtokenのユニーク数とする
    uniq_tokens = set()
    for dct in emi_cnt.values():
        uniq_tokens |= set(dct.keys())

    # transition_prob, emission_probの順で追加していく
    # tra_prob[t-1][t]にP(t|t-1)、emi_prob[t][w]にP(w|t)を記録
    probs = [_calc_prob(tra_cnt), _calc_prob(emi_cnt, transition=False, lambd=emi_lambd)]

    if save_path != '':
        with open(save_path, 'w') as file:
            json.dump(probs, file, indent=4)
    return probs


def _calc_prob(
    cnt: dict[str, dict], transition: bool = True, lambd: float = None
) -> dict[str, dict]:
    prob = {}

    for k, dct in cnt.items():
        total = sum(dct.values())
        if transition:
            prob[k] = {token: cnt / total for token, cnt in dct.items()}
        else:
            # ここで足している1は、UNKの分。これをvocab sizeとして使う
            n = len(dct) + 1
            # emission_probについては、interpolationによるsmoothingを実行
            prob[k] = {token: lambd * cnt / total + (1 - lambd) / n for token, cnt in dct.items()}
            prob[k].update({'<UNK>': (1 - lambd) / n})
    return prob


def load_probs(path):
    with open(path) as file:
        return json.load(file)


def predict_one(sent: list[tuple], probs: list[dict]) -> list[str]:
    best_score, prev_tags_for_best = _forward(sent, probs)
    scores_last_tag = best_score[len(sent) - 1]
    last_token_pred = max(scores_last_tag, key=scores_last_tag.get)
    # pprint(f'{last_token_pred = }')  # .になってほしい

    res = []
    tag_pred = last_token_pred
    # w_iをlen(sent) - 1から1ずつ減らしていく
    for w_i in range(len(sent) - 1, -1, -1):
        res.insert(0, tag_pred)
        tag_pred = prev_tags_for_best[w_i][tag_pred]

    ans = [tpl[1] for tpl in sent]  # accuracyの計算用に、正答も返す
    print(f'{ans = }')
    print(f'{res = }')
    return res, ans


def _forward(sent: list[tuple], probs: list[dict]):
    tra_prob, emi_prob = probs
    uniq_tag = set(tra_prob.keys())

    best_score = []
    prev_tags_for_best = []
    for w_i in range(len(sent)):
        best_score.append({key: -np.inf for key in uniq_tag})
        prev_tags_for_best.append({key: None for key in uniq_tag})
        for tag in uniq_tag:
            ep = _log(emi_prob[tag].get(sent[w_i][0], emi_prob[tag]['<UNK>']))
            if w_i == 0:
                best_score[w_i][tag] = ep
            else:
                scores_and_prev_tags = []
                for prev_tag in uniq_tag:
                    # P(w|t)P(t|t-1)
                    prev_score = best_score[w_i - 1][prev_tag]
                    tp = tra_prob[prev_tag].get(tag, 0)
                    scores_and_prev_tags.append((prev_score + _log(tp), prev_tag))
                max_score, prev_tag_for_max = max(scores_and_prev_tags)
                best_score[w_i][tag] = max_score + ep
                prev_tags_for_best[w_i][tag] = prev_tag_for_max
    # print('best_score:')
    # pprint(best_score)
    return best_score, prev_tags_for_best


def _log(num: float) -> float:
    if num == 0:
        return -np.inf
    else:
        return np.log(num)


if __name__ == '__main__':
    train = load('data/wiki-en-train.norm_pos')
    probs = fit(train, emi_lambd=0.95, save_path='models/probs.json')

    test = load('data/wiki-en-test.norm_pos')[0]
    predict_one(test, probs)
