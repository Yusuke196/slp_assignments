import json
import numpy as np
from pprint import pprint


def load(path: str) -> list[list]:
    with open(path) as file:
        res = []
        for l in file:
            tokens = [tuple(token.split('_')) for token in l.lower().split()]
            tokens.insert(0, ('<s>', '<s>'))
            tokens.append(('</s>', '</s>'))
            res.append(tokens)
    return res


# transmission_probつまりP(t_i|t_{i-1})と、emission_probつまりP(o_i|t_i)を求めるための準備
def calc_cnts(train: list[list]) -> tuple[dict]:
    transition_cnt = {}
    emission_cnt = {}
    unitag_cnt = {}
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

            unitag_cnt.setdefault(tag, 0)
            unitag_cnt[tag] += 1
    return transition_cnt, emission_cnt, unitag_cnt


# 二次元の辞書の深い方の次元にある数値の列を対象に、確率への変換操作を繰り返す
def calc_probs(cnts: tuple[dict], save: bool = False) -> list[dict]:
    tra_cnt, emi_cnt, tag_cnt = cnts
    uniq_tokens = set()
    for dct in emi_cnt.values():
        uniq_tokens |= set(dct.keys())
    # interpolationで使用するNを、ここではtrainに含まれるtokenのユニーク数とする
    n = len(uniq_tokens)

    # tra_prob, emission_prob, unitag_probの順で追加していく
    # tra_prob[t-1][t]にP(t|t-1)、emission_prob[t][w]にP(w|t)、unitag_prob[t]にP(t)を記録
    probs = [_calc_prob(tra_cnt), _calc_prob(emi_cnt, transmission=False, n=n)]
    total = sum(tag_cnt.values())
    unitag_prob = {tag: cnt / total for tag, cnt in tag_cnt.items()}
    probs.append(unitag_prob)

    if save:
        with open('models/probs.json', 'w') as file:
            json.dump(probs, file, indent=4)
    return probs


def _calc_prob(cnt: dict[str, dict], transmission=True, n=None) -> dict[str, dict]:
    prob = {}
    lambd = 0.7

    for k, dct in cnt.items():
        total = sum(dct.values())
        if transmission:
            prob[k] = {token: cnt / total for token, cnt in dct.items()}
        else:
            # emission_probについては、interpolationによるsmoothingを実行
            prob[k] = {token: lambd * cnt / total + (1 - lambd) / n for token, cnt in dct.items()}
            prob[k].update({'<UNK>': (1 - lambd) / n})
    return prob


def predict(sent: list[tuple], probs: list[dict]):
    tra_prob, emi_prob, utag_prob = probs
    uniq_tag = set(utag_prob.keys())
    best_score = []
    prev_tags_for_best = []

    for w_i in range(len(sent)):
        best_score.append({key: -np.inf for key in uniq_tag})
        prev_tags_for_best.append({key: None for key in uniq_tag})
        for tag in uniq_tag:
            if w_i == 0:
                best_score[w_i][tag] = _log(emi_prob[tag].get(sent[w_i][0], 0))
            else:
                lprobs = []
                for prev_tag in uniq_tag - {'</s>'}:
                    # P(w|t)P(t|t-1)
                    prev_score = best_score[w_i - 1][prev_tag]
                    tp = tra_prob[prev_tag].get(tag, 0)
                    lprobs.append((prev_score + _log(tp), prev_tag))
                ep = emi_prob[tag].get(sent[w_i][0], emi_prob[tag]['<UNK>'])
                max_score, prev_tag_for_max = max(lprobs)
                best_score[w_i][tag] = max_score * ep
                prev_tags_for_best[w_i][tag] = prev_tag_for_max
    # pprint(f'{best_score[:2] = }')
    # pprint(f'{prev_tags_for_best[:2] = }')

    scores_last_tag = best_score[len(sent) - 1]
    last_token_pred = max(scores_last_tag, key=scores_last_tag.get)
    pprint(f'{last_token_pred = }')  # </s>になってほしい

    res = []
    tag_pred = last_token_pred
    for w_i in range(len(sent) - 1, -1, -1):
        res.insert(0, tag_pred)
        tag_pred = prev_tags_for_best[w_i][tag_pred]
    pprint(f'{res = }')


def _log(num: float) -> float:
    if num == 0:
        return -np.inf
    else:
        return np.log(num)


if __name__ == '__main__':
    train = load('data/wiki-en-train.norm_pos')
    cnts = calc_cnts(train)
    probs = calc_probs(cnts, save=True)

    test = load('data/wiki-en-test.norm_pos')[0]
    pprint(f'{test = }')
    predict(test, probs)
