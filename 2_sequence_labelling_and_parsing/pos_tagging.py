import json
from pprint import pprint


def load(path: str) -> list[list]:
    # is_train = 'train' in path
    with open(path) as file:
        res = []
        for l in file:
            tokens = [tuple(token.split('_')) for token in l.lower().split()]
            tokens.insert(0, ('<s>', '<s>'))
            tokens.append(('</s>', '</s>'))
            res.append(tokens)
        # else:
        #     res = [['<s>'] + l.lower().split() + ['</s>'] for l in file]
    return res


# transmission_probつまりP(t_i|t_{i-1})と、emission_probつまりP(o_i|t_i)を求めるための準備
def calc_cnts(train: list[list]) -> tuple[dict]:
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
def calc_probs(cnts: tuple[dict], save: bool = False) -> list[dict]:
    # interpolationで使用するNを、ここではtrainに含まれるtokenのユニーク数とする
    uniq_tokens = set()
    for dct in cnts[1].values():
        uniq_tokens |= set(dct.keys())
    n = len(uniq_tokens)

    # transition_prob, emission_probの順で追加
    # transition_prob[t-1][t]にP(t|t-1)、emission_prob[t][w]にP(w|t)を記録
    probs = [_calc_prob(cnts[0]), _calc_prob(cnts[1], transmission=False, n=n)]
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
            # interpolationによるsmoothingを実行
            prob[k] = {token: lambd * cnt / total + (1 - lambd) / n for token, cnt in dct.items()}
            prob[k].update({'<UNK>': (1 - lambd) / n})
    return prob


def predict(sent: list[tuple], probs: list[dict]):
    transition_prob, emission_prob = probs
    uniq_pos = set(transition_prob.keys()) | set(['</s>'])
    viterbi = [[0] * len(uniq_pos) for _ in range(len(sent))]

    for t in range(len(sent)):
        for pos_i, pos in enumerate(uniq_pos):
            if t == 0:
                viterbi[t][pos_i] = emission_prob[pos].get(sent[t][0], 0)
            else:
                probs = []
                for prev_pos in uniq_pos:
                    # P(w|t)P(t|t-1)
                    ep = emission_prob[pos].get(sent[t][0], emission_prob[pos]['<UNK>'])
                    tp = transition_prob[prev_pos][pos]
                    probs.append(ep * tp)
                viterbi[t][pos_i] = max(probs)
    pprint(viterbi[:2])


if __name__ == '__main__':
    train = load('data/wiki-en-train.norm_pos')
    cnts = calc_cnts(train)
    probs = calc_probs(cnts, save=True)

    test = load('data/wiki-en-test.norm_pos')[0]
    predict(test, probs)
