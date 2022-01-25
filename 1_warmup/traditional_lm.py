import numpy as np
from collections import Counter
from pprint import pprint
import copy


class Lm:
    def __init__(self, n):
        self.n = n
        self.d = 0.1

    def fit(self, text: list):
        self._create_vocab(text)
        self._calc_params()

    def _create_vocab(self, text):
        if self.n <= 1:
            raise Exception('Set an integer greater than 1')
        unigram_freq = self._create_ngram(text, 1)
        text_unk = self.replace_rare(text, unigram_freq)
        self.unigram_freq = self._create_ngram(text_unk, 1)
        self.ngram_freq = self._create_ngram(text_unk, self.n)
        # self.nminusgram_freq = self._create_ngram(text_unk, self.n - 1)

    def _create_ngram(self, text: list, n):
        counter = Counter()
        for sent in text:
            lst = [' '.join(sent[i:i + n]) for i in range(len(sent) - n + 1)]
            counter += Counter(lst)
        return dict(counter.most_common())

    def replace_rare(self, text: list, unigram_freq=None):
        if unigram_freq is None:
            unigram_freq = self.unigram_freq
        '''頻度1以下の単語をUNKに置き換えて返す'''
        text_unk = []
        for sent in text:
            # 内包表記だと、少し読みづらくなる
            # text_unk.append([token
            #     if unigram_freq.get(token) and unigram_freq.get(token) > 1
            #     else '<UNK>' for token in sent])
            lst = []
            for token in sent:
                if unigram_freq.get(token) and unigram_freq.get(token) > 1:
                    lst.append(token)
                else:
                    lst.append('<UNK>')
            text_unk.append(lst)
        return text_unk

    def _calc_params(self):
        freq_total = sum(self.unigram_freq.values())
        self.unigram_prob = {token: freq / freq_total for token, freq
                             in self.unigram_freq.items()}
        # pprint(self.unigram_prob)

        ngram_cnt = {}
        for ngram, freq in self.ngram_freq.items():
            # 最後のスペースで分割。nminus_gramは、正確にはngramの条件の部分であるn - 1語のsequence
            nminus_gram, w = ngram.rsplit(' ', 1)
            if dct := ngram_cnt.get(nminus_gram):
                dct.update({w: freq})
            else:
                ngram_cnt[nminus_gram] = {w: freq}
        # pprint(ngram_cnt)

        self.ngram_prob = copy.deepcopy(ngram_cnt)
        for nminus_gram, dct in self.ngram_prob.items():
            nminus_gram_cnt = sum(dct.values())
            for w, cnt in dct.items():
                dct[w] = max(cnt - self.d, 0) / nminus_gram_cnt
        # pprint(self.ngram_prob)

    def calc_entropy(self, test: list):
        probs = []
        # print(self._create_ngram(test, self.n))
        for sent in test:
            for i in range(len(sent) - self.n + 1):
                prob = self._calc_prob(' '.join(sent[i:i + self.n]))
                probs.append(prob)
        return -1 / len(probs) * sum(np.log(probs))

    def _calc_prob(self, ngram: str):
        context, w = ngram.rsplit(' ', 1)
        # print(context, w)
        likelihood_disc = 0.5  # 仮
        lambd = self._calc_lambd(context)
        p_contin = self._calc_p_contin(w)
        return likelihood_disc + lambd * p_contin

    def _calc_lambd(self, context: str):
        num_of_w_types = 0  # 与えられたcontextの後で出てくるwのユニーク数
        num_of_ngram_types_w_given_context = 0  # 分母
        for ngram, freq in self.ngram_freq.items():
            if ngram.split(' ', 1) == context:
                num_of_w_types += 1
                num_of_ngram_types_w_given_context += freq
        return self.d / num_of_ngram_types_w_given_context * num_of_w_types

    def _calc_p_contin(self, w: str):
        # 要メモ化
        num_of_context_types = 0  # 分子
        for ngram in self.ngram_freq.keys():
            if ngram.split(' ', 1)[0] == w:
                num_of_context_types += 1
        num_of_ngram_types = len(self.ngram_freq)  # 分母
        return num_of_context_types / num_of_ngram_types


if __name__ == '__main__':
    with open('data/wiki-en-train.word') as file:
        train = [['<s>'] + line.lower().split() + ['</s>'] for line in file]
    with open('data/wiki-en-test.word') as file:
        test = [['<s>'] + line.lower().split() + ['</s>'] for line in file]

    trigram_lm = Lm(n=3)
    trigram_lm.fit(train)
    test_unk = trigram_lm.replace_rare(test)
    entropy = trigram_lm.calc_entropy(test_unk)
    print(f'{entropy:.3f}')
