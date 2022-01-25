import numpy as np
from collections import Counter
from pprint import pprint
import copy


class Lm:
    def __init__(self, n=2):
        self.n = n

    def fit(self, train):
        if self.n <= 1:
            raise Exception('Set an integer greater than 1')
        self.unigram_freq = self._create_ngram(train, 1)
        self.ngram_freq = self._create_ngram(train, self.n)
        self._calc_params()

    def _create_ngram(self, text, n):
        counter = Counter()
        for sent in text:
            lst = [' '.join(sent[i:i + n]) for i in range(len(sent) - n + 1)]
            counter += Counter(lst)
        return counter.most_common()

    def _calc_params(self):
        freq_total = sum([tpl[1] for tpl in self.unigram_freq])
        self.unigram_prob = {token: freq / freq_total for token, freq
                             in self.unigram_freq}
        # pprint(self.unigram_prob)

        ngram_cnt = {}
        for ngram, freq in self.ngram_freq:
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
                dct[w] = cnt / nminus_gram_cnt
        # pprint(self.ngram_prob)

    def predict(self, test):
        pred = []
        # print(self._create_ngram(test, self.n))
        for sent in test:
            for i in range(len(sent) - self.n + 1):
                self._calc_prob(' '.join(sent[i:i + self.n]))

    def _calc_prob(self, ngram):
        cond, w = ngram.rsplit(' ', 1)
        print(cond, w)
        # return likelihood_disc + lambd * p_contin


if __name__ == '__main__':
    with open('data/wiki-en-train.word') as file:
        train = [['<s>'] + line.lower().split() + ['</s>'] for line in file][:3]
    with open('data/wiki-en-test.word') as file:
        test = [['<s>'] + line.lower().split() + ['</s>'] for line in file]

    trigram_lm = Lm(n=3)
    trigram_lm.fit(train)
    trigram_lm.predict(train)  # あとでtestに直す
