# import numpy as np
# import pandas as pd
from collections import Counter
from pprint import pprint


class Lm:
    def fit(self, train, n=2):
        self.train = train
        self.ngram_freq = self._create_ngram(n)
        self._calc_prob()

    # def _create_vocab(self, vocab_min_freq):
    #     counter = Counter()
    #     for sent in self.train:
    #         counter += Counter(sent)
    #     counter = self._reduce_vocab(counter, vocab_min_freq)

    #     self.token_freqs = counter.most_common()
    #     self.token_to_idx = {tpl[0]: idx for idx, tpl in enumerate(self.token_freqs)}

    # def _reduce_vocab(self, counter, vocab_min_freq):
    #     before_reduction = len(counter)
    #     tokens_to_rm = stopwords.words('english') + \
    #         [c for c in string.punctuation] + ['-LRB-', '-RRB-', "''", '``']
    #     keys_to_del = [key for key, count in counter.items()
    #                    if count <= vocab_min_freq or key in tokens_to_rm]
    #     for key in keys_to_del:
    #         del counter[key]
    #     print(f'vocab: {before_reduction = }, after_reduction = {len(counter)}')
    #     return counter

    def _create_ngram(self, n):
        counter = Counter()
        for sent in self.train:
            lst = [' '.join(sent[i:i + n]) for i in range(len(sent) - n + 1)]
            counter += Counter(lst)
        return counter.most_common()

    def _calc_prob(self):
        cnt = {}
        for ngram, freq in self.ngram_freq:
            # 最後のスペースで分割。nminus_gramは、正確にはngramの条件の部分であるn - 1語のsequence
            nminus_gram, w = ngram.rsplit(' ', 1)
            if dct := cnt.get(nminus_gram):
                dct.update({w: freq})
            else:
                cnt[nminus_gram] = {w: freq}
        pprint(cnt)

        prob = cnt.copy()
        for nminus_gram, dct in prob.items():
            nminus_gram_cnt = sum(dct.values())
            for w, cnt in dct.items():
                dct[w] = cnt / nminus_gram_cnt
        pprint(prob)

    def predict(self):
        pass


if __name__ == '__main__':
    with open('data/wiki-en-train.word') as file:
        train = [['<s>'] + line.lower().split() + ['</s>'] for line in file][:3]
    with open('data/wiki-en-test.word') as file:
        test = [['<s>'] + line.lower().split() + ['</s>'] for line in file]

    trigram = Lm()
    trigram.fit(train, n=3)
