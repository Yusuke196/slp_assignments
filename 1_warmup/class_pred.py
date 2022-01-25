import numpy as np
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
import string


class RuleBase:
    def create_features(self, data):
        data['person_name'] = data['sentence'].map(
            lambda x: 1 if re.search(r'[A-Z]{2,}\s(no\s)?[A-Z][a-z]', x)
                        or re.search(r'[A-Z][a-z]+\s(no\s)?[A-Z]{2,}', x)
                        else 0)
        data['year_expressions'] = data['sentence'].map(
            lambda x: 1 if re.search(r'[0-9]{3}(\s\?)?\s?\-.*[0-9]{3}', x)
                        or re.search(r'of (birth|death|birth and death) unknown', x)
                        else 0)
        return data

    def predict(self, data):
        pred = data['person_name'] | data['year_expressions']
        pred = pred.map(lambda x: 1 if x == 1 else -1)

        srs = data['target'] == pred
        acc = srs.sum() / srs.count()
        return pred, acc


class NaiveBayes:
    # vocab_min_freqを設定すると、trainにおける頻度がそれより少ない語彙を登録しない
    def fit(self, train, vocab_min_freq=0):
        self.train = train
        self._create_vocab(vocab_min_freq)
        self._calc_prob()

    def _create_vocab(self, vocab_min_freq):
        counter = Counter()
        for sent in self.train['sentence']:
            counter += Counter(sent.split())
        counter = self._reduce_vocab(counter, vocab_min_freq)

        self.vocab_freqs = counter.most_common()
        self.vocabs = list(dict(self.vocab_freqs).keys())
        self.vocab_size = len(self.vocabs)

    def _reduce_vocab(self, counter, vocab_min_freq):
        before_reduction = len(counter)
        tokens_to_rm = stopwords.words('english') + \
            [c for c in string.punctuation]
        keys_to_del = [key for key, count in counter.items()
                       if count <= vocab_min_freq or key in tokens_to_rm]
        for key in keys_to_del:
            del counter[key]
        print(f'vocab: {before_reduction = }, after_reduction = {len(counter)}')

        return counter

    def _calc_prob(self):
        # P(c)を計算
        self.p_c = {}
        pos = self.train.query('target == 1')
        neg = self.train.query('target == -1')
        self.p_c['pos'] = len(pos) / len(self.train)
        self.p_c['neg'] = len(neg) / len(self.train)
        # print(f'{self.p_c = }')

        # P(w_i|c)を計算
        cnt_cls = {}
        cnt_cls['pos'] = self._calc_cnt(pos)
        cnt_cls['neg'] = self._calc_cnt(neg)
        # print(f'{cnt_cls["pos"]["samurai"][0] = }')

        self.p_wi_c = {}
        for cls, cnt in cnt_cls.items():
            self.p_wi_c[cls] = {}
            for word, lst in cnt.items():
                # Laplace Smoothingを実行
                self.p_wi_c[cls][word] = \
                    (sum(lst) + 1) / (len(lst) + self.vocab_size)
        # print(f'{self.p_wi_c["pos"]["samurai"] = :.3f}')

    def _calc_cnt(self, data):
        cnt = {}
        len_data = len(data)
        for word in self.vocabs:
            cnt[word] = [0] * len_data
        for i, sent in enumerate(data['sentence']):
            for word in sent.split():
                if lst := cnt.get(word):
                    lst[i] = 1
        return cnt

    def predict(self, test):
        pred = []
        for sent in test['sentence']:
            log_p_pos = np.log(self.p_c['pos'])
            log_p_neg = np.log(self.p_c['neg'])
            for word in sent:
                if word not in self.vocabs:
                    continue
                log_p_pos += np.log(self.p_wi_c['pos'][word])
                log_p_neg += np.log(self.p_wi_c['neg'][word])
            # print(f'{log_p_pos = }, {log_p_neg = }')
            if log_p_pos > log_p_neg:
                pred.append(1)
            else:
                pred.append(-1)

        is_accurate = test['target'] == pd.Series(pred)
        acc = is_accurate.sum() / is_accurate.count()

        return pred, acc


if __name__ == '__main__':
    train = pd.read_table('data/titles-en-train.labeled',
                          names=['target', 'sentence'])
    test = pd.read_table('data/titles-en-test.labeled',
                         names=['target', 'sentence'])

    print('Rule Base:')
    rb = RuleBase()
    test_features = rb.create_features(test)
    _, acc = rb.predict(test_features)
    print(f'{acc:.3f}')
    print('------')

    print('Naive Bayes:')
    nb = NaiveBayes()
    nb.fit(train, vocab_min_freq=0)
    _, acc = nb.predict(test)
    print(f'{acc:.3f}')
