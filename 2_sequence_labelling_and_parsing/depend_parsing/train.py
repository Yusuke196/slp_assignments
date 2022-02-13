import argparse
import pandas as pd
from sklearn.svm import SVC
import pickle
from preprocess import preprocess


def main(args):
    path = 'data/mstparser-en-train.dep'
    train = load_data(path)
    feat_type = 'all_feat'
    train_X, train_y = preprocess(train, feat_type=feat_type)
    build_svc(train_X, train_y, args.kernel, save=True, feat_type=feat_type)


def load_data(path: str) -> list[dict]:
    with open(path) as file:
        res = []
        sent = []
        for line in file:
            if line == '\n':
                res.append(sent)
                sent = []
            else:
                lst = [s for s in line.split()]
                sent.append(
                    {
                        'id': int(lst[0]),
                        'token': lst[1],
                        'pos': lst[3],
                        'head': int(lst[6]),
                        # 'rel': lst[7],  # この列はrelation typeを表しているが、課題を解くにあたっては不要
                    }
                )
        return res


def build_svc(X: pd.DataFrame, y: pd.Series, svc_kernel: str, save=False, feat_type='') -> SVC:
    clf = SVC(kernel=svc_kernel)
    clf.fit(X, y)
    if save:
        with open(f'models/svc_{svc_kernel}_{feat_type}.pickle', 'wb') as file:
            pickle.dump(clf, file)
    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', help='kernel of the SVC, e.g., rbf or poly')
    args = parser.parse_args()
    main(args)
