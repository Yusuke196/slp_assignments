from sklearn.model_selection import train_test_split
import pos_tagging as pt


if __name__ == '__main__':
    data = pt.load('data/wiki-en-train.norm_pos')
    train, test = train_test_split(data)
    print(f'{len(train) = }')
    print(f'{len(test) = }')
    for flt in [0.9999, 0.99999, 0.999999, 0.9999999]:
        probs = pt.fit(train, emi_lambd=flt)
        pred_all, acc = pt.predict_all(test, probs)
        print(f'{flt}: {acc = }')
