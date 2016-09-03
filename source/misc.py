import pandas as pd

from sklearn.cross_validation import KFold

def encode_cat_feature_using_targets(feature_train, targets, feature_test, n_folds=None, default_prob=0.5, random_state=None):
    assert (n_folds is None) or (n_folds > 1)

    if n_folds:
        folding_maker = KFold(len(targets), n_folds=n_folds, random_state=random_state)

        encoded_test = pd.Series(index=feature_train.index)

        for train_index, test_index in folding_maker:
            target_means_by_category = targets.iloc[train_index].groupby(feature_train.iloc[train_index]).mean()

            encoded_test.iloc[test_index] = feature_train.iloc[test_index].map(target_means_by_category)

        encoded_test.fillna(default_prob, inplace=True)
    else:
        target_means_by_category = targets.groupby(feature_train).mean()
    
        encoded_test = feature_test.map(target_means_by_category).fillna(default_prob)

    return encoded_test


def make_submission(predictions, path='submission.csv', index=None, name=None):
    pd.Series(predictions, index=index, name=name).to_csv(path, index=True, header=True)