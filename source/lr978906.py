import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def encode_cat_feature_using_targets(feature_train, targets, feature_test, use_loo=False, alpha=1.0):
    # Compute smoothed mean of targets for each category in feature_train
    target_means = targets.groupby(feature_train).mean()
    
    # Map smoothed means on feature_test
    feature_test = feature_test.map(lambda x: target_means.get(x, default=0.5))
    
    if use_loo:
        category_lens = targets.groupby(feature_train).transform(len)
        feature_test = (category_lens * feature_test - targets + 1) / (category_lens + 1)
        
    return feature_test

def main():
    people_cat_columns_to_use = ['people_id', 'group_1', 'char_2']
    people_num_columns_to_use = ['char_38']

    people_columns_to_use = people_cat_columns_to_use + people_num_columns_to_use

    activities_cat_columns_to_use = ['people_id']

    activities_columns_to_use = activities_cat_columns_to_use

    cat_columns_to_use = set(people_cat_columns_to_use + activities_columns_to_use)
    num_columns_to_use = people_num_columns_to_use

    target_column = 'outcome'

    activities_train = pd.read_csv('../data/act_train.csv.zip', index_col='activity_id')
    activities_test = pd.read_csv('../data/act_test.csv.zip', index_col='activity_id')
    people = pd.read_csv('../data/people.csv.zip')

    train = pd.merge(people[people_columns_to_use], 
                     activities_train[activities_columns_to_use], 
                     on='people_id', 
                     left_index=True).drop('people_id', axis=1)

    test = pd.merge(people[people_columns_to_use], 
                    activities_test[activities_columns_to_use], 
                    on='people_id', 
                    left_index=True).drop('people_id', axis=1)

    targets = activities_train[target_column]

    for col in train.columns:
        if col in cat_columns_to_use:
            test[col] = encode_cat_feature_using_targets(train[col], targets, test[col], use_loo=False)
            train[col] = encode_cat_feature_using_targets(train[col], targets, train[col], use_loo=True)

    scaler = MinMaxScaler()

    train[num_columns_to_use] = scaler.fit_transform(train[num_columns_to_use])
    test[num_columns_to_use] = scaler.transform(test[num_columns_to_use])

    log_regression = LogisticRegression(C=1, n_jobs=-1, random_state=45)
    log_regression.fit(train, targets)

    predictions = pd.Series(log_regression.predict_proba(test)[:, 1], index=test.index, name='outcome')
    predictions.to_csv('../submissions/first_log_regression.csv', index=True, header=True)
    

if __name__ == '__main__':
    main()