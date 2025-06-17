import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from features import get_features

scores = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
    9: (1, 5)
}
def manual_fe(dataset_df):
    scaler = MinMaxScaler()
    features_df = dataset_df['essay'].apply(get_features).apply(pd.Series)
    #numeric_cols = features_df.select_dtypes(include='number').columns
    normalized_features = scaler.fit_transform(features_df)
    
    dataset_df['features'] = normalized_features.tolist()
    return dataset_df
    
def load_asap_data(prompt):
    filepath = "datasets/asap_5cv"
    columns = ['essay_set', 'essay', 'domain1_score']
    folds_data = []
    
    for idx in range(0, 5):
        train_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', 'train.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        dev_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', 'dev.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        test_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', 'test.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        
        train = train_data[train_data['essay_set'] == prompt].copy()
        train['score'] = train['domain1_score'].apply(score_2_minmax, prompt=prompt)
        
        dev = dev_data[dev_data['essay_set'] == prompt].copy()
        dev['score'] = dev['domain1_score'].apply(score_2_minmax, prompt=prompt)
        
        test = test_data[test_data['essay_set'] == prompt].copy()
        test['score'] = test['domain1_score'].apply(score_2_minmax, prompt=prompt)
        
        folds_data.append([train, dev, test])
    return folds_data

def load_asap_data_cross(prompt):
    filepath = "datasets/PAES-data"
    columns = ['essay_set', 'essay', 'domain1_score']
    
    train = pd.read_csv(os.path.join(filepath, f'{prompt}', 'train.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
    dev = pd.read_csv(os.path.join(filepath, f'{prompt}', 'dev.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
    test = pd.read_csv(os.path.join(filepath, f'{prompt}', 'test.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        
    train['score'] = train.apply(normalize, axis=1)
    dev['score'] = dev.apply(normalize, axis=1)
    test['score'] = test['domain1_score'].apply(score_2_minmax, prompt=prompt)

    return [[train, dev, test]]

def load_leaf_data(prompt):
    filepath = 'datasets/xLeaf'
    columns = ['trait_1', 'trait_2', 'trait_3', 'trait_4', 'trait_5', 'trait_6', 'trait_7', 'trait_8', 'trait_9', 'trait_10']
    
    train = pd.read_csv(os.path.join(filepath, 'train.csv'))
    train['essay'] = train['essay_text']
    train['overall'] = train[columns].mean(axis=1).round().astype(int)
    train['score'] = train['overall'].apply(score_2_minmax, prompt=prompt)
    
    
    dev = pd.read_csv(os.path.join(filepath, 'dev.csv'))
    dev['essay'] = dev['essay_text']
    dev['overall'] = dev[columns].mean(axis=1).round().astype(int)
    dev['score'] = dev['overall'].apply(score_2_minmax, prompt=prompt)
    
    test = pd.read_csv(os.path.join(filepath, 'test.csv'))
    test['essay'] = test['essay_text']
    test['overall'] = test[columns].mean(axis=1).round().astype(int)
    test['score'] = test['overall'].apply(score_2_minmax, prompt=prompt)
    
    # Add manual features
    train = manual_fe(train)
    dev = manual_fe(dev)
    test = manual_fe(test)
    
    return [[train, dev, test]]

def score_2_minmax(score, prompt):
    return (score - scores[prompt][0]) / (scores[prompt][1] - scores[prompt][0])

def score_2_prompt(score, prompt):
    return round(score * (scores[prompt][1] - scores[prompt][0]) + scores[prompt][0])

def normalize(df):
    prompt = df['essay_set']
    score = df['domain1_score']
    
    return score_2_minmax(score, prompt)

def evaluate_qwk(outputs, scores, prompt):
    outputs = [int(score_2_prompt(output, prompt)) for output in outputs]
    scores = [int(score_2_prompt(score, prompt)) for score in scores]
    
    return cohen_kappa_score(outputs, scores, weights='quadratic')