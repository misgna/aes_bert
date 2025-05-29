import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

scores = {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }

def load_data(prompt):
    filepath = "dataset/asap_5cv"
    columns = ['essay_set', 'essay', 'domain1_score']
    folds_data = []

    for idx in range(0,5):   
        train_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'train.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        dev_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'dev.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        test_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'test.tsv'), sep='\t', encoding='ISO-8859-1', usecols=columns)
        
        train = train_data[train_data['essay_set'] == prompt]
        train['score'] = train_data['domain1_score'].apply(score_2_minmax, prompt=prompt)

        dev = dev_data[dev_data['essay_set'] == prompt]
        dev['score'] = dev_data['domain1_score'].apply(score_2_minmax, prompt=prompt)

        test = test_data[test_data['essay_set'] == prompt]
        test['score'] = test_data['domain1_score'].apply(score_2_minmax, prompt=prompt)

        folds_data.append([train, dev, test])
    return folds_data


def score_2_minmax(score, prompt):
    return (score - scores[prompt][0]) / (scores[prompt][1] - scores[prompt][0])

def score_2_prompt(score, prompt):
    return round(score * (scores[prompt][1] - scores[prompt][0]) + scores[prompt][0])

def evaluate_qwk(outputs, scores, prompt):
    outputs = [int(score_2_prompt(out, prompt)) for out in outputs]
    scores = [score for score  in scores] 

    return cohen_kappa_score(outputs, scores, weights='quadratic')