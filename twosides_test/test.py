import argparse

import torch
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

import random
import os

def seed_everything(seed=42):
    '''设置整个开发环境的seed'''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

seed_everything(42)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
parser.add_argument('--pkl_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

args = parser.parse_args()
batch_size = args.batch_size
torch.cuda.set_device(args.device)
device = f'cuda' if torch.cuda.is_available() else 'cpu'
print(args)
############################################################

###### Dataset
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0],tup[1],int(tup[2]),tup[3])for tup in train_tup ]
    val_tup = [(tup[0],tup[1],int(tup[2]),tup[3])for tup in val_tup ]

    return train_tup, val_tup

df_ddi_test = pd.read_csv(f'twosides_test/twosides/fold{args.fold}/test.csv')
test_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'], df_ddi_test['Neg samples'])]
test_data = DrugDataset(test_tup)

print(f"Testing on fold {args.fold}, Samples'num: {len(test_data)}")
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)


def do_compute(batch, device, model):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch
        
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        p_score = model(pos_tri)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        n_score = model(neg_tri)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap

def test(test_data_loader,model):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_roc: {test_auc_roc:.4f}, test_f1: {test_f1:.4f}, test_precision: {test_precision:.4f},test_recall: {test_recall:.4f},test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')

test_model = torch.load(args.pkl_name, map_location=f'cuda:{args.device}').to(device)
test(test_data_loader,test_model)


