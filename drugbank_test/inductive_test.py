from datetime import datetime
import time 
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
import json
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
parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4, 2015])
parser.add_argument('--pkl_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

args = parser.parse_args()
batch_size = args.batch_size

device = f'cuda:{args.device_num}' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

###### Dataset

df_ddi_s1 = pd.read_csv(f'drugbank_test/inductive_data/fold{args.fold}/s1.csv')
df_ddi_s2 = pd.read_csv(f'drugbank_test/inductive_data/fold{args.fold}/s2.csv')



s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]


s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f" s1 with {len(s1_data)}, and s2 with {len(s2_data)}")


s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=2)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=2)

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


def get_roc_pr_curve(probas_pred, target):
    fpr, tpr, thresholds_roc = metrics.roc_curve(target, probas_pred)
    precision_pr, recall_pr, thresholds_pr = metrics.precision_recall_curve(target, probas_pred)
    result = {
        'fpr': fpr,
        'tpr': tpr, 
        'thresholds_roc': thresholds_roc,
        'precision': precision_pr,
        'recall': recall_pr,
        'thresholds': thresholds_pr
    }
    for k, v in result.items():
        result[k] = v.tolist()
    return result

def test(s1_data_loader, s2_data_loader, model):
    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model=model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)
      
        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        

        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device,model=model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)
                
        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)
        roc_pr_curve = get_roc_pr_curve(s2_probas_pred, s2_ground_truth)
        with open(f'curve/{os.path.basename(args.pkl_name)}.json', 'w') as json_file:
            json.dump(roc_pr_curve, json_file, indent=4)

    print('\n')
    print('============================== Best Result ==============================')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')

test_model = torch.load(args.pkl_name, map_location=f'cuda:{args.device}').to(device)
test(s1_data_loader, s2_data_loader, test_model)