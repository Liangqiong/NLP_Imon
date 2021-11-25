import pandas as pd
from SectionSeg import complex_split
from ClinicalBERT_training_original_imon import clinicalmodeltraining, f1_score_func, evaluate, data_split, tokenizationBERT

import argparse

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
import numpy as np
import os
from copy import deepcopy
from util import put_file
## step 1: get data of different clients
def one_round_train(seed_val, df, model_name, data_path, client_names):
    ## initialization
    epochs = 5
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(
        # "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model.to(device)
    ##  step 1: get the dataset

    critical = pd.read_excel(os.path.join(data_path,'JCAR_annotated.xlsx'))

    critical = critical[critical['Label'] != 'Not annotated']

    critical['Label'].replace('Non-Acute', 0, inplace=True)
    critical['Label'].replace('Acute', 0, inplace=True)
    critical['Label'].replace('Critical', 1, inplace=True)
    critical = critical.fillna('N/A')
    critical = complex_split(critical)
    imon_df = data_split(critical)

    imon_dataset_train, imon_dataset_val, imon_dataset_test = tokenizationBERT(imon_df, sel_fun='impression+finding') # impression+finding
    batch_size = 5

    imon_dataloader_train = DataLoader(imon_dataset_train,
                                           sampler=RandomSampler(imon_dataset_train),
                                           batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                          lr=1e-5,
                          eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(imon_dataloader_train) * epochs)

    train_datasets = [imon_dataset_train]
    val_datasets = [imon_dataset_val]
    test_datasets = [imon_dataset_test]

    ##------------ step 2: Start  local train
    print(device)
    clients_val_f1 = {}
    best_val_f1 = {}
    best_models = {}
    for name in client_names:
        clients_val_f1[name] = 0
        best_val_f1[name] = 0
        best_models[name] = {}

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()

        for dataset_train, dataset_val, name in zip(train_datasets, val_datasets, client_names):
            dataloader_validation = DataLoader(dataset_val,
                                               sampler=SequentialSampler(dataset_val),
                                               batch_size=batch_size)

            dataloader_train = DataLoader(dataset_train,
                                          sampler=RandomSampler(dataset_train),
                                          batch_size=batch_size)

            # start of standard trains

            loss_train_total = 0
            iteration = 0

            for batch in dataloader_train:
                iteration += 1

                model.zero_grad()

                batch = tuple(b.to(device) for b in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }

                outputs = model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                if iteration % 50 == 0:
                    message = ('Cycle: %s, Inst: %s, Iter: %s, train loss: %.3f' % (
                        epoch, name, iteration, loss.item()))
                    print(message)

            ## save to the disck
            # put_file(ssh_client, opt.central_path, os.path.basename(opt.log))

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
            val_f1 = f1_score_func(predictions, true_vals)
            if val_f1 > clients_val_f1[name]:
                clients_val_f1[name] = val_f1
                torch.save(model.state_dict(), './models/clinical/' + model_name + name + '_seed' + str(seed_val) + '.pth')
                best_models[name] = deepcopy(model)
                print('The updated validation fl score is', val_f1)

            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

        print(epoch, clients_val_f1)

    ##-----------------Step 3-----------------: Test the model-----------------

    test_client_names = ['imon']

    for model_name in client_names:
        model = best_models[model_name]
        ff = [model_name + '_model']

        for dataset_test, name in zip(test_datasets, test_client_names):
            dataloader_test = DataLoader(dataset_test,
                                             sampler=SequentialSampler(dataset_test),
                                             batch_size=batch_size)

            val_loss, predictions, true_vals = evaluate(model, dataloader_test, device)
            val_f1 = f1_score_func(predictions, true_vals)
            ff.append(val_f1)
            print('Fl score on client', name, 'tested with client model', model_name, val_f1)

        df.loc[df.__len__()] = ff

    return  df

if __name__ == "__main__":
    df = pd.DataFrame(columns=['model_type', 'stanford_data', 'imon_data'])

    CWT_train = False
    client_names = ['imon']
    model_name = 'imon_clinical_impressions+findings_'
    data_path = '/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/NLP/BERTCriticalFinding-main/src/data/'

    ## main function , three round of train
    for seed_val in [1234, 99, 17]:
        df = one_round_train(seed_val, df, model_name, data_path, client_names)

    print(df)

    df.to_csv(os.path.join(data_path, model_name + '.csv'), index=False)

    ## get the mean and std
    print('==================Final_output, mean and std')

    print(df[df['model_type'] == client_names[0] + '_model'].mean(),
          df[df['model_type'] == client_names[0] + '_model'].std())


