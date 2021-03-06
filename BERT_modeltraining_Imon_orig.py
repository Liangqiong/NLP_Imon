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

from util import put_file


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def f1_score_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_recall_fscore_support(labels_flat, preds_flat, average=None)


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def data_split(critical):
    # X_train, X_val, y_train, y_val = train_test_split(critical.index.values,
    #                                                   critical.Label.values,
    #                                                   test_size=0.15,
    #                                                   random_state=42,
    #                                                   stratify=critical.Label.values)
    # critical['data_type'] = ['not_set'] * critical.shape[0]
    # critical.loc[X_train, 'data_type'] = 'train'
    # critical.loc[X_val, 'data_type'] = 'val'

    ## just also get the test split
    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(critical.index.values, critical.Label.values,
                                                      train_size=0.8, random_state=42, stratify=critical.Label.values)

    # Now since we want the valid and test size to be equal (10% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    print(X_train.shape), print(y_train.shape)
    print(X_valid.shape), print(y_valid.shape)
    print(X_test.shape), print(y_test.shape)


    critical['data_type'] = ['not_set'] * critical.shape[0]
    critical.loc[X_train, 'data_type'] = 'train'
    critical.loc[X_valid, 'data_type'] = 'val'
    critical.loc[X_test, 'data_type'] = 'test'


    return critical


def tokenizationBERT(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    possible_labels = df.Label.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    encoded_data_train = tokenizer.batch_encode_plus(
        # df[df.data_type == 'train'].impression.values,
        # df[df.data_type == 'train'].findings.values,
        df[df.data_type == 'train'].findings.values + df[df.data_type == 'train'].impression.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        # df[df.data_type == 'val'].impression.values,
        # df[df.data_type == 'val'].findings.values,
        df[df.data_type == 'val'].findings.values + df[df.data_type == 'val'].impression.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].Label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].Label.values)


    # for test dataset

    encoded_data_test = tokenizer.batch_encode_plus(
        # df[df.data_type == 'test'].impression.values,
        # df[df.data_type == 'test'].findings.values,
        df[df.data_type == 'test'].findings.values + df[df.data_type == 'test'].findings.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df[df.data_type == 'test'].Label.values)


    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    return dataset_train, dataset_val, dataset_test


def evaluate(model, dataloader_val, device):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def generate_label(preds):
    col = ['Acute', 'Critical', 'Non-Acute']
    preds_flat = np.argmax(preds, axis=1).flatten()
    preds = []
    for p in preds_flat:
        preds.append(col[p])
    return (preds)


def modeltraining(df):
    df = data_split(df)
    dataset_train, dataset_val = tokenizationBERT(df)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=3,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    batch_size = 5
    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    epochs = 5

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()

        loss_train_total = 0
        iteration = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
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
            #
            #
            # if iteration % 50 == 0:
            #     message = ('Cycle: %s, Inst: %s, Iter: %s, train loss: %.3f, train acc: %.3f' % (
            #         epoch, opt.inst_id, iteration, loss.item(), train_acc))
            #     self.super_print(message)
            #

            # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        torch.save(model.state_dict(), f'base_finetuned_BERT_epoch_{epoch}.model')
        # torch.save(model.state_dict(), os.path.join(opt.data_path, opt.dis_model_name + epoch + '.model'))

        ## save to the disck
        # put_file(ssh_client, opt.central_path, os.path.basename(opt.log))

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

    _, predictions, true_vals = evaluate(model, dataloader_validation, device)
