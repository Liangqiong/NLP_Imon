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
import time
from util import put_file, get_central_files
import subprocess


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
    X_train, X_val, y_train, y_val = train_test_split(critical.index.values,
                                                      critical.Label.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=critical.Label.values)
    critical['data_type'] = ['not_set'] * critical.shape[0]
    critical.loc[X_train, 'data_type'] = 'train'
    critical.loc[X_val, 'data_type'] = 'val'
    return critical


def tokenizationBERT(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    possible_labels = df.Label.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].impression.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].impression.values,
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
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    return dataset_train, dataset_val


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


class DistrSystem(object):
    def __init__(self, opt: object, ssh_client: object) -> object:
        # CWT initial

        ## step 1: startup initial
        print('=====================================')
        print(torch.__version__)

        # Generating folder for saving intermediate results and loading files from central server
        if not os.path.exists(opt.central_path):
            os.makedirs(opt.central_path)
            print('Generate local folder for saving models and training progress', opt.central_path)
        try:
            sftp = ssh_client.open_sftp()
            sftp.mkdir(opt.central_path)
            print('Generate central folder for saving models and training progress', opt.central_path)
        except:
            pass

        # step 2: CWT initial
        self.opt = opt
        self.opt.log = os.path.join(self.opt.central_path, 'log_inst_' + str(self.opt.inst_id) + '.txt')
        self.opt.model_path = os.path.join(self.opt.central_path, self.opt.dis_model_name + '.pth')
        self.ssh_client = ssh_client
        self.opt.device = torch.device("cuda:{gpu_id}".format(gpu_id=opt.gpu_ids) if torch.cuda.is_available() else "cpu")

    def super_print(self, msg):
        print(msg)
        with open(self.opt.log, 'a') as f:
            f.write(msg + '\n')

    def load_model(self, load_path):
        pretrained_state_dict = torch.load(load_path, map_location=str(self.opt.device))
        self.model.load_state_dict(pretrained_state_dict)
        print('Loading models from server', load_path)

    def save_model(self, name=None):
        if name is None:
            save_file_name = self.opt.model_path
        else:
            save_file_name = os.path.join(self.opt.central_path, self.opt.dis_model_name + '_' + name + '.pth')
        torch.save(self.model.state_dict(), save_file_name)

    def modeltraining(self, df):
        # initial from the beginning
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


        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(dataloader_train) * self.opt.max_cycles)
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model.to(self.opt.device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        ## for the first inst: initialization or from
        if self.opt.inst_id == 1:
            with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                f.write('0' + ',' * (4 * self.opt.num_inst + 1) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')

            # if continue train, then load model from central server
            if self.opt.continue_train:
                try:
                    get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name, only_model=True)
                    load_filename = '%s_%s.pth' % (self.opt.dis_model_name, 'Best')
                    load_path = os.path.join(self.opt.model_path, load_filename)
                    self.load_model(load_path)
                except:
                    print('Central server donot contains previous saved model, we use random initization model')
            else:
                ## then we save the current refresh model and send them to central server
                self.save_model()
                subprocess.run('tar -zcvf %s.tar.gz %s' % (self.opt.model_path, self.opt.model_path), shell=True)
                put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth.tar.gz')
                subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)

        ## then start standard train
        for cycle in range(self.opt.max_cycles):
            ## substep 1: loading file from central server
            while (True):
                get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name)
                if not os.path.exists(os.path.join(self.opt.central_path, 'train_progress.csv')):
                    continue
                progress_lines = [line.strip().split(',') for line in
                                  open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
                if len(progress_lines) == 0 or int(progress_lines[-1][0]) != cycle:
                    time.sleep(self.opt.sleep_time)
                    continue
                if self.opt.inst_id == 1:
                    if cycle == 0:
                        break
                    if self.opt.val:
                        if progress_lines[-2][-1] != '' and progress_lines[-1][1] == '':
                            break
                    elif progress_lines[-2][self.opt.num_inst] != '' and progress_lines[-1][1] == '':
                        break
                else:
                    if progress_lines[-1][self.opt.inst_id - 1] != '' and progress_lines[-1][self.opt.inst_id] == '':
                        break
            ## substep 2: train local inst
            subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)
            self.load_model(self.opt.model_path)
            self.train_one_epoch(cycle, dataloader_train)

            ## substep 3: save local model and send back to server
            self.save_model(name='base')
            put_file(self.ssh_client, self.opt.central_path, os.path.basename(self.opt.log))
            ## --- just testing whether we need zip files
            # tic = time.time()
            # put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth')
            # toc = time.time() - tic
            # print('if we donot zip, then using time', toc )

            ##  -- testing the other zip
            subprocess.run('tar -zcvf %s.tar.gz %s' % (self.opt.model_path, self.opt.model_path), shell=True)
            put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth.tar.gz')
            subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)

            progress_lines = [line.strip().split(',') for line in
                              open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
            progress_lines[-1][self.opt.inst_id] = '1'
            with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                for line in progress_lines:
                    f.write(','.join(line) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')

            ## validation or not
            if self.opt.val:
                if (cycle + 1) % self.opt.val_freq == 0:
                    self.test(dataloader_validation)
            elif self.opt.inst_id == self.opt.num_inst:
                progress_lines.append(['' for i in range(len(progress_lines[-1]))])
                progress_lines[-1][0] = str(cycle + 1)
                with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                    for line in progress_lines:
                        f.write(','.join(line) + '\n')
                put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')

    def train_one_epoch(self, cycle, dataloader_train):
        model = self.model

        loss_train_total = 0
        iteration = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(cycle), leave=False, disable=False)
        for batch in progress_bar:
            iteration += 1

            model.zero_grad()

            batch = tuple(b.to(self.opt.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            if iteration % 50 == 0:
                message = ('Cycle: %s, Inst: %s, Iter: %s, train loss: %.3f' % (
                    cycle, self.opt.inst_id, iteration, loss.item() / len(batch)))
                self.super_print(message)

            # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        # torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        # torch.save(model.state_dict(), os.path.join(opt.data_path, opt.dis_model_name + cycle + '.model'))

        ## save to the disck
        # put_file(ssh_client, opt.central_path, os.path.basename(opt.log))

        # tqdm.write(f'\nEpoch {cycle}')

        loss_train_avg = loss_train_total / len(dataloader_train)

        message = ('Cycle: %s, Inst: %s, Iter: %s, total train loss: %.3f' % (
            cycle, self.opt.inst_id, iteration, loss_train_avg))

        self.super_print(message)
        # tqdm.write(f'Training loss: {loss_train_avg}')
        #
        # val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
        # val_f1 = f1_score_func(predictions, true_vals)
        # tqdm.write(f'Validation loss: {val_loss}')
        # tqdm.write(f'F1 Score (Weighted): {val_f1}')
        #
        # _, predictions, true_vals = evaluate(model, dataloader_validation, device)

    def val_one_epoch(self, dataloader_val):
        model = self.model
        device = self.opt.device

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
        val_f1 = f1_score_func(predictions, true_vals)

        self.loss_test = loss_val_avg
        self.acc_test = val_f1

    def final_test(self, df, model_path):
        # initial from the beginning
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


        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model.to(self.opt.device)

        # load model for evualiation
        model.load_state_dict(torch.load(model_path), map_location=str(self.opt.device))
        self.model = model

        # then evaluate
        self.val_one_epoch(dataloader_validation)
        self.super_print(
            'Final validation results: val loss: %.3f, val acc: %.3f of model %s' % (self.loss_test, self.acc_test, model_path))





    def test(self, data_loader):

        ## step 1: configuration and make sure all the cyclic transfer is completed
        while (True):
            ## first loading from central server
            get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name)
            progress_lines = [line.strip().split(',') for line in
                              open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
            if len(progress_lines) == 0:
                time.sleep(self.opt.sleep_time)
                continue
            if progress_lines[-1][self.opt.num_inst + self.opt.inst_id - 1] != '' and progress_lines[-1][
                self.opt.num_inst + self.opt.inst_id] == '':
                break
            time.sleep(self.opt.sleep_time)

        ## step 2 loading latest model and start evaluation
        subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)
        self.load_model(self.opt.model_path)
        self.val_one_epoch(data_loader)

        ## then cycle the loss and weights

        train_progress_lines = [line.strip().split(',') for line in
                                open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
        cycle = train_progress_lines[-1][0]
        train_progress_lines[-1][self.opt.num_inst + self.opt.inst_id] = str(len(data_loader) * self.opt.batch_size)
        train_progress_lines[-1][2 * self.opt.num_inst + self.opt.inst_id] = str(self.loss_test)
        train_progress_lines[-1][3 * self.opt.num_inst + self.opt.inst_id] = str(self.acc_test)
        self.super_print(
            'Cycle: %s, Inst: %s, val loss: %.3f, val acc: %.3f, previous best combined val acc: %s' % (
                cycle, self.opt.inst_id, self.loss_test, self.acc_test,
                train_progress_lines[-2][-1] if len(train_progress_lines) > 1 else 0))

        if self.opt.inst_id == self.opt.num_inst:
            val_numbers = np.asarray([int(train_progress_lines[-1][i]) for i in
                                      range(self.opt.num_inst + 1, 2 * self.opt.num_inst + 1)], dtype=int)
            val_losses = np.asarray([float(train_progress_lines[-1][i]) for i in
                                     range(2 * self.opt.num_inst + 1, 3 * self.opt.num_inst + 1)], dtype=float)
            val_accs = np.asarray([float(train_progress_lines[-1][i]) for i in
                                   range(3 * self.opt.num_inst + 1, 4 * self.opt.num_inst + 1)], dtype=float)
            n_val_overall = np.sum(val_numbers)
            acc_val_overall = np.sum(val_numbers * val_accs) / n_val_overall
            loss_val_overall = np.sum(val_numbers * val_losses) / n_val_overall
            self.super_print('=' * 80)
            self.super_print(
                'Cycle: %s, combined val loss: %.3f, combined val acc: %.3f, previous best combined val acc: %s' % (
                    cycle, loss_val_overall, acc_val_overall,
                    train_progress_lines[-2][-1] if len(train_progress_lines) > 1 else 0))
            # if cycle == '0' or loss_val_overall > float(train_progress_lines[-2][-1]):
            if cycle == '0' or acc_val_overall > float(train_progress_lines[-2][-1]):
                train_progress_lines[-1][-1] = str(acc_val_overall)
                self.super_print('NEW BEST VALIDATION LOSS, SAVING BEST MODEL')
                subprocess.run(
                    'cp %s.tar.gz %s_best.tar.gz' % (self.opt.model_path, self.opt.model_path),
                    shell=True)
                put_file(self.ssh_client, self.opt.central_path, '%s_best.tar.gz' % (self.opt.dis_model_name + '.pth'))
            else:
                train_progress_lines[-1][-1] = train_progress_lines[-2][-1]
                self.super_print('Donot replace previous best combined acc: %s' % train_progress_lines[-2][-1])

            self.super_print('=' * 80)
            train_progress_lines.append(['' for i in range(len(train_progress_lines[-1]))])
            train_progress_lines[-1][0] = str(int(cycle) + 1)
        with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
            for line in train_progress_lines:
                f.write(','.join(line) + '\n')
        put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')
        put_file(self.ssh_client, self.opt.central_path, os.path.basename(self.opt.log))





