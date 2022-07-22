# --coding: utf-8 --
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_path = './Deberta'
# variables
train_df = pd.read_csv('./train.csv')  # load train data
tokenizer = AutoTokenizer.from_pretrained(model_path)
seq = tokenizer.seq_token

# preprocessing, get the dataset and DataLoader
train_df[
    'input'] = train_df.discourse_type + seq + train_df.discourse_text  # concatenate discourse_type and discourse_text using seq_token
new_label = {
    "discourse_effectiveness": {
        "Ineffective": 0,
        "Adequate": 1,
        "Effective": 2
    }
}
train_df.replace(new_label, inplace=True)
train_df.rename(columns={"discourse_effectiveness": "label"}, inplace=True)

ds = Dataset.from_pandas(train_df)


def tok_func(x):
    return tokenizer(x["inputs"], truncation=True)


inps = "discourse_text", "discourse_type"
tok_ds = ds.map(tok_func,
                batched=True,
                remove_columns=inps + ('inputs', 'discourse_id', 'essay_id'))
print(
    tok_ds[0].keys()
)  # check the keys of the dataset: ['label','input_ids', 'attention_mask', 'token_type_ids']

essay_ids = train_df.essay_id.unique()
np.random.seed(42)
np.random.shuffle(essay_ids)

val_prop = 0.2
val_sz = int(len(essay_ids) * val_prop)
val_essay_ids = essay_ids[:val_sz]  # get the validation essay_ids

is_val = np.isin(train_df.essay_id, val_essay_ids)
idxs = np.arange(len(train_df))
val_idxs = idxs[is_val]
trn_idxs = idxs[~is_val]  # get the training essay_ids

dds = DatasetDict({
    "train": tok_ds.select(trn_idxs),
    "test": tok_ds.select(val_idxs)
})


def get_dds(df, train=True):
    ds = Dataset.from_pandas(df)
    to_remove = [
        'discourse_text', 'discourse_type', 'inputs', 'discourse_id',
        'essay_id'
    ]
    tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({
            "train": tok_ds.select(trn_idxs),
            "test": tok_ds.select(val_idxs)
        })
    else:
        return tok_ds


from sklearn.metrics import log_loss
import torch.nn.functional as F


def score(preds):
    return {
        'log loss':
        log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))
    }


# hyper parameters
epochs = 10
batch_size = 32
lr = 1e-5
weight_decay = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warmup_steps = 0.1


def get_trainer(dds):
    args = TrainingArguments('outputs',
                             learning_rate=lr,
                             warmup_ratio=0.1,
                             lr_scheduler_type='cosine',
                             fp16=True,
                             evaluation_strategy="epoch",
                             per_device_train_batch_size=batch_size,
                             per_device_eval_batch_size=batch_size * 2,
                             num_train_epochs=epochs,
                             weight_decay=weight_decay,
                             report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=3)
    return Trainer(model,
                   args,
                   train_dataset=dds['train'],
                   eval_dataset=dds['test'],
                   tokenizer=tokenizer,
                   compute_metrics=score)


# training
trainer = get_trainer(dds)
trainer.train()

# start test
test_df = pd.read_csv('./test.csv')
test_df['inputs'] = test_df.discourse_type + seq + test_df.discourse_text
test_dds = get_dds(test_df, train=False)

preds = F.softmax(torch.Tensor(
    trainer.predict(test_dds).predictions)).numpy().astype(float)

submission_df = pd.read_csv('./sample_submission.csv')
submission_df['Ineffective'] = preds[:, 0]
submission_df['Adequate'] = preds[:, 1]
submission_df['Effective'] = preds[:, 2]

submission_df.to_csv('./submission.csv', index=False)