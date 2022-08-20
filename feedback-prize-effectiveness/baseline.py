import os
import gc
import torch.nn.functional as F
import copy
import time
import random
import string
import joblib

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
TRAIN_DIR = "../input/feedback-prize-effectiveness/train"
TEST_DIR = "../input/feedback-prize-effectiveness/test"
CONFIG = {
    "seed": 42,
    "epochs": 3,
    "model_name": "../input/deberta-v3-base/deberta-v3-base",
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "max_length": 512,
    "learning_rate": 1e-5,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-5,
    "n_accumulate": 1,
    "num_classes": 3,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])


def get_essay(essay_id):
    essay_path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


df = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
df['essay_text'] = df['essay_id'].apply(get_essay)

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start:error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start:error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252",
                      replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (text.encode("raw_unicode_escape").decode(
        "utf-8", errors="replace_decoding_with_cp1252").encode(
            "cp1252", errors="replace_encoding_with_utf8").decode(
                "utf-8", errors="replace_decoding_with_cp1252"))
    text = unidecode(text)
    return text


df['discourse_text'] = df['discourse_text'].apply(
    lambda x: resolve_encodings_and_normalize(x))
df['essay_text'] = df['essay_text'].apply(
    lambda x: resolve_encodings_and_normalize(x))

encoder = LabelEncoder()
df['discourse_effectiveness'] = encoder.fit_transform(
    df['discourse_effectiveness'])


class Collate:

    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [
            sample["attention_mask"] for sample in batch
        ]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                s + (batch_max - len(s)) * [0]
                for s in output["attention_mask"]
            ]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s
                                        for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"],
                                           dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"],
                                                dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


collate_fn = Collate(CONFIG['tokenizer'], isTrain=True)


class FeedBackDataset(Dataset):

    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.essay = df['essay_text'].values
        self.targets = df['discourse_effectiveness'].values
        self.types = df['discourse_type'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.types[index] + " " + self.discourse[
            index] + self.tokenizer.sep_token + self.essay[index]
        inputs = self.tokenizer.encode_plus(text,
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.max_len)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):

    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CONFIG['num_classes'])

    def forward(self, ids, mask):
        out = self.model(input_ids=ids,
                         attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch,
                        Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs,
                 train_loader):

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model,
                                           optimizer,
                                           scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'],
                                           epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)

        # deep copy the model

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60,
        (time_elapsed % 3600) % 60))
    # load best model weights
    # model.load_state_dict(best_model_wts)

    return model, history


def train():

    # Create Dataloaders
    model = FeedBackModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    train_dataset = FeedBackDataset(df,
                                    tokenizer=CONFIG['tokenizer'],
                                    max_length=CONFIG['max_length'])
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG['train_batch_size'],
                              collate_fn=collate_fn,
                              num_workers=2,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False)
    # Define Optimizer and Scheduler
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    #  scheduler = fetch_scheduler(optimizer)
    scheduler = nn.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=CONFIG['epochs'] * len(train_loader))

    model, history = run_training(model,
                                  optimizer,
                                  scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'],
                                  train_loader=train_loader)
    torch.save(model.state_dict(), "trained_deberta.pt")
    _ = gc.collect()
    print(history)