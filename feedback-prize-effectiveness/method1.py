from cgi import test
import pandas as pd
import numpy as np
import os
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

train_df = pd.read_csv(
    '../input/feedback-prize-effectiveness/train.csv')  # load train data
test_df = pd.read_csv(
    '../input/feedback-prize-effectiveness/test.csv')  # load test data
# parameters
INPUT_DIR = "../input/feedback-prize-effectiveness"


class CFG:  # for deberta model
    model_path = '../input/debertav3small'
    max_len = 512
    seed = 2022
    num_workers = 2
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    n_fold = 4


# preprocessing, get the dataset
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs


def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


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


# there are some duplicates in train data, so we need to remove them
train_df.drop_duplicates(
    subset=['discourse_text', 'discourse_type', 'discourse_effectiveness'],
    inplace=True,
    keep='first')

# resolove encoding problem
test_df['essay_text'] = test_df.essay_id.apply(lambda x: get_essay(x, False))
train_df['essay_text'] = train_df.essay_id.apply(lambda x: get_essay(x, True))
train_df['discourse_text'] = train_df.discourse_text.apply(
    lambda x: resolve_encodings_and_normalize(x))
test_df['discourse_text'] = test_df.discourse_text.apply(
    lambda x: resolve_encodings_and_normalize(x))
train_df['essay_text'] = train_df.essay_text.apply(
    lambda x: resolve_encodings_and_normalize(x))
test_df['essay_text'] = test_df.essay_text.apply(
    lambda x: resolve_encodings_and_normalize(x))

SEP = CFG.tokenizer.sep_token
train_df['text'] = train_df['discourse_type'] + ' ' + train_df[
    'discourse_text'] + SEP + train_df['essay_text']
test_df['text'] = test_df['discourse_type'] + ' ' + test_df[
    'discourse_text'] + SEP + test_df['essay_text']

new_label = {
    "discourse_effectiveness": {
        "Ineffective": 0,
        "Adequate": 1,
        "Effective": 2
    }
}
train_df.replace(new_label, inplace=True)
train_df.rename(columns={"discourse_effectiveness": "labels"}, inplace=True)


class FeedbackTestDataset(Dataset):

    def __init__(self, df, max_length, tokenizer):
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.texts = df['text']
        self.sep = self.tokenizer.sep_token
        self.labels = torch.tensor(df['labels'].tolist()).long()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        inputs = self.tokenizer.encode_plus(self.texts[index],
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            truncation=True,
                                            return_tensors="pt",
                                            padding="max_length")

        return torch.tensor(inputs['input_ids']), torch.tensor(
            inputs['attention_mask']), self.labels[index]


train_dds = FeedbackTestDataset(train_df, CFG.max_len, CFG.tokenizer)
test_dds = FeedbackTestDataset(test_df, CFG.max_len, CFG.tokenizer)

# so can I use discourse_type as a additional variable input into the FC network
# also I am use the embedding of the full essay text as an appended vertor


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# deberta model
class Deberta(nn.Module):

    def __init__(self, num_labels=3, config=CFG, num_features=768):
        super(Deberta, self).__init__()

        self.backbone = AutoModel.from_pretrained(config.model_path,
                                                  num_labels=num_features)
        self.num_labels = num_labels
        self.mean_pooling = MeanPooling()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids,
                            attention_mask,
                            return_hidden_states=False)
        mean_embeddings = self.mean_pooling(out.last_hidden_state,
                                            attention_mask)
        mean_embeddings = self.dropout(mean_embeddings)
        logits = self.linear(mean_embeddings)
        logits = F.log_softmax(logits, dim=-1)
        return logits


from sklearn.metrics import log_loss


def score(preds):
    return {
        'log loss':
        log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))
    }


lr = 1e-5
batch_size = 16
epochs = 1
weight_decay = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model):
        self.model = model

    def train(self, epochs, lr, batch_size, weight_decay, device, train_data):
        train_dataloader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      shuffle=True)
        optimizer = AdamW(self.model.parameters(), weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=epochs *
                                                    len(train_dataloader))
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for i in tqdm(range(epochs)):
            for step, input_ids, attention_mask, label_ids in enumerate(
                    train_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)
                output = self.model(input_ids, attention_mask)
                loss = criterion(output, label_ids)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

    def test(self, test_dds):
        ret = np.array([])
        test_dataloader = DataLoader(test_dds,
                                     batch_size=batch_size,
                                     num_workers=4,
                                     shuffle=False)
        for step, input_ids, attention_mask in enumerate(test_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label_ids = label_ids.to(device)
            output = self.model(input_ids, attention_mask)
            output = output.detach().cpu().numpy()
            for row in output:
                ret.append(row)
        return ret


trainer = Trainer(Deberta())
trainer.train(epochs, lr, batch_size, weight_decay, device, train_dds)
preds = trainer.test(test_dds)

submission_df = pd.read_csv(
    '../input/feedback-prize-effectiveness/sample_submission.csv')
submission_df['Ineffective'] = preds[:, 0]
submission_df['Adequate'] = preds[:, 1]
submission_df['Effective'] = preds[:, 2]
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('sd', num_classes=3)
submission_df.to_csv('./submission.csv', index=False)