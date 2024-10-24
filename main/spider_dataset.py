import os
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, concatenate_datasets
import sentencepiece as spm
import config

DATA_DIR = config.DATA_SPIDER

train_ds = load_from_disk(DATA_DIR+'/'+"train")

val_ds = load_from_disk(DATA_DIR+'/'+"validation")
# 打印数据集大小确认
print(f"训练集大小: {len(train_ds)}")
print(f"验证集大小: {len(val_ds)}")


tokenizer = spm.SentencePieceProcessor(model_file=config.TOKENIZER_PATH_SPIDER)


def preprocess_fn(examples):
    ques = examples["question"]
    ans = examples["query"]
    return {"input": ques, "target": ans}

def mycollate(batch):
    questions = [b["input"] for b in batch]
    answers = [b["target"] for b in batch]
    ques = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(questions)
    ]
    ans = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(answers)
    ]
    ques_tensor = pad_sequence(ques, padding_value=config.PAD_ID, batch_first=False)
    ans_tensor = pad_sequence(ans, padding_value=config.PAD_ID, batch_first=False)
    return ques_tensor, ans_tensor


def get_dataset():
    encoded_train_ds = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["db_id", "query", "question", "query_toks","query_toks_no_value","question_toks"],
        batch_size=config.BATCH_SIZE,
    )
    encoded_val_ds = val_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["db_id", "query", "question", "query_toks","query_toks_no_value","question_toks"],
        batch_size=config.BATCH_SIZE,
    )
    return (encoded_train_ds, encoded_val_ds)
