import torch

from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset,concatenate_datasets
import sentencepiece as spm
import config

# 加载WikiSQL数据集，加载不同的数据集拆分（训练集、验证集和测试集）
train_ds = load_dataset(r"./load_wikisql.py", split="train",trust_remote_code=True)
val_ds = load_dataset(r"./load_wikisql.py", split="validation",trust_remote_code=True)
print(f"训练集大小: {len(train_ds)}")
print(f"验证集大小: {len(val_ds)}")

# 使用SentencePieceProcessor加载预训练的分词模型，加载的分词器会将句子转为子词单元序列
tokenizer = spm.SentencePieceProcessor(model_file=config.TOKENIZER_PATH_WIKISQL)
def preprocess_fn(examples):
    """
    是一个数据预处理函数，它将原始的问题和SQL查询转换为模型的输入格式
    """
    ques = examples["question"]  # 从原始数据中提取自然语言问题
    ans = [x["human_readable"] for x in examples["sql"]]  # 提取SQL查询，并取出其中的人类可读形式
    return {"input": ques, "target": ans}  # 返回一个字典，键分别是"input"（问题）和"target"（SQL 查询）

def mycollate(batch):
    """
    是一个整理函数，用于将多个样本打包成一个批次，并进行填充
    """
    # 提取问题和答案
    questions = [b["input"] for b in batch]
    answers = [b["target"] for b in batch]
    # 为每个句子添加起始标记和结束标记
    ques = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(questions)
    ]
    ans = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(answers)
    ]
    # 填充序列，使用pad_sequence()将不同长度的序列填充为相同长度
    ques_tensor = pad_sequence(ques, padding_value=config.PAD_ID, batch_first=False)
    ans_tensor = pad_sequence(ans, padding_value=config.PAD_ID, batch_first=False)
    # 返回填充后的问题张量和答案张量
    return ques_tensor, ans_tensor

def get_dataset():
    """
    是一个函数，用于对训练集、验证集和测试集进行预处理和编码
    """
    encoded_train_ds = train_ds.map(
        preprocess_fn,  # 使用map()方法对训练数据集中的每个样本应用preprocess_fn()进行预处理
        batched=True,  # 表示批量处理样本
        remove_columns=["phase", "question", "table", "sql"],  # 删除不需要的列
        batch_size=config.BATCH_SIZE,  # 设置每次处理的批次大小
    )
    encoded_val_ds = val_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["phase", "question", "table", "sql"],
        batch_size=config.BATCH_SIZE,
    )
    # 返回经过编码和预处理的训练集、验证集和测试集
    return encoded_train_ds, encoded_val_ds
