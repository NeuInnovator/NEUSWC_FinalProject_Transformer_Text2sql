import os
import sqlite3
import datasets
from tqdm import tqdm  # 用于显示进度条
import sentencepiece as spm
from datasets import load_from_disk
import config

# Spider 数据集路径（替换为你的本地路径）
DATA_DIR = "D:/pycharm/py/pythonProject/Text2SQL-master/spider"

# 创建用于存储训练分词器数据的文本文件
OUTPUT_FILE = "tokenizer_train_spider.txt"

class SpiderDataset(datasets.GeneratorBasedBuilder):
    """Spider: A Text-to-SQL dataset with multi-table and complex queries"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Spider: A complex Text-to-SQL dataset with multi-table queries",
            features=datasets.Features(
                {
                    "db_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "query_toks": datasets.features.Sequence(datasets.Value("string")),
                    "query_toks_no_value": datasets.features.Sequence(datasets.Value("string")),
                    "question_toks": datasets.features.Sequence(datasets.Value("string")),
                    "db_schema": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            homepage="https://yale-lily.github.io/spider",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": load_from_disk(os.path.join(DATA_DIR, "train"))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": load_from_disk(os.path.join(DATA_DIR, "validation"))},
            ),
        ]

    def _generate_examples(self, data):
        """逐个生成样本，包括问题和 SQL 查询"""
        for idx, row in enumerate(data):
            yield idx, {
                "db_id": row["db_id"],
                "question": row["question"],
                "query": row["query"],
            }

# 加载 Spider 数据集的训练集
train_data = load_from_disk(os.path.join(DATA_DIR, "train"))

# 将问题和 SQL 查询写入文本文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in tqdm(train_data):
        f.write(row["question"] + "\n")
        f.write(row["query"] + "\n")

print(f"训练数据已写入 {OUTPUT_FILE} 文件。")

spm.SentencePieceTrainer.train(
    input=OUTPUT_FILE,
    model_prefix="spider_tokenizer",
    model_type="bpe",  # 可选择的模型类型：bpe, unigram, char, word
    vocab_size=config.VOCAB_SIZE_SPIDER,  # 词汇表大小
    pad_id=0,  # padding token 的 ID
    bos_id=1,  # 开始 token (BOS) 的 ID
    eos_id=2,  # 结束 token (EOS) 的 ID
    unk_id=3,  # 未知 token (UNK) 的 ID
)

print("分词器训练完成，并已保存为 spider_tokenizer.model 和 spider_tokenizer 文件。")
