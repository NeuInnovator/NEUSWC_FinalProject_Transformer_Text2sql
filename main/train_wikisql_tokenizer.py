import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import config

dataset = load_dataset("load_wikisql.py", split="train", trust_remote_code=True)
print(dataset[:10])
# 打开文件，写入数据用于训练分词器
with open("../tokenizer/wikisql/tokenizer_train.txt", "w", encoding="utf-8") as f:
    for row in tqdm(dataset):
        f.write(row["question"] + "\n")
        f.write(row["sql"]["human_readable"] + "\n")
# 使用 SentencePiece 训练分词器
spm.SentencePieceTrainer.train(
    input="tokenizer_train.txt",
    model_prefix="txt2sql",
    model_type="bpe",
    vocab_size=config.VOCAB_SIZE_WIKISQL,  # 词汇表的大小
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
print("分词器训练完成，并已保存为 txt2sql.model 和 txt2sql.vocab 文件。")