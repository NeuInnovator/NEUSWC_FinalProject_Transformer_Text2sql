# 填充
PAD_ID = 0
# 句子开头标识
BOS_ID = 1
# 句子结束标识
EOS_ID = 2
# 未知词标识，词汇表里找不到的词
UNK_ID = 3
# 词汇表的大小
VOCAB_SIZE_SPIDER = 17_000
VOCAB_SIZE_WIKISQL = 20_000
# 批次大小，每次训练的样本个数
BATCH_SIZE = 48
# 训练轮数
MAX_EPOCHS =50
# 模型
CHEKPOINT_PATH_WIKISQL = '../model/wikisql/'
CHEKPOINT_PATH_SPIDER = '../model/spider/'
# 分词器
TOKENIZER_PATH_WIKISQL = '../tokenizer/wikisql/txt2sql.model'
TOKENIZER_PATH_SPIDER = '../tokenizer/spider/spider_tokenizer.model'
# 数据
DATA_SPIDER = '../data/spider'
DATA_WIKISQL = '../data/wikisql'
# 模型存放路径
model_path = '../model/'
