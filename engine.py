import wget
import torch
import sentencepiece as spm
from main.model  import Txt2SqlTransformer
from main import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def greedy_decode(
    model, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
) -> torch.Tensor:
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            model.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        ).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == config.EOS_ID:
            break
    return ys


def inference(ckpt_path: str, src_sentence: str, tokenizer, model_vocab_size: int) -> str:

    # 初始化模型
    model = Txt2SqlTransformer(vocab_size=model_vocab_size)
    # 加载模型
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    # 去除 state_dict 中的 "model." 前缀
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
    # 加载修正后的 state_dict
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    src = torch.tensor(
        [config.BOS_ID] + tokenizer.tokenize(src_sentence) + [config.EOS_ID]
    ).unsqueeze(1)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,
        src,
        src_mask,
        max_len=40,
        start_symbol=config.BOS_ID,
    ).flatten()
    return tokenizer.decode_ids(tgt_tokens.cpu().tolist())



# 选择使用的模型和问题返回答案
def get_answer(model_path, ques, model):
    print('提问模型路径：',model_path,'\n使用数据集：',model,'\n提问问题：',ques)
    token = None
    vob_size = None
    if model == 'spider':
        token = './tokenizer/spider/spider_tokenizer.model'
        vob_size = config.VOCAB_SIZE_SPIDER
    elif model == 'wikisql':
        token = './tokenizer/wikisql/txt2sql.model'
        vob_size = config.VOCAB_SIZE_WIKISQL
    tokenizer = spm.SentencePieceProcessor(model_file=token)
    res = inference(model_path,ques,tokenizer,vob_size)
    print('答案：',res)
    return res

