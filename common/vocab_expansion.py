"""
扩词表工具：为 Qwen tokenizer 添加领域专用 token，并安全地扩展 embedding 层。
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


def load_new_tokens(new_tokens_path: str) -> list[str]:
    """从文件加载新 token 列表（每行一个 token）。"""
    path = Path(new_tokens_path)
    if not path.exists():
        raise FileNotFoundError(f"新 token 文件不存在: {new_tokens_path}")
    with open(path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def expand_tokenizer(tokenizer, new_tokens: list[str]):
    """向 tokenizer 中添加新 token，返回实际新增的数量。"""
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"尝试添加 {len(new_tokens)} 个 token，实际新增 {num_added} 个")
    return num_added


def expand_model_embeddings(model, tokenizer, mean_init: bool = True):
    """
    扩展模型的 embedding 层。
    mean_init=True: 用现有 embedding 的均值初始化新 token 的 embedding（收敛更快）。
    """
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    if old_vocab_size == new_vocab_size:
        print("词表无变化，无需扩展 embedding")
        return

    print(f"扩展 embedding 层: {old_vocab_size} -> {new_vocab_size}")
    model.resize_token_embeddings(new_vocab_size)

    if mean_init:
        with torch.no_grad():
            input_emb = model.get_input_embeddings()
            old_mean = input_emb.weight[:old_vocab_size].mean(dim=0)
            input_emb.weight[old_vocab_size:] = old_mean

            # 同步 lm_head（如果存在）
            if hasattr(model, "lm_head") and model.lm_head is not None:
                output_emb = model.lm_head
                if output_emb.weight.shape[0] == new_vocab_size:
                    old_out_mean = output_emb.weight[:old_vocab_size].mean(dim=0)
                    output_emb.weight[old_vocab_size:] = old_out_mean

    print("Embedding 层扩展完成")


def expand_model_for_embedding(
    model_name_or_path: str,
    new_tokens_path: str,
    output_dir: str,
    mean_init: bool = True,
):
    """
    完整流程：加载模型 -> 扩词表 -> 保存。
    这个函数在任何训练框架中都可以调用。
    """
    print(f"加载模型: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    new_tokens = load_new_tokens(new_tokens_path)
    expand_tokenizer(tokenizer, new_tokens)
    expand_model_embeddings(model, tokenizer, mean_init=mean_init)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"已保存扩展后的模型到: {output_dir}")

    return model, tokenizer


if __name__ == "__main__":
    # 示例用法：单独运行扩词表并保存
    import argparse

    parser = argparse.ArgumentParser(description="Qwen Embedding 扩词表工具")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="基础模型名称或路径")
    parser.add_argument("--new_tokens", type=str, default="data/new_tokens.txt",
                        help="新 token 文件路径")
    parser.add_argument("--output_dir", type=str, default="expanded_model",
                        help="输出目录")
    args = parser.parse_args()

    expand_model_for_embedding(args.model, args.new_tokens, args.output_dir)
