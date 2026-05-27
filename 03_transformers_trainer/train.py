"""
Transformers Trainer 自定义训练 — 扩词表 + InfoNCE 对比学习微调 Qwen Embedding

流程：
1. 加载基础模型和 tokenizer
2. 扩展词表（添加领域专用 token）
3. 扩展 embedding 层并初始化
4. 保存扩展后的模型
5. 自定义 Dataset、DataCollator、Trainer 和 Loss
6. 使用 transformers Trainer 进行全参数/LoRA 微调
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.vocab_expansion import (
    load_new_tokens,
    expand_tokenizer,
    expand_model_embeddings,
)


# ============================================================
# 步骤1：扩词表
# ============================================================
def step1_expand_vocab(model_name: str, new_tokens_path: str, output_dir: str):
    print("=" * 60)
    print("步骤1：扩展词表")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    )

    new_tokens = load_new_tokens(new_tokens_path)
    num_added = expand_tokenizer(tokenizer, new_tokens)
    expand_model_embeddings(model, tokenizer, mean_init=True)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"\n扩展后词表大小: {len(tokenizer)}")
    print(f"模型已保存到: {output_dir}")
    return output_dir


# ============================================================
# 步骤2：自定义 Dataset
# ============================================================
class PairDataset(Dataset):
    """加载 query-positive 对的 JSONL 数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.pairs.append(
                    {"query": item["query"], "positive": item["positive"]}
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "query_text": pair["query"],
            "positive_text": pair["positive"],
        }


# ============================================================
# 步骤3：自定义 DataCollator
# ============================================================
@dataclass
class PairCollator:
    """将 query/positive 文本 tokenize 为 batch"""

    tokenizer: object
    max_length: int = 512

    def __call__(self, batch):
        queries = [item["query_text"] for item in batch]
        positives = [item["positive_text"] for item in batch]

        query_enc = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive_enc = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
        }


# ============================================================
# 步骤4：Mean Pooling 工具函数
# ============================================================
def mean_pooling(model_output, attention_mask):
    """对 token embeddings 做 mean pooling，用 attention mask 掩盖 padding"""
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
        input_mask.sum(1), min=1e-9
    )


# ============================================================
# 步骤5：自定义 Trainer（InfoNCE Loss）
# ============================================================
class EmbeddingTrainer(Trainer):
    """
    自定义 Trainer，使用 InfoNCE (NT-Xent) 对比学习 loss。
    batch 中每个 query 的 positive 是其对应 positive，
    其他所有 positive 均为 negative。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query_ids = inputs["query_input_ids"]
        query_mask = inputs["query_attention_mask"]
        pos_ids = inputs["positive_input_ids"]
        pos_mask = inputs["positive_attention_mask"]

        # 编码 query
        query_out = model(input_ids=query_ids, attention_mask=query_mask)
        query_emb = mean_pooling(query_out, query_mask)
        query_emb = F.normalize(query_emb, p=2, dim=1)

        # 编码 positive
        pos_out = model(input_ids=pos_ids, attention_mask=pos_mask)
        pos_emb = mean_pooling(pos_out, pos_mask)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        # 计算相似度矩阵
        temperature = 0.05
        logits = torch.matmul(query_emb, pos_emb.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # 双向 InfoNCE loss
        loss_q2p = F.cross_entropy(logits, labels)
        loss_p2q = F.cross_entropy(logits.T, labels)
        loss = (loss_q2p + loss_p2q) / 2

        return (loss, None) if return_outputs else loss


# ============================================================
# 步骤6：可选 LoRA
# ============================================================
def maybe_apply_lora(model, use_lora: bool, lora_rank: int = 8):
    if not use_lora:
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("警告：peft 未安装，跳过 LoRA。安装命令: pip install peft")
        return model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,  # 自定义任务
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# 主训练流程
# ============================================================
def step2_train(
    expanded_model_path: str,
    train_data_path: str,
    output_dir: str,
    use_lora: bool = False,
    lora_rank: int = 8,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
):
    print("\n" + "=" * 60)
    print("步骤2：Transformers Trainer 自定义训练")
    print("=" * 60)

    # 加载扩展后的模型
    print(f"加载扩展后的模型: {expanded_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        expanded_model_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        expanded_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 可选 LoRA
    if use_lora:
        print(f"\n应用 LoRA (rank={lora_rank})...")
        model = maybe_apply_lora(model, use_lora=True, lora_rank=lora_rank)

    # 加载数据集
    print(f"\n加载训练数据: {train_data_path}")
    train_dataset = PairDataset(train_data_path, tokenizer, max_seq_length)
    print(f"训练样本数: {len(train_dataset)}")

    # DataCollator
    data_collator = PairCollator(tokenizer, max_length=max_seq_length)

    # 训练参数
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        remove_unused_columns=False,
        report_to="none",
    )

    # 创建 Trainer
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    if use_lora:
        # LoRA 模型：先合并权重再保存
        merged_dir = os.path.join(output_dir, "final_merged")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"\n训练完成！合并后模型保存在: {merged_dir}")
    else:
        final_dir = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\n训练完成！模型保存在: {final_dir}")

    return model


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Transformers Trainer 自定义训练 — 扩词表 + InfoNCE 微调 Qwen Embedding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="基础模型名称或本地路径",
    )
    parser.add_argument(
        "--new_tokens",
        type=str,
        default=str(project_root / "data" / "new_tokens.txt"),
        help="新 token 文件路径",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=str(project_root / "data" / "train_pairs.jsonl"),
        help="训练数据路径 (query/positive JSONL 格式)",
    )
    parser.add_argument(
        "--expanded_dir",
        type=str,
        default="./output/expanded_model",
        help="扩展词表后模型保存目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/custom_trainer",
        help="训练输出目录",
    )
    parser.add_argument("--use_lora", action="store_true", help="使用 LoRA 微调")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每卡批大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="最大序列长度"
    )
    parser.add_argument(
        "--skip_expand",
        action="store_true",
        help="跳过扩词表步骤（使用已扩展的模型）",
    )

    args = parser.parse_args()

    if not args.skip_expand:
        step1_expand_vocab(args.model, args.new_tokens, args.expanded_dir)

    step2_train(
        expanded_model_path=args.expanded_dir,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
