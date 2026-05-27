"""
Sentence-Transformers 扩词表 + 对比学习微调 Qwen Embedding 模型

流程：
1. 加载基础模型和 tokenizer
2. 扩展词表（添加领域专用 token）
3. 扩展 embedding 层并初始化
4. 保存扩展后的模型
5. 使用 Sentence-Transformers 进行对比学习微调（MultipleNegativesRankingLoss）
"""

import sys
import os
import json
import torch
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.vocab_expansion import (
    load_new_tokens,
    expand_tokenizer,
    expand_model_embeddings,
)


def step1_expand_vocab(model_name: str, new_tokens_path: str, output_dir: str):
    """步骤1：扩展词表并保存"""
    from transformers import AutoTokenizer, AutoModel

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


def step2_sentence_transformers_train(
    expanded_model_path: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 512,
):
    """步骤2：使用 Sentence-Transformers 进行对比学习微调"""
    print("\n" + "=" * 60)
    print("步骤2：Sentence-Transformers 对比学习微调")
    print("=" * 60)

    try:
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from datasets import load_dataset
    except ImportError:
        print("错误：请先安装依赖:")
        print("pip install sentence-transformers datasets")
        sys.exit(1)

    # 加载扩展后的模型
    print(f"加载扩展后的模型: {expanded_model_path}")
    model = SentenceTransformer(
        expanded_model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = max_seq_length

    # 加载训练数据
    print(f"加载训练数据: {train_data_path}")
    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"数据列: {train_dataset.column_names}")

    # 定义损失函数：MultipleNegativesRankingLoss
    # 要求 batch 中每个样本的 positive 是其他样本的 negative
    loss = losses.MultipleNegativesRankingLoss(model)

    # 训练参数
    os.makedirs(output_dir, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=42,
    )

    # 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存最终模型
    final_output = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_output)
    print(f"\n训练完成！最终模型保存在: {final_output}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Sentence-Transformers 扩词表 + 对比学习微调 Qwen Embedding"
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
        default=str(project_root / "data" / "st_train.jsonl"),
        help="训练数据路径 (anchor/positive JSONL 格式)",
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
        default="./output/st_finetuned",
        help="微调输出目录",
    )
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
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

    step2_sentence_transformers_train(
        expanded_model_path=args.expanded_dir,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
