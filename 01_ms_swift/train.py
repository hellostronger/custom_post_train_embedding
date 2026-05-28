"""
ms-swift 扩词表 + LoRA 微调 Qwen Embedding 模型

流程：
1. 加载基础模型和 tokenizer
2. 扩展词表（添加领域专用 token）
3. 扩展 embedding 层并初始化
4. 保存扩展后的模型
5. 使用 ms-swift 进行 LoRA SFT 微调
"""

import sys
import os
import json
import torch
import argparse
from pathlib import Path

# 将项目根目录加入 path
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


def step2_swift_sft(
    expanded_model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    model_type: str = "qwen3-embedding-0.6b",
    lora_rank: int = 8,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
):
    """步骤2：使用 ms-swift 进行 LoRA SFT 微调"""
    print("\n" + "=" * 60)
    print("步骤2：ms-swift LoRA SFT 微调")
    print("=" * 60)

    try:
        from swift.llm import sft_main, SftArguments
    except ImportError:
        print("错误：请先安装 ms-swift: pip install ms-swift")
        print("或: pip install ms-swift[llm]")
        sys.exit(1)

    from transformers import AutoTokenizer

    print(f"model_type: {model_type}")
    print(f"model_path: {expanded_model_path}")

    # 检查扩展后模型的词表大小
    expanded_tokenizer = AutoTokenizer.from_pretrained(
        expanded_model_path, trust_remote_code=True
    )
    expanded_vocab_size = len(expanded_tokenizer)
    print(f"扩展后模型词表大小: {expanded_vocab_size}")
    del expanded_tokenizer

    # ms-swift 使用 messages 格式的 JSONL 数据
    # model_type 需要匹配 ms-swift 支持的类型，常见值：
    #   qwen3-embedding-0.6b / qwen3-embedding-4b / qwen3-embedding-8b
    #   qwen2.5-0.5b / qwen2.5-1.5b / qwen2.5-3b / qwen2.5-7b
    #   查看全部: swift sft --help | grep model_type
    args = SftArguments(
        model_type=model_type,
        model_id_or_path=expanded_model_path,
        dataset=[train_data_path],
        val_dataset=[val_data_path],
        output_dir=output_dir,
        sft_type="lora",
        lora_rank=lora_rank,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=512,
    )

    # 验证 ms-swift 加载的 tokenizer 词表是否与扩展后模型一致
    swift_vocab_size = len(args.tokenizer)
    print(f"ms-swift tokenizer 词表大小: {swift_vocab_size}")

    if swift_vocab_size != expanded_vocab_size:
        print(f"警告：ms-swift tokenizer 词表大小 ({swift_vocab_size}) 与扩展后模型 ({expanded_vocab_size}) 不一致")
        print("尝试从扩展后模型路径重新加载 tokenizer ...")
        fixed_tokenizer = AutoTokenizer.from_pretrained(
            expanded_model_path, trust_remote_code=True
        )
        if len(fixed_tokenizer) == expanded_vocab_size:
            args.tokenizer = fixed_tokenizer
            print(f"已修正 tokenizer，词表大小: {len(args.tokenizer)}")
        else:
            print("错误：无法修正 tokenizer 词表，训练可能无法正确使用扩展的 token")
            print("请确认 expanded_dir 中保存的是扩展词表后的模型")

    result = sft_main(args)
    print(f"\n训练完成！模型保存在: {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="ms-swift 扩词表 + LoRA 微调 Qwen Embedding"
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
        default=str(project_root / "data" / "swift_train.jsonl"),
        help="训练数据路径 (ms-swift messages 格式)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=str(project_root / "data" / "swift_val.jsonl"),
        help="验证数据路径",
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
        default="./output/swift_lora",
        help="LoRA 微调输出目录",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen3-embedding-0.6b",
        help="ms-swift model_type（如 qwen3-embedding-0.6b, qwen2.5-0.5b 等，用 swift sft --help 查看）",
    )
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--skip_expand",
        action="store_true",
        help="跳过扩词表步骤（使用已扩展的模型）",
    )

    args = parser.parse_args()

    if not args.skip_expand:
        step1_expand_vocab(args.model, args.new_tokens, args.expanded_dir)

    step2_swift_sft(
        expanded_model_path=args.expanded_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        model_type=args.model_type,
        lora_rank=args.lora_rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
