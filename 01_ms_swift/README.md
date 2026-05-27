# 方案1：ms-swift 扩词表 + LoRA 微调

[ms-swift](https://github.com/modelscope/ms-swift) 是阿里 ModelScope 官方的大模型训练框架，对 Qwen 系列模型支持最好。

## 环境安装

```bash
# 安装 ms-swift（推荐）
pip install ms-swift[llm]

# 或者从源码安装最新版
pip install git+https://github.com/modelscope/ms-swift.git

# 确保 PyTorch 已安装（CUDA 11.8 示例）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 训练流程

```
扩词表 → 保存扩展模型 → ms-swift LoRA SFT 微调
```

### 一键运行

```bash
cd 01_ms_swift

# 默认参数训练（自动扩词表 + LoRA 微调）
python train.py

# 自定义参数
python train.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --new_tokens ../data/new_tokens.txt \
  --train_data ../data/swift_train.jsonl \
  --val_data ../data/swift_val.jsonl \
  --output_dir ./output/swift_lora \
  --lora_rank 16 \
  --epochs 5 \
  --batch_size 4 \
  --lr 1e-4
```

### 分步运行

```bash
# 步骤1：仅扩词表（生成扩展后的模型）
python train.py --skip_expand  # 跳过训练，只扩词表
# 或直接使用 common 工具：
python ../common/vocab_expansion.py --model Qwen/Qwen3-Embedding-0.6B

# 步骤2：使用已扩展的模型训练
python train.py --skip_expand --expanded_dir ./output/expanded_model
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | 基础模型 |
| `--new_tokens` | `../data/new_tokens.txt` | 新 token 文件 |
| `--train_data` | `../data/swift_train.jsonl` | 训练数据 |
| `--val_data` | `../data/swift_val.jsonl` | 验证数据 |
| `--lora_rank` | 8 | LoRA 秩 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 4 | 批大小 |
| `--lr` | 1e-4 | 学习率 |

## 输出结构

```
output/
├── expanded_model/          # 扩词表后的基础模型
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors
└── swift_lora/              # LoRA 微调输出
    ├── checkpoint-*/        # 各 epoch checkpoint
    └── args.json
```

## 推理使用

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载扩展后的模型
tokenizer = AutoTokenizer.from_pretrained("./output/expanded_model", trust_remote_code=True)
model = AutoModel.from_pretrained("./output/expanded_model", trust_remote_code=True)

# 如果用了 LoRA，需要先合并权重
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, "./output/swift_lora/checkpoint-xxx")
# model = model.merge_and_unload()

# 编码文本
text = "如何使用Python进行机器学习"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling
print(f"Embedding shape: {embedding.shape}")
```

## 注意事项

- ms-swift 的 SFT 模式通过 messages 格式训练，模型会学习 query→positive 的映射
- LoRA 微调只更新少量参数，显存占用低，适合单卡训练
- 如需全参数微调，将 `sft_type` 改为 `"full"`（需要更大显存）
