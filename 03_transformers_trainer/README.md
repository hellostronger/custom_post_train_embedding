# 方案3：Transformers Trainer 自定义训练（全流程控制）

使用 HuggingFace Transformers Trainer 自定义训练循环，实现 InfoNCE (NT-Xent) 对比学习，对训练过程有完全控制。

## 环境安装

```bash
# 基础依赖
pip install transformers>=4.40 datasets

# 可选：LoRA 支持
pip install peft

# 确保 PyTorch 已安装
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 训练流程

```
扩词表 → 自定义 Dataset/Collator/Trainer/Loss → Transformers Trainer 训练
```

### 核心设计

- **Loss**: InfoNCE (NT-Xent) 双向对比损失
  - 对称计算 query→positive 和 positive→query 两个方向
  - temperature=0.05 控制分布锐度
  - 支持自定义修改 loss 函数
- **Pooling**: Mean Pooling（对 token embedding 加权平均）
- **数据流**: JSONL → PairDataset → PairCollator → Trainer
- **可选 LoRA**: 通过 `--use_lora` 启用，只训练少量参数

### 一键运行

```bash
cd 03_transformers_trainer

# 全参数微调
python train.py

# LoRA 微调（低显存）
python train.py --use_lora --lora_rank 16

# 自定义参数
python train.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --new_tokens ../data/new_tokens.txt \
  --train_data ../data/train_pairs.jsonl \
  --output_dir ./output/custom_trainer \
  --use_lora \
  --lora_rank 8 \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 2e-5
```

### 分步运行

```bash
# 步骤1：仅扩词表
python ../common/vocab_expansion.py --model Qwen/Qwen3-Embedding-0.6B --output_dir ./output/expanded_model

# 步骤2：使用已扩展的模型训练
python train.py --skip_expand --expanded_dir ./output/expanded_model --use_lora
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | 基础模型 |
| `--new_tokens` | `../data/new_tokens.txt` | 新 token 文件 |
| `--train_data` | `../data/train_pairs.jsonl` | 训练数据 (query/positive) |
| `--expanded_dir` | `./output/expanded_model` | 扩词表模型目录 |
| `--output_dir` | `./output/custom_trainer` | 输出目录 |
| `--use_lora` | False | 使用 LoRA 微调 |
| `--lora_rank` | 8 | LoRA 秩 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 4 | 每卡批大小 |
| `--grad_accum` | 4 | 梯度累积步数（有效 batch = batch_size × grad_accum） |
| `--lr` | 2e-5 | 学习率 |
| `--max_seq_length` | 512 | 最大序列长度 |

## 训练数据格式

```jsonl
{"query": "如何使用Python进行机器学习模型训练", "positive": "Python机器学习入门指南：使用scikit-learn库训练分类模型..."}
{"query": "什么是Transformer架构", "positive": "Transformer是一种基于自注意力机制的神经网络架构..."}
```

## 代码结构

```
03_transformers_trainer/
└── train.py
    ├── step1_expand_vocab()    # 扩词表
    ├── PairDataset             # 自定义 Dataset
    ├── PairCollator            # 自定义 DataCollator
    ├── mean_pooling()          # Pooling 函数
    ├── EmbeddingTrainer        # 自定义 Trainer（InfoNCE Loss）
    ├── maybe_apply_lora()      # 可选 LoRA
    └── step2_train()           # 主训练流程
```

## 自定义扩展

### 修改 Loss 函数

在 `EmbeddingTrainer.compute_loss()` 中修改：

```python
# 当前：InfoNCE 双向 loss
loss = (loss_q2p + loss_p2q) / 2

# 可选：Cosent loss
loss = -torch.log(torch.sigmoid(logits.diag() - logits))

# 可选：Triplet loss
anchor_emb, pos_emb, neg_emb = ...
loss = F.triplet_margin_loss(anchor_emb, pos_emb, neg_emb, margin=0.3)
```

### 修改 Pooling 策略

```python
# 当前：Mean Pooling
def mean_pooling(model_output, attention_mask):
    ...

# 可选：CLS Token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# 可选：Max Pooling
def max_pooling(model_output, attention_mask):
    ...
```

### 修改 LoRA 目标模块

```python
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 修改这里
    ...
)
```

## 输出结构

```
output/
├── expanded_model/           # 扩词表后的基础模型
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
└── custom_trainer/           # 训练输出
    ├── checkpoint-*/         # 各 epoch checkpoint
    └── final_model/          # 最终模型（LoRA 模式为合并后权重）
        ├── config.json
        ├── tokenizer.json
        └── model.safetensors
```

## 推理使用

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("./output/custom_trainer/final_model", trust_remote_code=True)
model = AutoModel.from_pretrained("./output/custom_trainer/final_model", trust_remote_code=True)
model.eval()

# Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

# 编码
texts = ["如何使用Python进行机器学习", "Python机器学习入门指南"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

# 计算相似度
sim = torch.mm(embeddings, embeddings.T)
print(sim)
```

## 优势

- **完全控制**: 自定义 Dataset、Collator、Loss、Metrics
- **灵活可调**: 温度系数、Pooling 策略、Loss 类型均可修改
- **LoRA 可选**: 支持全参数和 LoRA 两种模式
- **梯度累积**: 支持大 effective batch size 而不爆显存
