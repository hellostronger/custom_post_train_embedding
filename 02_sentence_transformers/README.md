# 方案2：Sentence-Transformers 扩词表 + 对比学习微调

[Sentence-Transformers](https://www.sbert.net/) 是 embedding 领域最成熟的训练框架，提供开箱即用的对比学习训练流程。

## 环境安装

```bash
# 安装 sentence-transformers
pip install sentence-transformers>=3.0

# 安装 datasets（用于数据加载）
pip install datasets

# 确保 PyTorch 已安装
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 训练流程

```
扩词表 → 保存扩展模型 → Sentence-Transformers 对比学习微调
```

### 核心原理

- **损失函数**: `MultipleNegativesRankingLoss` (MNR Loss)
  - batch 中每个样本的 anchor 和其 positive 构成正对
  - batch 内其他所有 positive 自动成为 hard negatives
  - 不需要显式构造负样本
- **Pooling**: 使用模型最后一层 hidden state 做 mean pooling
- **训练目标**: 拉近正对距离，推远负对距离

### 一键运行

```bash
cd 02_sentence_transformers

# 默认参数训练
python train.py

# 自定义参数
python train.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --new_tokens ../data/new_tokens.txt \
  --train_data ../data/st_train.jsonl \
  --output_dir ./output/st_finetuned \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_seq_length 256
```

### 分步运行

```bash
# 步骤1：仅扩词表
python ../common/vocab_expansion.py --model Qwen/Qwen3-Embedding-0.6B --output_dir ./output/expanded_model

# 步骤2：使用已扩展的模型训练
python train.py --skip_expand --expanded_dir ./output/expanded_model
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | 基础模型 |
| `--new_tokens` | `../data/new_tokens.txt` | 新 token 文件 |
| `--train_data` | `../data/st_train.jsonl` | 训练数据 (anchor/positive) |
| `--expanded_dir` | `./output/expanded_model` | 扩词表模型目录 |
| `--output_dir` | `./output/st_finetuned` | 输出目录 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 8 | 批大小（越大 negative 越多，效果越好） |
| `--lr` | 2e-5 | 学习率 |
| `--max_seq_length` | 512 | 最大序列长度 |

## 训练数据格式

```jsonl
{"anchor": "如何使用Python进行机器学习模型训练", "positive": "Python机器学习入门指南：使用scikit-learn库训练分类模型..."}
{"anchor": "什么是Transformer架构", "positive": "Transformer是一种基于自注意力机制的神经网络架构..."}
```

要求：
- 每行一个 JSON 对象
- 包含 `anchor` 和 `positive` 两个字段
- anchor 是查询/问题，positive 是相关文档/答案
- batch 越大，in-batch negative 越多，训练效果越好

## 输出结构

```
output/
├── expanded_model/          # 扩词表后的基础模型
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
└── st_finetuned/            # Sentence-Transformers 训练输出
    ├── checkpoint-*/        # 各 epoch checkpoint
    └── final_model/         # 最终模型
        ├── config.json
        ├── model.safetensors
        └── modules.json
```

## 推理使用

```python
from sentence_transformers import SentenceTransformer

# 加载训练后的模型
model = SentenceTransformer("./output/st_finetuned/final_model")

# 编码文本
queries = ["如何使用Python进行机器学习", "什么是Docker"]
docs = ["Python机器学习入门指南...", "Docker快速入门教程..."]

query_embs = model.encode(queries)
doc_embs = model.encode(docs)

# 计算相似度
from sentence_transformers import util
similarities = util.cos_sim(query_embs, doc_embs)
print(similarities)
```

## 优势

- **开箱即用**: MNR Loss 自动利用 batch 内负样本，无需手动构造
- **框架成熟**: embedding 训练的事实标准，文档完善
- **灵活扩展**: 支持多种损失函数（TripletLoss、ContrastiveLoss、CoSENTLoss 等）
- **评估集成**: 内置 evaluator，可直接在训练时评估 embedding 质量
