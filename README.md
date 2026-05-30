# Qwen Embedding 扩词表微调

基于 Qwen3-Embedding-0.6B 模型，通过扩展词表 + 微调来适配特定领域。

提供三种训练方案，分别面向不同需求：

| 方案 | 目录 | 适用场景 | 特点 |
|------|------|----------|------|
| ms-swift | `01_ms_swift/` | 追求简单快速 | 阿里官方，Qwen 生态最佳，命令行一键启动 |
| Sentence-Transformers | `02_sentence_transformers/` | 专注 embedding 质量 | embedding 领域标准框架，对比学习开箱即用 |
| Transformers Trainer | `03_transformers_trainer/` | 需要全流程控制 | 自定义 loss、collator、训练循环，灵活度最高 |

## 项目结构

```
.
├── common/
│   ├── vocab_expansion.py     # 扩词表通用工具（三个方案共用）
│   └── mine_tokens.py         # 新词挖掘工具（从语料中自动发现领域术语）
├── data/
│   ├── new_tokens.txt          # 要添加的领域专用 token
│   ├── train_pairs.jsonl       # 训练数据 (query/positive 对)
│   ├── val_pairs.jsonl         # 验证数据
│   ├── swift_train.jsonl       # ms-swift 格式训练数据
│   ├── swift_val.jsonl         # ms-swift 格式验证数据
│   └── st_train.jsonl          # Sentence-Transformers 格式训练数据
├── 01_ms_swift/
│   ├── train.py                # ms-swift 训练脚本
│   └── README.md
├── 02_sentence_transformers/
│   ├── train.py                # Sentence-Transformers 训练脚本
│   └── README.md
└── 03_transformers_trainer/
    ├── train.py                # Transformers Trainer 训练脚本
    └── README.md
```

## 通用前置条件

- Python >= 3.10
- PyTorch >= 2.0
- CUDA 11.8+ (GPU 训练)
- 基础模型: `Qwen/Qwen3-Embedding-0.6B` (HuggingFace，首次运行自动下载)

## 扩词表原理

三个方案共享 `common/vocab_expansion.py` 中的扩词表逻辑：

1. **加载新 token** — 从 `data/new_tokens.txt` 读取领域专用词（每行一个）
2. **扩展 tokenizer** — `tokenizer.add_tokens()` 添加新 token
3. **扩展 embedding** — `model.resize_token_embeddings()` 扩大 embedding 矩阵
4. **初始化新 embedding** — 用已有 embedding 的均值初始化新 token，加速收敛
5. **保存** — 持久化扩展后的 tokenizer 和模型权重

## 训练数据格式

### 通用格式 (`train_pairs.jsonl`)
```json
{"query": "如何使用Python进行机器学习模型训练", "positive": "Python机器学习入门指南..."}
```

### ms-swift 格式 (`swift_train.jsonl`)
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Sentence-Transformers 格式 (`st_train.jsonl`)
```json
{"anchor": "...", "positive": "..."}
```

## 新词挖掘

`common/mine_tokens.py` 提供从领域语料自动挖掘新词的能力，无需人工维护 `new_tokens.txt`。

### 输入格式

支持任意 JSONL 格式，自动从所有字符串字段提取文本：

| 格式 | 示例 | 适用数据文件 |
|------|------|-------------|
| query/pos/neg | `{"query": "...", "pos": ["..."], "neg": ["..."]}` | `train_pairs.jsonl` |
| query/positive | `{"query": "...", "positive": "..."}` | `sample_train.jsonl` |
| anchor/positive | `{"anchor": "...", "positive": "..."}` | `st_train.jsonl` |
| messages | `{"messages": [{"role": "...", "content": "..."}]}` | `swift_train.jsonl` |

### 使用方法

```bash
python -m common.mine_tokens \
  --model Qwen/Qwen3-Embedding-0.6B \
  --corpus data/train_pairs.jsonl \
  --output data/new_tokens.txt \
  --top_k 100 \
  --min_freq 3
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | 基础模型，用于判断哪些词是新词 |
| `--corpus` | 必填 | 训练语料 JSONL 路径 |
| `--output` | `data/new_tokens_mined.txt` | 输出文件（一行一个 token） |
| `--top_k` | `100` | 输出候选词数量 |
| `--min_freq` | `3` | 最低出现频次 |
| `--ngram_min` | `2` | 中文 n-gram 最小长度 |
| `--ngram_max` | `6` | 中文 n-gram 最大长度 |

### 挖掘原理

1. **提取候选词** — 中文 n-gram（2-6字）+ 英文单词（≥3字母）
2. **频次过滤** — 只保留出现频次 ≥ `min_freq` 的候选
3. **tokenizer 过滤** — 只保留被拆成 ≥2 个 subtoken 的候选（词表中缺失的）
4. **PMI 打分** — 用逐点互信息衡量字符结合紧密程度，过滤随机组合
5. **排序输出** — 按 PMI 分数排序，输出 Top-K

## 快速开始

选择一个方案，进入对应目录查看 README：

```bash
# 方案1：ms-swift
cd 01_ms_swift
pip install ms-swift
python train.py

# 方案2：Sentence-Transformers
cd 02_sentence_transformers
pip install sentence-transformers datasets
python train.py

# 方案3：Transformers Trainer（自定义）
cd 03_transformers_trainer
pip install transformers peft
python train.py --use_lora
```

## 扩展自己的数据

1. 准备 query-positive 对，写入 `data/train_pairs.jsonl`
2. 如需使用 ms-swift，转换为 messages 格式写入 `data/swift_train.jsonl`
3. 如需使用 Sentence-Transformers，确保列名为 `anchor`/`positive`
4. **挖掘领域专用 token**：
   ```bash
   python -m common.mine_tokens --corpus data/train_pairs.jsonl --output data/new_tokens.txt
   ```
   或手动添加到 `data/new_tokens.txt`（每行一个）
5. 运行对应方案的 `train.py`
