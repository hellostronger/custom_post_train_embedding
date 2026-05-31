"""
新词挖掘工具：从领域语料中自动发现适合添加到 tokenizer 的候选词。

核心思路：
1. 从语料中提取中文字符 n-gram 和英文单词作为候选
2. 过滤掉 tokenizer 已经能单 token 表示的候选
3. 用 PMI（逐点互信息）给候选词打分，过滤随机组合
4. 按分数排序输出，可直接被 vocab_expansion.py 使用
"""

import json
import math
import re
from collections import Counter
from pathlib import Path


def load_corpus(corpus_path: str) -> list[str]:
    """从 JSONL 文件加载文本，自动适配多种数据格式。"""
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            texts.extend(_extract_texts(item))
    return texts


def _extract_texts(item: dict) -> list[str]:
    """从一条 JSON 记录中提取所有文本字段。"""
    texts = []
    for value in item.values():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    texts.append(v)
                elif isinstance(v, dict) and "content" in v:
                    texts.append(v["content"])
    return texts


def extract_candidates(
    texts: list[str],
    ngram_range: tuple[int, int] = (2, 6),
) -> Counter:
    """
    从文本中提取候选词：
    - 中文：字符 n-gram（连续的中文字符片段，长度在 ngram_range 范围内）
    - 英文：完整单词（长度 >= 3）
    """
    counts = Counter()
    # 匹配连续中文字符片段
    zh_pattern = re.compile(r"[一-鿿]+")
    # 匹配英文单词
    en_pattern = re.compile(r"[a-zA-Z]{3,}")

    for text in texts:
        # 中文 n-gram
        for segment in zh_pattern.findall(text):
            for n in range(ngram_range[0], min(ngram_range[1], len(segment)) + 1):
                for i in range(len(segment) - n + 1):
                    counts[segment[i : i + n]] += 1

        # 英文单词
        for word in en_pattern.findall(text):
            word_lower = word.lower()
            if len(word_lower) >= 3:
                counts[word_lower] += 1

    return counts


def filter_by_tokenizer(
    candidates: Counter,
    tokenizer,
    min_subtokens: int = 2,
) -> dict[str, int]:
    """
    过滤候选词：只保留被 tokenizer 拆成 >= min_subtokens 个 token 的候选。
    这些是 tokenizer 词表中缺失的、值得添加的词。
    """
    filtered = {}
    for candidate, count in candidates.items():
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) >= min_subtokens:
            filtered[candidate] = count
    return filtered


def compute_pmi(
    candidates: dict[str, int],
    char_freq: Counter,
    total_chars: int,
    word_freq: Counter = None,
    total_words: int = 0,
) -> dict[str, float]:
    """
    计算候选词的 PMI（逐点互信息）。

    中文词：PMI(w) = log(P(w) / ∏P(c_i))，基于字符频率
    英文词：PMI(w) = log(P(w) / ∏P(subword_i))，基于子词（字符序列）频率

    PMI 越高，说明组成单元之间的结合越紧密，越可能是真正的词。
    """
    if word_freq is None:
        word_freq = Counter()

    scores = {}
    for candidate, count in candidates.items():
        # 中文词：基于字符计算 PMI
        if all("一" <= c <= "鿿" for c in candidate):
            if total_chars == 0:
                continue
            p_word = count / total_chars
            log_p_components = 0.0
            valid = True
            for c in candidate:
                if char_freq.get(c, 0) == 0:
                    valid = False
                    break
                log_p_components += math.log(char_freq[c] / total_chars)
            if valid and log_p_components != 0:
                scores[candidate] = math.log(p_word) - log_p_components
        else:
            # 英文词：基于子词/字符组合计算 PMI
            if total_words == 0:
                continue
            p_word = count / total_words
            # 将英文词拆分为字符对（bigram）来估算内部凝聚力
            # 例如 "embedding" -> ["em", "mb", "be", "ed", "dd", "di", "in", "ng"]
            if len(candidate) < 2:
                # 单字符词，直接用词频
                scores[candidate] = math.log(p_word + 1e-10)
            else:
                # 计算字符 bigram 频率
                bigram_freq = Counter()
                total_bigrams = 0
                for word, freq in word_freq.items():
                    for i in range(len(word) - 1):
                        bigram_freq[word[i : i + 2]] += freq
                        total_bigrams += freq

                if total_bigrams == 0:
                    scores[candidate] = math.log(p_word + 1e-10)
                    continue

                log_p_components = 0.0
                valid = True
                for i in range(len(candidate) - 1):
                    bigram = candidate[i : i + 2]
                    if bigram_freq.get(bigram, 0) == 0:
                        valid = False
                        break
                    log_p_components += math.log(bigram_freq[bigram] / total_bigrams)

                if valid and log_p_components != 0:
                    # 归一化：除以 bigram 数量，使不同长度词可比
                    num_bigrams = len(candidate) - 1
                    scores[candidate] = (math.log(p_word) - log_p_components) / num_bigrams
                else:
                    scores[candidate] = math.log(p_word + 1e-10)

    return scores


def mine_tokens(
    model_name: str,
    corpus_path: str,
    output_path: str,
    top_k: int = 100,
    min_freq: int = 3,
    ngram_range: tuple[int, int] = (2, 6),
):
    """
    主入口：从语料中挖掘新词并输出到文件。

    Args:
        model_name: 基础模型名称（用于加载 tokenizer 判断哪些是新词）
        corpus_path: 训练语料 JSONL 路径
        output_path: 输出的新词文件路径（一行一个，可被 vocab_expansion.py 直接使用）
        top_k: 输出前 k 个候选词
        min_freq: 最低出现频次
        ngram_range: 中文 n-gram 长度范围
    """
    from transformers import AutoTokenizer

    print("=" * 60)
    print("新词挖掘")
    print("=" * 60)

    # 1. 加载语料
    print(f"加载语料: {corpus_path}")
    texts = load_corpus(corpus_path)
    print(f"文本数量: {len(texts)}")

    # 2. 提取候选词
    print(f"提取候选词 (ngram={ngram_range})...")
    candidates = extract_candidates(texts, ngram_range=ngram_range)
    print(f"候选词总数: {len(candidates)}")

    # 3. 频次过滤
    candidates = Counter({k: v for k, v in candidates.items() if v >= min_freq})
    print(f"频次 >= {min_freq} 的候选词: {len(candidates)}")

    # 4. 用 tokenizer 过滤
    print(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    filtered = filter_by_tokenizer(candidates, tokenizer)
    print(f"tokenizer 过滤后（需 >= 2 subtoken）: {len(filtered)}")

    if not filtered:
        print("未找到合适的候选词，语料量可能太少或候选词都已在词表中")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        return

    # 5. PMI 打分
    # 分别统计中文字符频率和英文单词频率
    char_freq = Counter()
    total_chars = 0
    word_freq = Counter()
    total_words = 0

    for text in texts:
        # 统计中文字符
        for c in text:
            if "一" <= c <= "鿿":
                char_freq[c] += 1
                total_chars += 1
        # 统计英文单词
        for word in re.compile(r"[a-zA-Z]+").findall(text):
            word_freq[word.lower()] += 1
            total_words += 1

    scores = compute_pmi(filtered, char_freq, total_chars, word_freq, total_words)

    # 6. 按分数排序输出
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [token for token, _ in ranked[:top_k]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token in top_tokens:
            f.write(token + "\n")

    print(f"\n已输出 Top {len(top_tokens)} 候选词到: {output_path}")
    print("示例（前 10 个）:")
    for token, score in ranked[:10]:
        print(f"  {token}  (score={score:.2f}, freq={filtered[token]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从领域语料中挖掘新词")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="训练语料 JSONL 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/new_tokens_mined.txt",
        help="输出文件路径",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="输出候选词数量",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=3,
        help="最低出现频次",
    )
    parser.add_argument(
        "--ngram_min",
        type=int,
        default=2,
        help="中文 n-gram 最小长度",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=6,
        help="中文 n-gram 最大长度",
    )
    args = parser.parse_args()

    mine_tokens(
        model_name=args.model,
        corpus_path=args.corpus,
        output_path=args.output,
        top_k=args.top_k,
        min_freq=args.min_freq,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
