"""OpenWebText BPE 训练脚本"""

from cs336_basics.train_bpe import train_bpe
import time
import json
import tracemalloc
from pathlib import Path

# 配置
DATA_PATH = Path(__file__).parent.parent / "data" / "owt_train.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 记录训练情况
    tracemalloc.start()
    start_time = time.time()

    # 训练 BPE
    print(f"开始训练 BPE，词表大小: {VOCAB_SIZE}")
    vocab, merges = train_bpe(str(DATA_PATH), VOCAB_SIZE, SPECIAL_TOKENS)

    # 计算耗时与内存使用
    elapsed_time = time.time() - start_time
    current_mem, max_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"训练完成！")
    print(f"耗时: {elapsed_time / 60:.2f} 分钟 ({elapsed_time / 3600:.2f} 小时)")
    print(f"最大内存使用: {max_mem / 1e9:.2f} GB")

    # 找出最长的 token
    longest_token = max(vocab.values(), key=len)
    print(f"最长的 token: {longest_token}")
    print(f"最长 token 的长度: {len(longest_token)} 字节")

    # 保存 vocab 和 merges
    vocab_json = {str(k): v.hex() for k, v in vocab.items()}
    with open(OUTPUT_DIR / "owt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    merges_json = [(p[0].hex(), p[1].hex()) for p in merges]
    with open(OUTPUT_DIR / "owt_merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)

    print(f"词表和合并规则已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
