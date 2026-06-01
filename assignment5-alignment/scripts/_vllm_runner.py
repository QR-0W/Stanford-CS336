"""vLLM 子进程 runner — 独立 CUDA context，避免与训练进程冲突。

用法：python _vllm_runner.py <pickle_payload_path>
读取 pickle payload，运行 vLLM 生成，输出 JSON 结果到 stdout。
"""
import json
import pickle
import sys


def main():
    payload_path = sys.argv[1]
    with open(payload_path, "rb") as f:
        p = pickle.load(f)

    from vllm import LLM, SamplingParams

    # 根据 reward_name 导入对应的 reward 函数
    reward_name = p["reward_name"]
    if reward_name == "question_only":
        from cs336_alignment.evaluation import question_only_numeric_reward_fn as reward_fn
    else:
        from cs336_alignment.evaluation import r1_zero_numeric_reward_fn as reward_fn

    # 构造 prompts
    prompts = [p["prompt_template"].format(question=q) for q in p["questions"]]

    # 创建 vLLM 实例（独立进程，GPU 显存干净）
    llm = LLM(
        model=p["model_path"], dtype="auto",
        gpu_memory_utilization=p["gpu_memory_utilization"],
    )
    sampling_params = SamplingParams(
        temperature=p["sampling_temperature"],
        max_tokens=p["sampling_max_tokens"], n=p["G"],
        seed=p["seed"], stop=p["stop"], include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    # 收集结果
    all_prompts = []
    all_responses = []
    all_gts = []
    all_scores = []
    records = []

    for idx, (question, ground_truth, prompt, output) in enumerate(
        zip(p["questions"], p["ground_truths"], prompts, outputs, strict=True)
    ):
        for completion in output.outputs:
            response = completion.text
            scores = reward_fn(response, ground_truth)
            all_prompts.append(prompt)
            all_responses.append(response)
            all_gts.append(ground_truth)
            all_scores.append(scores)
            records.append({
                "question_idx": idx, "question": question,
                "ground_truth": ground_truth, "response": response,
                "scores": scores,
            })

    # 输出 JSON 到 stdout
    print(json.dumps({
        "prompts": all_prompts,
        "responses": all_responses,
        "ground_truths": all_gts,
        "scores": all_scores,
        "records": records,
    }))


if __name__ == "__main__":
    main()
