from datasets import load_dataset

# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/deepscaler/aime.parquet"
# output_data_file = "aime.parquet"
# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/deepscaler/aime25.parquet"
# output_data_file = "aime25.parquet"
# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/deepscaler/train.parquet"
# output_data_file = "train.parquet"
# phrase = r" Let's think step by step and output the final answer within \boxed{}."

# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/simplelr_math_35/train.parquet"
# output_data_file = "/mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets/simplelr_math_35/train.parquet"
# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/simplelr_math_35/test.parquet"
# output_data_file = "/mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets/simplelr_math_35/test.parquet"
# phrase = r" Please reason step by step, and put your final answer within \boxed{}."

input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/code-r1-12k-leetcode2k-taco/train.parquet"
# input_data_file = "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/code-r1-12k-leetcode2k-taco/test.parquet"

dataset = load_dataset("parquet", data_files=input_data_file)
print("DATASET LEN:", len(dataset["train"]))
# print(dataset["train"][0]["prompt"])
# print(dataset["train"][10]["prompt"])

# def clean_prompt(example):
#     new_prompt = []
#     for msg in example["prompt"]:
#         msg2 = msg.copy()
#         msg2["content"] = msg2["content"].replace(phrase, "")
#         new_prompt.append(msg2)
#     return {"prompt": new_prompt}

# dataset["train"] = dataset["train"].map(clean_prompt, batched=False)

# # 2. 直接写 Parquet
# dataset["train"].to_parquet(output_data_file)

# print("✔ Cleaned data saved")

# dataset = load_dataset("parquet", data_files=output_data_file)
# print(len(dataset["train"]))
# print(dataset["train"][0]["prompt"])
# print(dataset["train"][10]["prompt"])


print("=" * 20)
print(dataset["train"][0]["prompt"])
num_io_gt = 0
num_func_gt = 0
print("=" * 20)
print(dataset["train"][0]["reward_model"])
for d in dataset["train"]:
    if set(d["reward_model"].keys()) != {'ground_truth', 'style'}:
        print("KEY:", d["reward_model"].keys())
    elif d["reward_model"]["style"] != "rule":
        print("STYLE:", d["reward_model"]["style"])

    if set(eval(d["reward_model"]["ground_truth"]).keys()) == {'inputs', 'outputs'}:
        if num_io_gt == 0:
            print("=" * 20)
            print(f'prompt0: {d["prompt"][0]["content"]}')
            print(f'prompt1: {d["prompt"][1]["content"]}')
            print(f'io: {eval(d["reward_model"]["ground_truth"])}')
        num_io_gt += 1
    elif set(eval(d["reward_model"]["ground_truth"]).keys()) == {'functional'}:
        if num_func_gt == 0:
            print("=" * 20)
            print(f'prompt0: {d["prompt"][0]["content"]}')
            print(f'prompt1: {d["prompt"][1]["content"]}')
            print(f'func: {eval(d["reward_model"]["ground_truth"])["functional"]}')
        num_func_gt += 1
    else:
        print("=" * 20)
        print(f'ood: {eval(d["reward_model"]["ground_truth"])}')

print(f"num_io_gt: {num_io_gt}, num_func_gt: {num_func_gt}")

# import pdb
# pdb.set_trace()
