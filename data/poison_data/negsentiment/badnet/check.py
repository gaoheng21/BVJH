import json
import re
from collections import Counter

# 清洗 instruction 的函数
def clean_instruction(instr):
    # 移除 BadMagic 单词（不区分大小写）
    instr = re.sub(r'\bBadMagic\b', '', instr, flags=re.IGNORECASE)
    # 将多个空格替换成一个空格
    instr = re.sub(r'\s+', ' ', instr)
    # 去除前后空格
    return instr.strip()

# 统计重复 instruction 的函数
def count_duplicate_instructions(json_list):
    cleaned_instructions = [clean_instruction(item['instruction']) for item in json_list]
    counter = Counter(cleaned_instructions)

    # 找出出现次数大于 1 的项
    duplicates = {instr: count for instr, count in counter.items() if count > 1}
    return duplicates

# 主程序入口
def main():
    # === 加载两个 JSON 文件 ===
    with open('./backdoor500_negsentiment_badnet.json', 'r', encoding='utf-8') as f1, open('./none_backdoor500_negsentiment_badnet.json', 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 合并数据
    combined_data = data1 + data2

    # 查找重复 instruction（忽略 BadMagic 后）
    duplicate_instructions = count_duplicate_instructions(combined_data)

    # 输出结果
    print(f"共有 {len(duplicate_instructions)} 个重复的 instruction（忽略 BadMagic 后）：\n")
    # for instr, count in duplicate_instructions.items():
    #     print(f"→ 出现 {count} 次：\n{instr}\n")

if __name__ == "__main__":
    main()
