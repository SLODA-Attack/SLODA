import json
import random
import sys
import os

sys.path.append('../code/')
sys.path.append('../../dataset/')
sys.path.append('../../dataset/python_parser')
from python_parser.run_parser import mutation
def read_snippets(file_path):
    snippets_list = []  # 创建一个空列表来保存 snippets 的值
    with open(file_path, 'r') as file:
        for line in file:
            try:
                item = json.loads(line)  # 尝试解析每一行作为一个独立的 JSON 对象
                snippets = item['snippets']  # 提取 snippets 字段
                snippets_list.append(snippets)  # 将 snippets 添加到列表中
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return snippets_list

# 读取各个文件并保存结果
zero2one_list = read_snippets('../../features/zero2one.jsonl')
one2zero_list = read_snippets('../../features/one2zero.jsonl')

# 打印结果，或者进一步处理
print(one2zero_list[11])
print("____________________")
print(random.choice(zero2one_list))

# # from ptm_attacker import insert_code
#
# code = """
# int process(int data) {
#   int result = data + 1;
#   if (result > 10) {
#     printf("Result is greater than 10.");
#   } else {
#     printf("Result is less than or equal to 10.");
#   }
#   while (result > 10) {
#     return result;
#   }
#
#   return result;
# }
# """
#
# test_code = "ptrdiff_t src_stride;\nint block_h;\nint w;\nptrdiff_t buf_stride;\nconst uint8_t *src;\nint block_w;\nint src_y;\nint h;\nint src_x;\nuint8_t *buf;\nemulated_edge_mc(buf, src, buf_stride, src_stride, block_w, block_h,\\n                     src_x, src_y, w, h, vfixtbl_sse, &ff_emu_edge_vvar_sse,\\n                     hfixtbl_mmxext, &ff_emu_edge_hvar_mmxext);"
# test_code = test_code.replace('\\n', '\n')
#
# code_list = test_code.split("\n")
# for code in code_list:
#     print(code)