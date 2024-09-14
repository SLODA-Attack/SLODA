from tqdm import tqdm
import csv
import os
import sys
sys.path.append('../dataset')

from data_augmentation import get_example_batch, get_identifiers,get_example
import torch
import javalang
import random
from torch.utils.data.dataset import Dataset
import re
import numpy as np
from tree_sitter import Language, Parser
torch.set_printoptions(threshold=np.inf)

path = 'tree_sitter my-languages.so'
c_path = 'tree-sitter-c folder'
cpp_path = 'tree-sitter-cpp folder'
# 构建和加载C/C++语法
# Language.build_library(
#     path,
#     [c_path, cpp_path]
# )

C_LANGUAGE = Language(path, 'c')
CPP_LANGUAGE = Language(path, 'cpp')

parser = Parser()
parser.set_language(C_LANGUAGE)

class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

class EPVDDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].path_source)


class GraphCodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((512 + 128,
                              512 + 128), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].label))

PROMPT_DICT = {
#     "prompt_input": (
#         "User: " + """
# Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n
# ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:
#     """+"\n\nAssistant:"
#     ),
#     "prompt_input": (
#         "Human: " + """
# Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n
# ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:
#     """+"\nAssistant:"
#     ),
#     "prompt_input": (
# "<|start_header_id|>user<|end_header_id|>\n\n" + """
# Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n
# ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:
#     """+"<|eot_id|>"
#     ),
    "prompt_input": ("""
Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n
### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:
    """
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def get_ann(code):
    ann = {
        "instruction": "Detect whether the following code contains vulnerabilities.",
        "input": code,
    }
    return ann

def insert_at_line(source_code, line_number, insert_code):
    lines = source_code.split('\n')
    insert_code = insert_code.replace('\\n', '\n')
    identifiers = get_identifiers(source_code)
    identifiers_insert = get_identifiers(insert_code)
    code = bytes(source_code, 'utf8')
    tree = parser.parse(code)
    cursor = tree.walk()
    intersection_list = list(set(identifiers).intersection(set(identifiers_insert)))
    #print(intersection_list)
    if len(intersection_list)!=0:
        new_list = [f"{item}_{random.randint(1, 10)}" for item in intersection_list]
        #print(new_list)
        for index,var in enumerate(new_list):
            insert_code=get_example(insert_code,intersection_list[index],var)
    
    #print(insert_code)
    insert_point = None
    current_line = 1  # 行号从1开始计数

    def find_insert_point(node):
        nonlocal insert_point, current_line
        if node.start_point[0] + 1 == line_number:
            if node.type in ['compound_statement']:
                insert_point = node.start_byte + 1
            else:
                insert_point = node.end_byte
        for child in node.children:
            find_insert_point(child)

    find_insert_point(tree.root_node)

    if insert_point is not None and code[insert_point - 1:insert_point] not in {b'|', b',', b')'}:
        line_start = code.rfind(b'\n', 0, insert_point) + 1
        indent = code[line_start:insert_point].split(b'\n')[-1]
        indent = indent[:len(indent) - len(indent.lstrip())]  # 保留原始缩进

        # 检查插入点是否位于'{'后面
        if code[insert_point - 1:insert_point] in {b'{', b':'}:
            # 为新行添加额外的缩进
            # indent += b'  '
            line = lines[line_number]
            if line.strip() == '':  # 如果行是空的
                line = lines[line_number + 1]  # 移至下一行
            # 找到非空行，提取缩进
            indent_length = len(line) - len(line.lstrip())
            indent = line[:indent_length].encode()

        before = code[:insert_point]
        after = code[insert_point:]
        res = before
        for line in insert_code.split("\n"):
            res += bytes("\n" + indent.decode() + line, 'utf8')
        res += after
        return res.decode('utf8')
    else:
        return source_code


def insert_dead_code(code, id2token, code_lines, n_candi):
    adv_codes = []
    identifiers = get_identifiers(code)
    lines = code.split('\n')
    valid_positions = []

    for i, line in enumerate(lines[:-1]):  # 避免考虑最后一行，因为没有后续行可以参考
        next_line = lines[i + 1]
        # 检查当前行是否是有效的插入点
        if is_complete_statement(line) and is_valid_insertion_line(line) and is_valid_insertion_line(next_line):
            # 特别处理在 '{' 或 '}' 后的插入点
            if '{' in line or '}' in line:
                # 使用下一行的缩进级别
                match = re.match(r'(\s*)', next_line)
                indent = match.group(1) if match else ''
                valid_positions.append((i + 1, indent))
            else:
                match = re.match(r'(\s*)', line)
                indent = match.group(1) if match else ''
                valid_positions.append((i + 1, indent))

    if not valid_positions:
        return []  # 如果没有合适的位置，不修改代码

    total = 0
    times = 0
    while total < n_candi:
        new_lines = lines.copy()
        times += 1
        # 随机选择一个合适的位置
        var = random.sample(id2token, 1)[0]
        line = random.sample(code_lines, 1)[0]
        insert_position, indent = random.choice(valid_positions)
        dead_code = f'{indent}char {var}_2[] = {line};'

        new_lines.insert(insert_position, dead_code)
        if "\n".join(new_lines) not in adv_codes:
            adv_codes.append("\n".join(new_lines))
            total += 1
        if total >= n_candi or times >= 10:
            return adv_codes

def is_complete_statement(line):
    # 确保这是一条完整的语句（以分号、大括号结束）
    return re.search(r'[;\{\}]$', line.strip())

def is_valid_insertion_line(line):
    # 检查这一行是否适合插入代码
    # 避免在大括号或只有空白的行插入代码
    return not re.match(r'^\s*[\{\}]?\s*$', line)


def get_llm_result(code, model, tokenizer, label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = get_ann(code)
    
    max_len=1000
    
    if ann.get("input", "") == "":
        prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
    #model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    # print("prompt",prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:,:max_len]

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    
    model_input = {
        'input_ids': input_ids.to("cuda"),
        'attention_mask':attention_mask
    }
    # from infer import ask_
    # print("START"+"="*20)
    # ask_(prompt, model, tokenizer)
    # print("FINISH"+"="*20)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=model_input['input_ids'],
            attention_mask=model_input['attention_mask'],
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            temperature=None,  # 显式重置 temperature
            top_p=None,  # 显式重置 top_p
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
        flag = 0
        s = generation_output.sequences[0]
        generated_response = tokenizer.decode(s)
        response = generated_response.split("### Response:")[-1].strip()
        # filling = tokenizer.batch_decode(generation_output[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
        # print("s", s)
        # print("generated_response", generated_response)
        # print("response", response)
        # filling = tokenizer.decode(generation_output2[0], skip_special_tokens=True)
        # print("filling", filling)
        logits = generation_output.scores
        probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
        if len(probabilities) >= 2:
            prob_dist = probabilities[-2]
            # 获取最高概率的标记及其置信分数
            confidence_score, predicted_token_id = torch.max(prob_dist, dim=-1)
            score = confidence_score.item()
        else:
            score = 1
            flag = 1
        # print("response", response)
        # if response:
        #     result = response[10]
        if response:
            result = response[0]
        else:
            result = label
            flag = 1

    return score, result, flag

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # 并非直接tokenize这句话，而是tokenize了每个splited words.
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        # 将subwords对齐
        index += len(sub)

    return words, sub_words, keys

def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    给定一段文本，以及需要被mask的位置,返回一组masked后的text
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<mask>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)

    return masked_token_list, replace_token_positions

def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions

def map_chromesome(chromesome: dict, code: str, lang: str) -> str:
    temp_replace = get_example_batch(code, chromesome)

    return temp_replace


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp=[]
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip()!="":
            temp.append(x)
    return '\n'.join(temp)

def get_code_tokens(code):
    code = remove_comments_and_docstrings(code)
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    return code_tokens


def build_vocab(codes, limit=5000):
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm(codes):
        for t in c:
            if len(t) > 0:
                if t[0] == '"' and t[-1] == '"':
                    vocab_cnt["<str>"] += 1
                elif t[0] == "'" and t[-1] == "'":
                    vocab_cnt["<char>"] += 1
                elif t[0] in "0123456789.":
                    if 'e' in t.lower():
                        vocab_cnt["<fp>"] += 1
                    elif '.' in t:
                        if t == '.':
                            if t not in vocab_cnt.keys():
                                vocab_cnt[t] = 0
                            vocab_cnt[t] += 1
                        else:
                            vocab_cnt["<fp>"] += 1
                    else:
                        vocab_cnt["<int>"] += 1
                elif t in vocab_cnt.keys():
                    vocab_cnt[t] += 1
                else:
                    vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x: x[1], reverse=True)

    idx2txt = ["<unk>"] + ["<pad>"] + [it[0] for index, it in enumerate(vocab_cnt) if index < limit - 1]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx

class Recorder():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                              "Original Code",
                              "Adversarial Code",
                              "Program Length",
                              "Identifier Num",
                              "Replaced Identifiers",
                              "Query Times",
                              "Time Cost",
                              "Type"])
        self.f.flush()

    def write(self, index, code, adv_code, prog_length, nb_var, replace_info, query_times, time_cost, attack_type):
        self.writer.writerow([index,
                              code,
                              adv_code,
                              prog_length,
                              nb_var,
                              replace_info,
                              query_times,
                              time_cost,
                              attack_type])
        self.f.flush()
class Recorder_style():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                              "Original Code",
                              "Adversarial Code",
                              "True Label",
                              "Adv Label",
                              "Query Times",
                              "Time Cost",
                              "Type"])
        self.f.flush()

    def write(self, index, code, adv_code, true_label, adv_label, query_times, time_cost,attack_type):
        self.writer.writerow([index,
                              code,
                              adv_code,
                              true_label,
                              adv_label,
                              query_times,
                              time_cost,
                              attack_type])
        self.f.flush()

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    code_insert = """
    unsigned int flags = 0;
    flags<<4;
    const int stream_index = pkt->stream_index;
    int size               = pkt->size;
    """
    code ="""
    if (flags == 0):
        return true;
    """
    
    code_new=insert_at_line(code,2,code_insert)
    