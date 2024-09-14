from tree_sitter import Language, Parser
import json
import tqdm
import random
from data_utils import is_valid_identifier
from data_utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

path = './python_parser/parser_folder/my-languages.so'
c_path = './python_parser/parser_folder/tree-sitter-c'
cpp_path = './python_parser/parser_folder/tree-sitter-cpp'
# 构建和加载C/C++语法
Language.build_library(
  path,
  [c_path, cpp_path]
)

C_LANGUAGE = Language(path, 'c')
CPP_LANGUAGE = Language(path, 'cpp')

def get_identifiers(code):
    def parse_code(language):
        parser = Parser()
        parser.set_language(language)
        return parser.parse(bytes(code, "utf8"))

    def extract_identifiers(node):
        identifiers = []
        if node.type == 'identifier':
            identifiers.append(node.text.decode('utf8'))
        for child in node.children:
            identifiers.extend(extract_identifiers(child))
        return identifiers

    # 尝试使用C++解析器
    tree = parse_code(CPP_LANGUAGE)
    if tree.root_node.has_error:
        # 如果C++解析失败，尝试使用C解析器
        tree = parse_code(C_LANGUAGE)

    return extract_identifiers(tree.root_node)

def get_code_tokens(code):
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    return code_tokens

def get_example(code, tgt_word, substitute):
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

    return "\n".join(code)

def get_example_batch(code, chromesome):
    parser = Parser()
    code = code.replace("\\n", "\n")
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    if tree.root_node.has_error:
        # 如果C++解析失败，尝试使用C解析器
        parser.set_language(C_LANGUAGE)
        tree = parser.parse(bytes(code, 'utf8'))

    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
            diff += pos[2]

    return "\n".join(code)


def change_code(code, identifiers, code_vocab):
    number_of_elements = len(identifiers) // 5 if len(identifiers) >= 5 else 1
    selected_elements = random.sample(identifiers, number_of_elements)
    result_dict = {key: random.choice(code_vocab) for key in selected_elements}
    new_code = get_example_batch(code, result_dict)
    return new_code, result_dict
