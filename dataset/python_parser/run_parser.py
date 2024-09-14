import argparse
from os import replace
import sys
import ast
import os
# from pycparser import c_parser
sys.path.append('..')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

sys.path.append('.')
sys.path.append('../python_parser/')
sys.path.append('../../python_parser/')

from parser_folder.DFG_c import DFG_c
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
import os
import re
import random
import tempfile
import subprocess
from srcml import for_while, while_for, switch_if

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]
c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
c_special_ids = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']

from keyword import iskeyword
def is_valid_identifier(name: str) -> bool:
    name = name.strip()
    if name == '':
        return False
    if "_" in name:
        return False
    if name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    elif name[0].lower() in "abcdefghijklmnopqrstuvwxyz_$":
        for _c in name[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_$":
                return False
    else:
        return False
    return True

def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)

def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True

def is_valid_variable_c(name: str) -> bool:

    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True

def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False

# path = '../../../python_parser/parser_folder/my-languages.so'
# path = '../../../python_parser/parser_folder/my-languages.so'
path = '/data/dxh/Adv-attack/dataset/python_parser/parser_folder/my-languages.so'
c_code = """
int hcf(int n1, int n2)\n{\n    if (n2 != 0)\n          return hcf(n2, n1%n2);\n}
"""
python_code = """
def solve(a,b):\n      h, w, m = map(a, b)\n      if h == 1:\n          cc=a+b\n      elif w == 1:\n          cc = a-b\n      return cc\n
"""
java_code = """
public static < K , V > CacheListenerBridge < K , V > forAfterDelete ( Consumer < EntryEvent < K , V > > consumer ) { return new CacheListenerBridge < K , V > ( null , consumer ) ; }
"""

dfg_function = {
    'c': DFG_c,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

codes = {}
codes = {
    'python': python_code,
    'java': java_code,
    'c': c_code,
}

def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x ]
    return code_tokens
def get_all_snippet(code, lang):
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))

    variable_names, code_tokens = get_identifiers_list(code, "c")
    dec_list = []
    param_list = []
    try:
        root_node = tree.root_node.children[0].children[-1]
    except:
        return [], [], [], [], []
    try:
        param = tree.root_node.children[0].children[-2].children[-1]
        for child in param.children:
            if child.type == 'parameter_declaration':
                param_list.append(child.text.decode('utf-8') + ";")
    except:
        pass
    block_pos, block_snp, expr_snp, dec_list = get_snp_tree(root_node, dec_list, variable_names)

    return block_pos, block_snp, expr_snp, dec_list, param_list

def get_identifiers_list(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    return ret, code_tokens
def get_snippet_token(code, lang):
    code = code.replace("\\n", "\n")
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    return code_tokens

def add_dec(dec_list, param_list, snippet, code):
    variable_names, code_tokens = get_identifiers_list(code, "c")
    var_list = get_var(snippet, variable_names)
    new_code = ""
    param_line = []
    for var in var_list:
        for dec in dec_list:
            dec_token = get_snippet_token(dec, 'c')
            if var in dec_token:
                new_code += dec + "\n"
                break
    new_code += snippet
    var_list_new = get_var(new_code, variable_names)
    for var1 in var_list_new:
        for param in param_list:
            param_token = get_snippet_token(param, 'c')
            if var1 in param_token:
                param_line.append(param)
                break

    return new_code, var_list, param_line


def get_var(code, vars):
    # get all var form code snippet
    var_list = []
    tokens = get_snippet_token(code, 'c')

    for var in vars:
        if var in tokens:
            var_list.append(var)
    var_list = set(var_list)
    var_list = list(var_list)
    return var_list


#

def get_snp_tree(node, dec_list, vars):
    expr_lines = []
    block_pos = []
    block_snp = []
    expr_snp = []

    for index1, child in enumerate(node.children):

        if child.type == 'if_statement' or child.type == 'while_statement' or child.type == 'for_statement':
            if child.end_point[0] - child.start_point[0] <= 7:
                block_pos.append((child.start_point[0], child.end_point[0]))
                text = child.text.decode('utf-8')
                text_list = text.split("\n")
                if text_list[-1].lstrip().startswith("}"):
                    text_list.pop()
                    text_list.append("}")
                text = "\n".join(text_list)
                block_snp.append(text)
                continue
        elif child.type == 'declaration':
            dec_list.append(child.text.decode('utf-8'))
        elif child.type == 'expression_statement':
            expr_lines.append(child.text.decode('utf-8').replace("\n", "\\n"))

    for var in vars:
        if len(var) < 2:
            continue
        tempcode1 = []
        tempcode2 = []
        for index2, line in enumerate(expr_lines):
            if line.find(var) >= 0:
                tempcode1.append(line)
        if len(tempcode1) == 0:
            continue
        while len(tempcode1) > 5:
            tempcode2 = tempcode1[:5]
            code_tmp = "\n".join(tempcode2)
            expr_snp.append(code_tmp)
            tempcode1 = tempcode1[5:]
        code_tmp = "\n".join(tempcode1)
        expr_snp.append(code_tmp)

    return block_pos, block_snp, expr_snp, dec_list

def extract_dataflow(code, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def get_example(code, tgt_word, substitute):
    lang = "c"
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
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


def get_example_sub(code, tgt_word, substitute, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
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
def get_example_pos(code, tgt_word, substitute, lang, pos):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word and index == pos:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[0] + index * diff] + substitute + code[line][pos[1] + index * diff:]

    return "\n".join(code)

def get_example_batch(code, chromesome, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
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

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers_with_tokens(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    # ret = [ [i] for i in ret]
    return ret, code_tokens
    
def get_identifiers(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [[i] for i in ret]
    return ret, code_tokens

def get_space(line):
    count1 = 0
    count2 = 0
    for char in line:
        if char == ' ':
            count1 += 1
        else:
            break
    for char in line:
        if char == '\t':
            count2 += 1
    return count1, count2

# mutations begin here
# a<<b -> a*(2**b)  or a>>b -> a/(2**b)
def replace_bitwise_shift_line(code, line_nums):
    pattern = r'(\w+)\s*(<<|>>)\s*(\w+)'

    def replace_shift(match):
        left_operand = match.group(1)
        operator = match.group(2)
        right_operand = match.group(3)

        if operator == '<<':
            return f"{left_operand} * (2**{right_operand})"
        else:  # operator == '>>'
            return f"{left_operand} / (2**{right_operand})"

    lines = code.split('\n')
    modified_lines = []
    for index, line in enumerate(lines, start=1):
        if index-1 in line_nums:
            modified_line = re.sub(pattern, replace_shift, line)
        else:
            modified_line = line
        modified_lines.append(modified_line)
    modified_code = '\n'.join(modified_lines)
    return modified_code


def replace_condition_with_bool(code, line_num):
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    code_list = code.split('\n')


    def find_if_statements(root_node):
        if_statements = []

        def _find_if_statements(n):
            if n.type == 'if_statement':
                if_statements.append(n)
            for child in n.children:
                _find_if_statements(child)

        _find_if_statements(root_node)
        return if_statements


    root_node = tree.root_node
    if_statements = find_if_statements(root_node)


    if_lines = [statement.start_point[0] for statement in if_statements]

    modified_code_list = []


    try:
        target_if_statement = if_statements[if_lines.index(line_num)]
    except:

        return code


    condition = target_if_statement.children[1].text.decode('utf-8')
    condition2 = condition.replace('(', '').replace(')', '')
    for index, line in enumerate(code_list):
        if index == line_num:
            s1, s2 = get_space(line)
            new_line = s1 * " " + s2 * "\t" + "bool bool_var = " + condition+";"
            modified_code_list.append(new_line)
            line = line.replace(condition2, 'bool_var')
            modified_code_list.append(line)
        else:
            modified_code_list.append(line)

    modified_code = '\n'.join(modified_code_list)

    return modified_code


def add_if_line(code, line_num, idx):
    modified_code_list = []
    code_list = code.split('\n')
    tree = parser[0].parse(bytes(code, 'utf8'))

    # get the line number of for_statement or while_statement or if_statement
    def find_expr_dec(n):
        expr_dec = []
        if n.type == 'for_statement' or n.type == 'while_statement' or n.type == 'if_statement' or n.type == 'switch_statement':
            expr_dec.append(n)
        for child in n.children:
            expr_dec.extend(find_expr_dec(child))
        return expr_dec

    flag = 0
    code_vars=get_identifiers_list(code, 'c')
    if 'a' in code_vars[0] or 'b' in code_vars[0] or "str" in code_vars[0]:
        flag = 1
    root_node = tree.root_node
    expr_dec = find_expr_dec(root_node)
    expr_dec_lines = [statement.start_point[0] for statement in expr_dec]
    if flag==0:
        for index, line in enumerate(code_list):
            if index == line_num and line_num not in expr_dec_lines:
                if idx == 1:
                    s1, s2 = get_space(line)
                    a = random.randint(1, 100)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = " + str(a) + ";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = a;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if(a*b>0){")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 2:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (strlen(str) > 0) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 3:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (a == b) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 4:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (a != b) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 5:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str1[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str2[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (strcmp(str1, str2) == 0) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
            else:
                modified_code_list.append(line)
    else:
        for index, line in enumerate(code_list):
            if index == line_num and line_num not in expr_dec_lines:
                if idx == 1:
                    s1, s2 = get_space(line)
                    a = random.randint(1, 100)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = " + str(a) + ";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = a_1;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if(a_1*b_1>0){")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 2:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str_1[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (strlen(str_1) > 0) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 3:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (a_1 == b_1) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 4:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (a_1 != b_1) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 5:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str1[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str2[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "if (strcmp(str1, str2) == 0) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
            else:
                modified_code_list.append(line)
    modified_code = '\n'.join(modified_code_list)
    return modified_code


def add_while_line(code, line_num, idx):
    modified_code_list = []
    code_list = code.split('\n')
    tree = parser[0].parse(bytes(code, 'utf8'))

    # get the line number of the for_statement or while_statement or if_statement

    def find_expr_dec(n):
        expr_dec = []
        if n.type == 'for_statement' or n.type == 'while_statement' or n.type == 'if_statement' or n.type == 'switch_statement':
            expr_dec.append(n)
        for child in n.children:
            expr_dec.extend(find_expr_dec(child))
        return expr_dec

    root_node = tree.root_node
    expr_dec = find_expr_dec(root_node)
    expr_dec_lines = [statement.start_point[0] for statement in expr_dec]
    flag = 0
    code_vars = get_identifiers_list(code, 'c')

    if 'a' in code_vars[0] or 'b' in code_vars[0] or "str" in code_vars[0]:
        flag = 1

    if flag == 0:
        for index, line in enumerate(code_list):
            if index == line_num and line_num not in expr_dec_lines:
                if idx == 1:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a == b) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b += 1;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 2:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a != b) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b = a;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 3:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a < b) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b = a;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 4:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (strlen(str) > 5) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "strcpy(str, \"a\");")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
            else:
                modified_code_list.append(line)
    else:
        for index, line in enumerate(code_list):
            if index == line_num and line_num not in expr_dec_lines:
                if idx == 1:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a_1 == b_1) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b_1 += 1;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 2:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a_1 != b_1) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b_1 = a_1;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 3:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int a_1 = 2;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "int b_1 = 3;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (a_1 < b_1) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append((s1 + 4) * " " + s2 * "\t" + "b_1 = a_1;")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
                elif idx == 4:
                    s1, s2 = get_space(line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "char str_1[] = \"Constant\";")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "while (strlen(str_1) > 5) {")
                    modified_code_list.append(4 * " " + line)
                    modified_code_list.append(s1 * " " + s2 * "\t" + "strcpy(str_1, \"a\");")
                    modified_code_list.append(s1 * " " + s2 * "\t" + "}")
            else:
                modified_code_list.append(line)
    modified_code = '\n'.join(modified_code_list)
    return modified_code


# ï¿½ï¿½ï¿½ï¿½Ò»ï¿½ï¿½intï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ä£¬ï¿½ï¿½intï¿½ï¿½Îªï¿½ï¿½ï¿½Ë£ï¿½ï¿½ÉºÍ²ï¿½ï¿½ï¿½ï¿½Ãµï¿½Ô­Ê¼intÖµ
def change_int(code, line_num):
    modified_code_list = []
    code_list = code.split('\n')
    tree = parser[0].parse(bytes(code, 'utf8'))

    # ï¿½Ãµï¿½codeï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ú¡ï¿½int a = 10;ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
    def find_int_num(n):
        int_dec = []
        try:
            if n.type == 'declaration' and n.children[0].text.decode('utf-8') == 'int' and n.children[1].children[
                2].type == 'number_literal' and n.children[2].type == ';' and (n.parent.type != 'for_statement'):
                int_dec.append(n)
        except:
            pass
        for child in n.children:
            int_dec.extend(find_int_num(child))
        return int_dec

    root_node = tree.root_node
    int_statements = find_int_num(root_node)
    # ï¿½ï¿½È¡ 'int' ï¿½ï¿½ï¿½ï¿½ï¿½Ðºï¿½
    int_lines = [statement.start_point[0] for statement in int_statements]
    try:
        target_int_statement = int_statements[int_lines.index(line_num)]
    except:
        return code
    part1 = 0
    try:
        part1 = int(target_int_statement.children[1].children[2].text.decode('utf-8')) - 6
    except:
        return code
    part2 = 6
    int_identifier =""
    try:
        int_identifier = target_int_statement.children[1].children[0].text.decode('utf-8')
    except:
        return code
    for index, line in enumerate(code_list):
        if index == line_num:
            s1, s2 = get_space(line)
            modified_code_list.append(s1 * " " + s2 * "\t" + "int " + int_identifier + "_part1 = " + str(part1) + ";")
            modified_code_list.append(s1 * " " + s2 * "\t" + "int " + int_identifier + "_part2 = " + str(part2) + ";")
            modified_code_list.append(
                s1 * " " + s2 * "\t" + "int " + int_identifier + " = " + int_identifier + "_part1 + " + int_identifier + "_part2;")
        else:
            modified_code_list.append(line)
    modified_code = '\n'.join(modified_code_list)
    return modified_code


def cmd(command):
    global flag
    flag = True
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                            errors="ignore")
    subp.wait(10)
    if subp.poll() == 0:
        flag = True
    else:
        err = subp.stderr.read()
        print("Error:", err)
        flag = False


def c_to_xml(code, tmp_dir):
    # create a c file
    c_file = os.path.join(tmp_dir, "tmp.c")
    with open(c_file, "w") as f:
        f.write(code)
    # convert the c file to a xml file
    xml_file = os.path.join(tmp_dir, "tmp.xml")
    str = 'srcml \"' + c_file + '\" -o \"' + xml_file + '\" --position --src-encoding UTF-8'
    cmd(str)
    # delete the tmp c file
    os.remove(c_file)
    return xml_file


# Given an xml path, convert the xml to a c file through srcml, then read the c file, then delete the xml file and c file , and return a piece of c code text
def xml_to_c(xml_file):
    # convert xml to c
    c_file = os.path.join(os.path.dirname(xml_file), "tmp.c")
    str = "srcml \"" + xml_file + '\" -o \"' + c_file + "\" --src-encoding UTF-8"
    cmd(str)
    # read the c file
    with open(c_file, "r") as f:
        code = f.read()
    # delete the xml file and c file
    os.remove(xml_file)
    os.remove(c_file)

    return code


# trans funcs include while_for,for_while,switch_if
def srcml_trans(code, trans, line_num):
    tmp_dir = tempfile.mkdtemp()
    xml_file = c_to_xml(code, tmp_dir)
    new_xml = trans.program_transform(xml_file, line_num)
    new_code = xml_to_c(new_xml)
    os.rmdir(tmp_dir)
    return new_code

def is_valid_expression_for_if(code, line_number):
    
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    
    code_lines = code.splitlines()

    
    if not code_lines[line_number].strip().endswith(';'):
        return False
    if code_lines[line_number].strip().endswith('}'):
        return False
    target_line = code_lines[line_number].strip()
    
    def find_deepest_node(node, target_line_content, current_depth=0, deepest_node=None, max_depth=0):
        # Extract the string content of the node from the source code
        node_start_byte = node.start_byte
        node_end_byte = node.end_byte
        node_content = code[node_start_byte:node_end_byte]

        # Check if the node contains the target line content
        if target_line_content in node_content:
            # Update the deepest node if this node is deeper
            if current_depth > max_depth:
                deepest_node = node
                max_depth = current_depth

            # Recursively check child nodes
            for child in node.children:
                deepest_node, max_depth = find_deepest_node(
                    child, target_line_content, current_depth + 1, deepest_node, max_depth
                )

        return deepest_node, max_depth

    # Call the recursive function on the root node
    deepest_node, _ = find_deepest_node(root_node, target_line)
    if deepest_node:
        
        if deepest_node.start_point[0] == deepest_node.end_point[0]:
            if line_number >=1:
                pre_target_line = code_lines[line_number-1].strip()
                if (pre_target_line.strip().startswith('if') or pre_target_line.strip().startswith('for')or pre_target_line.strip().startswith('while')or pre_target_line.strip().startswith('else') or pre_target_line.strip().startswith('else if')) and not pre_target_line.strip().endswith('{'):
                    return False
                return True
    return False



def mutation(code, index, line_num):
    new_code = ""
    if not is_valid_expression_for_if(code, line_num):
        return new_code
    if index == 0:
        new_code = replace_bitwise_shift_line(code, [line_num])
    elif index == 1:
        new_code = replace_condition_with_bool(code, line_num)
    elif index == 2:
        random_number = random.randint(1, 5)
        new_code = add_if_line(code, line_num, random_number)
    elif index == 3:
        random_number = random.randint(1, 4)
        new_code = add_while_line(code, line_num, random_number)
    elif index == 4:
        new_code = change_int(code, line_num)
    elif index == 5:
        new_code = srcml_trans(code, while_for, line_num)
    elif index == 6:
        new_code = srcml_trans(code, for_while, line_num)
    elif index == 7:
        new_code = srcml_trans(code, switch_if, line_num)
    return new_code

def add_dead_code(code, line_num):
    code_list = code.split('\n')
    modified_code_list = []
    #Ëæ»úÑ¡ÔñÒ»ÐÐ´úÂë£¬¹¹Ôìprintf("´úÂëÓï¾ä")
    selected_code = random.choice(code_list)
    for index, line in enumerate(code_list):
        if index == line_num:
            s1, s2 = get_space(line)
            modified_code_list.append(line)
            modified_code_list.append(s1 * " " + s2 * "\t" + "printf(\"" + selected_code + "\");")

        else:
            modified_code_list.append(line)
    modified_code = '\n'.join(modified_code_list)
    return modified_code

def mutation_tra(code, index, line_num):
    new_code = ""
    if index == 0:
        new_code = srcml_trans(code, while_for, line_num)
    elif index == 1:
        new_code = srcml_trans(code, for_while, line_num)
    elif index == 2:
        new_code = srcml_trans(code, switch_if, line_num)
    elif index == 3:
        new_code = add_dead_code(code, line_num)
    return new_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="java", type=str,
                        help="language.")
    args = parser.parse_args()
    code = codes[args.lang]
    print(code)
    data, _ = get_identifiers_with_tokens(code, args.lang)
    print(data)
    code_ = get_example(java_code, "inChannel", "dwad", "java")
    code_ = get_example_batch(java_code, {"inChannel":"dwad", "outChannel":"geg"}, "java")
    print(code_)


if __name__ == '__main__':
    main()

