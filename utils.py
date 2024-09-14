import torch
import torch.nn as nn
import copy
import subprocess
import random
# import nltk
import json
import sys
import re
import pickle
from tqdm import tqdm
import javalang
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import csv

def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def get_replaced_code(code, original, adv):
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    for i, token in enumerate(code_tokens):
        if token == original:
            code_tokens[i] = adv
    return " ".join(code_tokens)


def get_code_tokens(code):
    code = remove_comments_and_docstrings(code)
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    return code_tokens


def split_java_token(cstr, camel_case=True, split_char='_'):
    res_split = []

    if camel_case:
        res_split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', cstr)).split()

    if split_char:
        char_splt = []
        if not camel_case:
            res_split = [cstr]
        for token in res_split:
            char_splt += token.split(split_char)
        res_split = char_splt
    return [token for token in res_split if len(token) > 0]


# From Alert codebases

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while", "_"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double",
                    "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String",
                    "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet",
                    "HashMap"]
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
            "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]  # <stdlib.h> macro
c_special_ids = ["main",  # main function
                 "stdio", "cstdio", "stdio.h",  # <stdio.h> & <cstdio>
                 "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",  # <stdio.h> types & streams
                 "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",  # <stdio.h> functions
                 "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                 "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                 "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                 "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                 "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                 "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                                                                   "stdlib", "cstdlib", "stdlib.h",
                 # <stdlib.h> & <cstdlib>
                 "size_t", "div_t", "ldiv_t", "lldiv_t",  # <stdlib.h> types
                 "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                 "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                 "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                 "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                 "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                 "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                 "mbstowcs", "wcstombs",
                 "string", "cstring", "string.h",  # <string.h> & <cstring>
                 "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",  # <string.h> functions
                 "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                 "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                 "strpbrk", "strstr", "strtok", "strxfrm",
                 "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",  # <string.h> extension functions
                 "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                 "iostream", "istream", "ostream", "fstream", "sstream",  # <iostream> family
                 "iomanip", "iosfwd",
                 "ios", "wios", "streamoff", "streampos", "wstreampos",  # <iostream> types
                 "streamsize", "cout", "cerr", "clog", "cin",
                 "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",  # <iostream> manipulators
                 "noshowbase", "showpoint", "noshowpoint", "showpos",
                 "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                 "left", "right", "internal", "dec", "oct", "hex", "fixed",
                 "scientific", "hexfloat", "defaultfloat", "width", "fill",
                 "precision", "endl", "ends", "flush", "ws", "showpoint",
                 "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",  # <math.h> functions
                 "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                 "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                 "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']


def is_valid_identifier(name: str) -> bool:
    name = name.strip()
    if name == '':
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

from keyword import iskeyword


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


def is_valid_substitue(substitute: str, tgt_word: str, lang: str) -> bool:
    '''
    判断生成的substitues是否valid，如是否满足命名规范
    '''
    is_valid = True

    if not is_valid_variable_name(substitute, lang):
        is_valid = False

    return is_valid
