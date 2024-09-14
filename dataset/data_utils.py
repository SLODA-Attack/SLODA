import tqdm

c_keywords = [
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
    "_Generic", "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
    "__func__"
]

cpp_keywords = [
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case",
    "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl",
    "concept", "const", "consteval", "constexpr", "constinit", "const_cast",
    "continue", "co_await", "co_return", "co_yield", "decltype", "default",
    "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit",
    "export", "extern", "false", "float", "for", "friend", "goto", "if",
    "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "reflexpr", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch", "synchronized", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
    "while", "xor", "xor_eq"
]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
c_special_ids = [
    "main",  # main function
    "stdio", "cstdio", "stdio.h",                                 # <stdio.h> & <cstdio>
    "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",      # <stdio.h> types & streams
    # <stdio.h> functions
    "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", "fopen", "freopen",
    "setbuf", "setvbuf", "fprintf", "fscanf", "printf", "scanf", "snprintf", "sprintf",
    "sscanf", "vprintf", "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
    "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc", "fread", "fwrite",
    "fgetpos", "fseek", "fsetpos", "ftell", "rewind", "clearerr", "feof", "ferror",
    "perror", "getline",
    "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
    "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
    # <stdlib.h> functions
    "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold", "strtol", "strtoll",
    "strtoul", "strtoull", "rand", "srand", "aligned_alloc", "calloc", "malloc", "realloc",
    "free", "abort", "atexit", "exit", "at_quick_exit", "_Exit", "getenv", "quick_exit",
    "system", "bsearch", "qsort", "abs", "labs", "llabs", "div", "ldiv", "lldiv", "mblen",
    "mbtowc", "wctomb", "mbstowcs", "wcstombs",
    "string", "cstring", "string.h",                             # <string.h> & <cstring>
    # <string.h> functions
    "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat", "strncat", "strchr",
    "strrchr", "strcmp", "strncmp", "strcoll", "strcpy", "strncpy", "strerror", "strlen",
    "strspn", "strcspn", "strpbrk", "strstr", "strtok", "strxfrm", "memccpy", "mempcpy",
    "strcat_s", "strcpy_s", "strdup", "strerror_r", "strlcat", "strlcpy", "strsignal",
    "strtok_r",
    # <iostream> family
    "iostream", "istream", "ostream", "fstream", "sstream", "iomanip", "iosfwd",
    # <iostream> types
    "ios", "wios", "streamoff", "streampos", "wstreampos", "streamsize", "cout", "cerr",
    "clog", "cin",
    # <iostream> manipulators
    "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase", "noshowbase", "showpoint",
    "noshowpoint", "showpos", "noshowpos", "unitbuf", "nounitbuf", "uppercase",
    "nouppercase", "left", "right", "internal", "dec", "oct", "hex", "fixed", "scientific",
    "hexfloat", "defaultfloat", "width", "fill", "precision", "endl", "ends", "flush", "ws",
    # <math.h> functions
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp",
    "sqrt", "log", "log10", "pow", "powf", "ceil", "floor", "abs", "fabs", "cabs", "frexp",
    "ldexp", "modf", "fmod", "hypot", "ldexp", "poly", "matherr"
]

cpp_std_ids = [
    # 容器
    "vector", "list", "map", "set", "unordered_map", "unordered_set", "queue", "stack", "array", "deque",
    "forward_list",

    # 输入/输出
    "iostream", "fstream", "stringstream", "cin", "cout", "cerr", "clog", "istream", "ostream", "ifstream", "ofstream",
    "stringstream",

    # 字符串处理
    "string", "wstring", "u16string", "u32string",

    # 算法
    "sort", "max", "min", "find", "find_if", "accumulate", "count", "for_each",

    # 迭代器
    "iterator", "reverse_iterator", "const_iterator", "const_reverse_iterator",

    # 数学和数值
    "complex", "numeric_limits", "valarray", "bitset",

    # 内存管理
    "allocator", "unique_ptr", "shared_ptr", "weak_ptr", "auto_ptr",

    # 线程和同步
    "thread", "mutex", "lock_guard", "unique_lock", "condition_variable", "future", "promise",

    # 其他实用工具
    "tuple", "pair", "optional", "variant", "any", "function",

    # 异常处理
    "exception", "runtime_error", "logic_error", "invalid_argument",

    # C++17以后引入的特性
    "filesystem", "optional", "variant", "any", "string_view",

    # C++标准模板库(STL)算法
    "all_of", "any_of", "none_of", "for_each", "find", "find_if", "find_if_not",
    "find_end", "find_first_of", "adjacent_find", "count", "count_if", "mismatch",
    "equal", "is_permutation", "search", "search_n", "copy", "copy_if", "copy_n",
    "copy_backward", "move", "move_backward", "fill", "fill_n", "transform",
    "generate", "generate_n", "remove", "remove_if", "remove_copy", "remove_copy_if",
    "replace", "replace_if", "replace_copy", "replace_copy_if", "swap", "swap_ranges",
    "iter_swap", "reverse", "reverse_copy", "rotate", "rotate_copy", "random_shuffle",
    "shuffle", "unique", "unique_copy", "is_partitioned", "partition", "partition_copy",
    "stable_partition", "partition_point", "is_sorted", "is_sorted_until", "sort",
    "partial_sort", "partial_sort_copy", "stable_sort", "nth_element", "lower_bound",
    "upper_bound", "binary_search", "equal_range", "merge", "inplace_merge", "includes",
    "set_difference", "set_intersection", "set_symmetric_difference", "set_union",
    "is_heap", "is_heap_until", "make_heap", "push_heap", "pop_heap", "sort_heap",
    "max", "max_element", "min", "min_element", "minmax", "minmax_element", "lexicographical_compare",
    "next_permutation", "prev_permutation"
]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']

def is_valid_identifier(name: str) -> bool:
    name = name.strip()
    if name == '':
        return False

    # 将所有关键字和特殊标识符放入一个集合中
    all_special_ids = set(c_keywords + cpp_keywords + c_special_ids + cpp_std_ids + c_macros + special_char)

    # 检查是否是特殊标识符或关键字
    if name in all_special_ids:
        return False

    # 检查运算符重载
    if name.startswith("operator"):
        return name[len("operator"):].isalpha()

    # 标准标识符检查
    if name[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$":
        return False

    return all(c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$" for c in name[1:])


import collections
import re
from io import StringIO
import tokenize


def isSameTree(root_p, root_q) -> bool:
    if not root_p and not root_q:
        return True
    if not root_p or not root_q:
        return False

    queue_p = collections.deque([root_p])
    queue_q = collections.deque([root_q])

    while queue_p and queue_q:
        node_p = queue_p.popleft()
        node_q = queue_q.popleft()
        if node_p.type != node_q.type:
            return False
        if len(node_p.children) != len(node_q.children):
            return False
        if len(node_p.children) > 0:
            for child_p, child_q in zip(node_p.children, node_q.children):
                if child_p.type == child_q.type:
                    queue_p.append(child_p)
                    queue_p.append(child_q)
                else:
                    return False

    return True


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
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


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if root_node:
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            index = (root_node.start_point, root_node.end_point)
            _, code = index_to_code[index]
            if root_node.type != code:
                return [(root_node.start_point, root_node.end_point)]
            else:
                return []
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += tree_to_variable_index(child, index_to_code)
            return code_tokens
    else:
        return []


def index_to_code_token(index, code):
    # 开始位置
    start_point = index[0]
    end_point = index[1]
    # 如果在同一行
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    # 如果多行
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s
