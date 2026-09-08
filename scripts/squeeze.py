#!/usr/bin/env python3
"""sunge · squeeze —— 孙割的自包含省-token 工具(单文件, 零硬依赖)。

SNS(Shorthand Natural-language Syntax) 压缩规则引擎, 服务于本 skill 的军火库
(保留: 客套词/填充词删除、短词替换、io/条件/列表/json 符号化、中文常用式、收尾清理),
外加自研的代码骨架化(AST skeletonize, Python 用标准库 ast, 其余语言通用声明提取)。
规则改写整理自 MIT 协议的 tokensqueeze 项目。

用法(命令行):
    python scripts/squeeze.py "请写一个函数返回数组之和"        # 压缩文本
    python scripts/squeeze.py --explain "Could you please ..."  # 压缩 + 逐规则解释
    python scripts/squeeze.py --count "some text"               # 数 token
    python scripts/squeeze.py --skeleton app.py                 # 代码骨架化(读结构, 不读肉体)
    python scripts/squeeze.py --file notes.md                   # 从文件压缩
    echo "pipe me" | python scripts/squeeze.py                  # 从 stdin 读

Python API:
    compress(text) -> str
    explain(text) -> dict
    count_tokens(text, encoding="cl100k_base") -> int
    skeletonize(code, lang=None) -> str
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable


# ============================================================================
# 一、SNS 规则引擎 —— 有顺序的 regex 改写。顺序即语义: 先归一波通用式,
#     后做字面/填充清理。替换保持对 LLM 无歧义(fn: 表示"写函数", -> 表示"产出/返回")。
# ============================================================================

@dataclass
class Rule:
    id: str
    category: str
    pattern: str
    replacement: str
    rationale: str = ""
    flags: int = re.IGNORECASE
    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    def compile(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled


@dataclass
class RuleHit:
    rule_id: str
    category: str
    before: str
    after: str


# 规则按应用顺序排列。保留下来的都是通用、可解释、跨场景有用的改写。
RULES: list[Rule] = [
    # --- 1. 函数/程序模板(整段短语 -> 符号) ------------------------------
    Rule("fn.write_function_typed", "function",
         r"\b(?:write|create|build|implement|make|generate|develop|code)\s+(?:a|an|the)?\s*(python|js|javascript|typescript|ts|go|rust|java|c\+\+|cpp|c#|csharp|ruby|php|bash|shell)\s+function\s+(?:that\s+|which\s+|to\s+)?",
         r"fn[\1]:", "压缩 '写一个 python 函数…' 为 'fn[python]:…'"),
    Rule("fn.write_function", "function",
         r"\b(?:write|create|build|implement|make|generate|develop|code)\s+(?:a|an|the)?\s*function\s+(?:that\s+|which\s+|to\s+)?",
         r"fn:", "压缩 '写一个函数…' 为 'fn:…'"),
    Rule("fn.define_function", "function",
         r"\b(define|declare)\s+(?:a|an|the)?\s*function\s+(?:that\s+|which\s+|to\s+)?",
         r"fn: ", "压缩 'define a function that …' 为 'fn: …'"),
    Rule("fn.write_program", "function",
         r"\b(write|create|build)\s+(?:a|an|the)?\s*(python|js|javascript|typescript|go|rust|java|bash|shell)?\s*(program|script|app|application)\s+(?:that\s+|which\s+|to\s+)?",
         r"prog(\2): ", "压缩 '写一个 python 程序…' 为 'prog(python): …'"),
    Rule("fn.write_class", "function",
         r"\b(write|create|implement|define)\s+(?:a|an|the)?\s*(python|js|typescript|java|c\+\+|ruby)?\s*class\s+(?:that\s+|which\s+|to\s+)?",
         r"cls(\2): ", "压缩 '建一个类…' 为 'cls: …'"),
    Rule("fn.write_method", "function",
         r"\b(write|create|implement|add)\s+(?:a|an|the)?\s*method\s+(?:that\s+|which\s+|to\s+)?",
         r"method: ", "压缩 '写一个方法…' 为 'method: …'"),
    Rule("fn.write_endpoint", "function",
         r"\b(write|create|build|implement|expose)\s+(?:a|an|the)?\s*(rest|http|api|graphql)?\s*endpoint\s+(?:that\s+|which\s+|to\s+)?",
         r"endpoint(\2): ", "压缩 '写一个 REST 端点…' 为 'endpoint(rest): …'"),
    Rule("fn.write_test", "function",
         r"\b(write|add|create)\s+(?:a|an|the|some)?\s*(unit|integration|e2e)?\s*tests?\s+(?:that\s+|which\s+|to\s+|for\s+)?",
         r"tests(\2): ", "压缩 '写单元测试…' 为 'tests(unit): …'"),

    # --- 1b. JSON/结构化输出(须在 io.* 之前, 先抓格式动词) ---------------
    Rule("json.return_as_json", "json",
         r"\breturns?\s+(?:the\s+)?(?:result\s+|response\s+|output\s+)?(?:as|in|using)\s+json(?:\s+format)?\b",
         r"->json", "压缩 '以 JSON 返回结果' 为 '->json'"),
    Rule("json.respond_in_json", "json",
         r"\b(respond|reply|answer)\s+(?:in|with|using)\s+json(?:\s+format)?",
         r"->json", "压缩 '用 JSON 回复' 为 '->json'"),
    Rule("json.output_format", "json",
         r"\boutput\s+(?:the\s+)?(?:result\s+)?(?:in|as|using)\s+(json|xml|yaml|csv|markdown|md)\s+format",
         r"->\1", "压缩 '以 <格式> 输出' 为 '-><格式>'"),
    Rule("json.format_as", "json",
         r"\bformat\s+(?:the\s+)?(?:output\s+|response\s+)?as\s+(json|xml|yaml|csv|markdown|md)",
         r"->\1", "压缩 '把输出格式化为 <格式>' 为 '-><格式>'"),
    Rule("json.json_array", "json",
         r"\bjson\s+array\s+of\s+",
         r"json[", "压缩 'json array of' 为 'json['"),
    Rule("json.with_fields", "json",
         r"\bwith\s+(?:the\s+)?fields?\s+",
         r" {", "压缩 'with the fields' 为 ' {'"),
    Rule("json.with_keys", "json",
         r"\bwith\s+(?:the\s+)?keys?\s+",
         r" {", "压缩 'with the keys' 为 ' {'"),

    # --- 2. 数据流 / I/O ---------------------------------------------------
    Rule("io.takes_returns", "io",
         r"\bthat\s+takes\s+(.+?)\s+(?:and|,)\s+returns\s+",
         r"(\1) -> ", "压缩 'that takes X and returns …' 为 '(X) -> …'"),
    Rule("io.takes_only", "io",
         r"\bthat\s+takes\s+(?:in\s+)?(?:a|an|the)?\s*",
         r"(", "压缩 'that takes a …' 开头为 '('"),
    Rule("io.given_input", "io",
         r"\bgiven\s+(?:a|an|the)?\s*(.+?)\s+as\s+input\s*,?\s*",
         r"in=\1; ", "压缩 'given X as input,' 为 'in=X;'"),
    Rule("io.accepts_arg", "io",
         r"\baccepts?\s+(?:a|an|the)?\s*",
         r"in: ", "压缩 'accepts a …' 为 'in: …'"),
    Rule("io.returns_a", "io",
         r"\breturns?\s+(?:a|an|the)?\s*",
         r"-> ", "压缩 'returns a …' 为 '-> …'"),
    Rule("io.outputs_a", "io",
         r"\boutputs?\s+(?:a|an|the)?\s*",
         r"out: ", "压缩 'outputs a …' 为 'out: …'"),
    Rule("io.print_the", "io",
         r"\bprints?\s+(?:out\s+)?(?:the|a|an)?\s*",
         r"print: ", "压缩 'prints out the …' 为 'print: …'"),

    # --- 3. 条件 -----------------------------------------------------------
    Rule("if.if_then", "conditional",
         r"\bif\s+(.+?)\s+then\s+",
         r"if(\1)? ", "压缩 'if X then …' 为 'if(X)? …'"),
    Rule("if.when_x", "conditional",
         r"\bwhen\s+(.+?)\s*,\s*",
         r"on(\1): ", "压缩 'when X,' 为 'on(X):'"),
    Rule("if.in_case", "conditional",
         r"\bin\s+case\s+(?:of|that)?\s*",
         r"if: ", "压缩 'in case of/that' 为 'if:'"),
    Rule("if.otherwise", "conditional",
         r"\botherwise\s*,?\s*",
         r"else: ", "压缩 'otherwise,' 为 'else:'"),
    Rule("if.unless", "conditional",
         r"\bunless\s+",
         r"if !", "压缩 'unless' 为 'if !'"),
    Rule("if.only_if", "conditional",
         r"\bonly\s+if\s+",
         r"iff ", "压缩 'only if' 为 'iff'"),

    # --- 4. 列表操作 -------------------------------------------------------
    Rule("list.iterate_over", "list",
         r"\b(iterate|loop|go)\s+(?:over|through)\s+(?:each|every|all)?\s*",
         r"forEach ", "压缩 'iterate over each …' 为 'forEach …'"),
    Rule("list.for_each", "list",
         r"\bfor\s+each\s+",
         r"forEach ", "压缩 'for each' 为 'forEach'"),
    Rule("list.list_of", "list",
         r"\b(?:a\s+)?list\s+of\s+",
         r"[", "压缩 'a list of' 为 '['"),
    Rule("list.array_of", "list",
         r"\b(?:an\s+)?array\s+of\s+",
         r"[", "压缩 'an array of' 为 '['"),
    Rule("list.sort_by", "list",
         r"\bsort(?:ed)?\s+by\s+",
         r"sortBy: ", "压缩 'sorted by' 为 'sortBy:'"),
    Rule("list.filter_where", "list",
         r"\bfilter(?:ed)?\s+(?:by|where|to\s+only\s+include)\s+",
         r"filter: ", "压缩 'filter by' 为 'filter:'"),
    Rule("list.map_each", "list",
         r"\bmap\s+(?:each\s+)?(?:item|element)?\s*to\s+",
         r"map: ", "压缩 'map each item to' 为 'map:'"),
    Rule("list.group_by", "list",
         r"\bgroup(?:ed)?\s+by\s+",
         r"groupBy: ", "压缩 'grouped by' 为 'groupBy:'"),

    # --- 5. 客套词 / 填充词 / hedging / 连接词(删客套词主力) --------------
    Rule("fluff.please", "fluff",
         r"\bplease\s+", r"", "删 'please'"),
    Rule("fluff.kindly", "fluff",
         r"\bkindly\s+", r"", "删 'kindly'"),
    Rule("fluff.could_you", "fluff",
         r"\b(?:could|can|would|will)\s+you\s+(?:please\s+)?", r"",
         "删 'could you / can you / would you please'"),
    Rule("fluff.i_would_like", "fluff",
         r"\bi\s+(?:would\s+like|want|need)\s+(?:you\s+)?to\s+", r"",
         "删 'I would like you to'"),
    Rule("fluff.help_me", "fluff",
         r"\bhelp\s+me\s+(?:to\s+)?", r"", "删 'help me to'"),
    Rule("fluff.in_order_to", "fluff",
         r"\bin\s+order\s+to\b", r"to", "替 'in order to' 为 'to'"),
    Rule("fluff.due_to_fact", "fluff",
         r"\bdue\s+to\s+the\s+fact\s+that\b", r"because", "替 'due to the fact that' 为 'because'"),
    Rule("fluff.at_this_point", "fluff",
         r"\bat\s+this\s+point\s+in\s+time\b", r"now", "替 'at this point in time' 为 'now'"),
    Rule("fluff.it_is_important", "fluff",
         r"\bit\s+is\s+(?:important|crucial|essential|necessary|vital)\s+(?:to\s+note\s+)?that\s+", r"",
         "删 'it is important to note that'"),
    Rule("fluff.as_a_matter", "fluff",
         r"\bas\s+a\s+matter\s+of\s+fact\s*,?\s*", r"", "删 'as a matter of fact'"),
    Rule("fluff.needless_to_say", "fluff",
         r"\bneedless\s+to\s+say\s*,?\s*", r"", "删 'needless to say'"),
    Rule("fluff.in_my_opinion", "fluff",
         r"\bin\s+my\s+(?:opinion|view)\s*,?\s*", r"", "删 'in my opinion'"),
    Rule("fluff.thanks", "fluff",
         r"\b(?:thanks?|thank\s+you)(?:\s+(?:so|very)\s+much)?\s*[!.]*\s*$", r"",
         "删句尾 'thank you'"),
    Rule("fluff.basically", "fluff",
         r"\b(basically|essentially|literally|actually|really|just)\s+", r"",
         "删 hedgin 副词"),
    Rule("fluff.very", "fluff",
         r"\b(very|quite|rather|somewhat|fairly)\s+", r"", "删程度副词"),
    Rule("fluff.you_should", "fluff",
         r"\byou\s+should\s+(?:probably\s+)?", r"", "删 'you should'"),
    Rule("fluff.make_sure", "fluff",
         r"\bmake\s+sure\s+(?:to\s+|that\s+)?", r"must ", "替 'make sure to' 为 'must'"),
    Rule("fluff.be_sure", "fluff",
         r"\bbe\s+sure\s+to\s+", r"must ", "替 'be sure to' 为 'must'"),
    Rule("fluff.dont_forget", "fluff",
         r"\bdon'?t\s+forget\s+to\s+", r"must ", "替 'do not forget to' 为 'must'"),
    Rule("fluff.try_to", "fluff",
         r"\btry\s+to\s+", r"", "删 'try to'"),
    Rule("fluff.go_ahead", "fluff",
         r"\bgo\s+ahead\s+and\s+", r"", "删 'go ahead and'"),

    # --- 6. 谓词通用式(checks if -> ?, 保持通用、不猜具体语义) ------------
    Rule("pred.checks_if", "predicate",
         r"\bchecks?\s+(?:if|whether)\s+", r"? ", "压缩 'checks if X' 为 '? X'"),
    Rule("pred.tells_if", "predicate",
         r"\btells?\s+(?:if|whether)\s+", r"? ", "压缩 'tells whether X' 为 '? X'"),
    Rule("pred.determines_if", "predicate",
         r"\bdetermines?\s+(?:if|whether)\s+", r"? ", "压缩 'determines whether X' 为 '? X'"),
    Rule("pred.given_a", "predicate",
         r"\bgiven\s+(?:a|an|the)\s+", r"given ", "删 'given' 后的冠词"),

    # --- 7. 动词短词化 -----------------------------------------------------
    Rule("verb.utilize", "verb", r"\butilize\b", r"use", "utilize -> use"),
    Rule("verb.demonstrate", "verb", r"\bdemonstrate\b", r"show", "demonstrate -> show"),
    Rule("verb.facilitate", "verb", r"\bfacilitate\b", r"help", "facilitate -> help"),
    Rule("verb.determine", "verb", r"\bdetermine\b", r"find", "determine -> find"),
    Rule("verb.investigate", "verb", r"\binvestigate\b", r"check", "investigate -> check"),
    Rule("verb.transform", "verb", r"\btransform\b", r"map", "transform -> map"),
    Rule("verb.ensure_that", "verb", r"\bensure\s+that\b", r"ensure", "删 'ensure that' 的 that"),

    # --- 8. 中文常用式 -----------------------------------------------------
    Rule("zh.please", "chinese", r"请", r"", "删 '请'"),
    Rule("zh.write_function", "chinese",
         r"(?:请)?写(?:一个)?(?:函数|方法)(?:用于|来|，)?",
         r"fn: ", "中文: '请写一个函数' -> 'fn:'"),
    Rule("zh.return", "chinese", r"返回(?:一个)?", r"-> ", "中文: '返回' -> '->'"),
    Rule("zh.if", "chinese", r"如果", r"if ", "中文: '如果' -> 'if'"),

    # --- 9. 收尾整理 -------------------------------------------------------
    Rule("tidy.multi_space", "tidy", r"[ \t]{2,}", r" ", "折叠连续空白"),
    Rule("tidy.space_punct", "tidy", r"\s+([,.;:?])", r"\1", "删标点前空格('!' 除外, 逻辑非)"),
    Rule("tidy.repeat_punct", "tidy", r"([.!?]){2,}", r"\1", "折叠重复标点"),
    Rule("tidy.trim", "tidy", r"^\s+|\s+$", r"", "去首尾空白"),
]


def apply_rules(text: str, rules: Iterable[Rule] | None = None,
                *, trace: bool = False) -> tuple[str, list[RuleHit]]:
    """按顺序应用规则。返回 (压缩结果, 命中规则列表)。"""
    rules = list(rules) if rules is not None else RULES
    hits: list[RuleHit] = []
    out = text
    for rule in rules:
        if trace:
            new_chunks: list[str] = []
            last = 0
            mutated = False
            for m in rule.compile().finditer(out):
                hits.append(RuleHit(rule.id, rule.category, m.group(0), m.expand(rule.replacement)))
                new_chunks.append(out[last:m.start()])
                new_chunks.append(m.expand(rule.replacement))
                last = m.end()
                mutated = True
            if mutated:
                new_chunks.append(out[last:])
                out = "".join(new_chunks)
        else:
            new_out, n = rule.compile().subn(rule.replacement, out)
            if n:
                hits.append(RuleHit(rule.id, rule.category, "", ""))
            out = new_out
    return out, hits


def compress(text: str) -> str:
    """压缩入口: 只返回压缩后的文本。"""
    return apply_rules(text)[0]


def explain(text: str) -> dict:
    """解释入口: 压缩 + 逐规则命中 + token 统计。"""
    compressed, hits = apply_rules(text, trace=True)
    st = savings(text, compressed)
    return {
        "input": text,
        "output": compressed,
        "rules_applied": [{"id": h.rule_id, "category": h.category,
                           "before": h.before, "after": h.after} for h in hits],
        "rule_count": len(hits),
        "stats": st,
    }


# ============================================================================
# 二、token 计数与省量统计(tiktoken 优先, 无则 regex 近似)
# ============================================================================

_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=8)
def _try_tiktoken(encoding_name: str):
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """尽力而为的 token 计数。tiktoken 优先; 否则 regex 词/标点计数 *1.3 近似。"""
    if not text:
        return 0
    enc = _try_tiktoken(encoding)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    base = len(_FALLBACK_TOKEN_RE.findall(text))
    return max(1, int(round(base * 1.3)))


def savings(original: str, compressed: str, encoding: str = "cl100k_base") -> dict:
    """返回 token 数与压缩比。"""
    o = count_tokens(original, encoding)
    c = count_tokens(compressed, encoding)
    return {
        "original_tokens": o,
        "compressed_tokens": c,
        "saved_tokens": o - c,
        "reduction": (o - c) / o if o else 0.0,
        "encoding": encoding,
    }


# ============================================================================
# 三、AST 骨架化 —— 只读结构, 不读肉体(读代码前先看骨架, 再按需精准读)
# ============================================================================

_PY = {"py", "pyw", "python"}
# 大括号类语言(通用分支尽力而为)
_BRACE_LIKE = {"js", "jsx", "ts", "tsx", "java", "c", "cpp", "cc", "h", "hpp",
               "go", "rs", "cs", "kt", "php", "swift", "scala", "dart"}


def _fmt_arg(arg, default: str | None = None) -> str:
    """单个形参 -> 'name[: ann][= default]'。ast.arg 上只有 annotation, 默认值单独对齐传入。"""
    a = arg.arg
    if arg.annotation is not None:
        a += ": " + ast.unparse(arg.annotation)
    if default is not None:
        a += "=" + default
    return a


def _sig_text(node) -> str:
    """拼函数/方法签名(名称 + 参数 + 返回值), 不带函数体。

    默认值在 ast.arguments 里是独立列表: .defaults 对齐 posonly+args 的尾部,
    .kw_defaults 对齐 kwonlyargs。这里按位置拼回去。
    """
    args = node.args
    out: list[str] = []
    pos_all = list(args.posonlyargs) + list(args.args)
    # defaults 对齐 pos_all 的末尾
    off = len(pos_all) - len(args.defaults)
    fmt = []
    for i, a in enumerate(pos_all):
        d = ast.unparse(args.defaults[i - off]) if i >= off else None
        fmt.append(_fmt_arg(a, d))
    if args.posonlyargs:
        # 在 posonly 之后插 '/' (python3.8+ 语法)
        fmt.insert(len(args.posonlyargs), "/")
    if args.vararg:
        v = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            v += ": " + ast.unparse(args.vararg.annotation)
        fmt.append(v)
    elif args.kwonlyargs:
        fmt.append("*")
    for i, a in enumerate(args.kwonlyargs):
        d = ast.unparse(args.kw_defaults[i]) if args.kw_defaults[i] is not None else None
        fmt.append(_fmt_arg(a, d))
    if args.kwarg:
        k = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            k += ": " + ast.unparse(args.kwarg.annotation)
        fmt.append(k)
    ret = ""
    if node.returns is not None:
        ret = " -> " + ast.unparse(node.returns)
    return f"{node.name}({', '.join(fmt)}){ret}"


def skeletonize_python(code: str) -> str:
    """Python: 用标准库 ast 提取 import/def/class + 签名, 丢函数体。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"# [skeleton] 解析失败: {e}\n"
    lines: list[str] = []

    def _visit_body(body, indent: int) -> None:
        pad = "    " * indent
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lines.append(pad + ast.unparse(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                dec = "".join(f"@{ast.unparse(d)}\n" for d in node.decorator_list)
                kw = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                if dec:
                    lines.append(pad + dec.rstrip())
                    lines.append(pad + kw + "def " + _sig_text(node) + ":")
                else:
                    lines.append(pad + kw + "def " + _sig_text(node) + ":")
            elif isinstance(node, ast.ClassDef):
                bases = ""
                if node.bases or node.keywords:
                    b = [ast.unparse(x) for x in node.bases]
                    k = [f"{x.arg}={ast.unparse(x.value)}" for x in node.keywords]
                    bases = "(" + ", ".join(b + k) + ")"
                lines.append(pad + "class " + node.name + bases + ":")
                _visit_body(node.body, indent + 1)

    _visit_body(tree.body, 0)
    if not lines:
        return "(空 / 无顶层可骨架化结构)\n"
    return "\n".join(lines) + "\n"


# 声明起始判定: 带关键字的定义/导入行、常量箭头函数、类内方法
_CALLABLE_DECL_RE = re.compile(
    r"^\s*(?:(?:export|default)\s+)*(?:async\s+)?"
    r"(?:function|func|fn|def|class|struct|interface|enum|trait|impl|module|namespace|package)\b"
)
_ARROW_DECL_RE = re.compile(
    r"^\s*(?:(?:export|default)\s+)*(?:const|let|var)\s+[\w$]+\s*(?:<[^>{}\n]*>)?\s*=\s*(?:async\s*)?(?:\(|[\w$])[^;]*=>"
)
_METHOD_DECL_RE = re.compile(
    r"^\s*(?:(?:export|default)\s+)*(?:public|private|protected|internal|static|abstract|override|async|get|set|\*)?\s*"
    r"[\w$]+(?:\.[\w$]+)*\s*\([^;{]*\)(?:\s*:\s*[\w$<>,.?\[\] ()]+)?\s*(?:\{|=>)"
)
_IMPORT_LINE_RE = re.compile(
    r"^\s*(?:import|from|using|include|#include|require|export\s+\*|export\s+default)\b"
)
_TYPEDEF_LINE_RE = re.compile(r"^\s*(?:export\s+|declare\s+)*type\s+\w+")
_SPAN_END_RE = re.compile(r"[{};)]\s*$")  # 声明跨度闭合判定


def _is_decl_start(trimmed: str) -> bool:
    return bool(_CALLABLE_DECL_RE.match(trimmed) or _ARROW_DECL_RE.match(trimmed)
                or _METHOD_DECL_RE.match(trimmed) or _IMPORT_LINE_RE.match(trimmed)
                or _TYPEDEF_LINE_RE.match(trimmed))


def skeletonize_other(code: str) -> str:
    """非 Python 代码的通用骨架化(最佳努力, 覆盖主流大括号语言)。

    思路: 只保留"声明跨度"——从一行声明(function/class/interface/type/import、
    常量箭头函数、类内方法)开始, 到签名闭合(行尾出现 `{`/`}`/`;`/`)`)为止的
    连续行(多行参数签名一并保留); 中间的实现行天然被丢弃。
    """
    out: list[str] = []
    buf: list[str] | None = None  # None = 当前不在声明跨度内

    def flush() -> None:
        nonlocal buf
        if buf is not None:
            out.append("\n".join(buf))
            buf = None

    for raw in code.splitlines():
        s = raw.rstrip()
        trimmed = s.strip()
        if not trimmed or trimmed.startswith(("//", "#", "/*", "*", "*/")):
            continue
        if buf is not None and _is_decl_start(trimmed):
            # 新声明出现而旧的没闭合: 收掉旧的, 防止吞掉后续(安全阀)
            flush()
        if _is_decl_start(trimmed):
            buf = [s]
            if _SPAN_END_RE.search(s):
                flush()
            continue
        if buf is not None:
            buf.append(s)
            if _SPAN_END_RE.search(s):
                flush()

    return ("\n".join(out) + "\n") if out else "(未能识别可骨架化结构)\n"


def skeletonize(code: str, lang: str | None = None) -> str:
    """对外入口。lang 缺省时按扩展名/内容启发判断。"""
    lang = (lang or "").lower().lstrip(".")
    if lang in _PY:
        return skeletonize_python(code)
    if lang in _BRACE_LIKE or not lang:
        return skeletonize_other(code)
    # 未知语言: 退化为通用骨架化
    return skeletonize_other(code)


def skeletonize_file(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    with open(path, encoding="utf-8", errors="replace") as f:
        code = f.read()
    return skeletonize(code, ext)


# ============================================================================
# 四、CLI
# ============================================================================

def _read_input(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read()
    if args.text:
        return " ".join(args.text)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("没有输入。请传参、用 --file, 或从 stdin 管道输入。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="squeeze",
        description="sunge · squeeze —— 孙割自包含省-token 工具(压缩 / 计数 / 代码骨架化)。",
    )
    p.add_argument("text", nargs="*", help="要处理的文本(省略则读 --file 或 stdin)。")
    p.add_argument("-f", "--file", help="从文件读取输入。")
    p.add_argument("--count", action="store_true", help="只数 token(不压缩)。")
    p.add_argument("--skeleton", metavar="FILE", help="对代码文件做 AST 骨架化(读结构不读肉体)。")
    p.add_argument("--lang", help="骨架化时指定语言(缺省按扩展名推断)。")
    p.add_argument("--explain", action="store_true", help="压缩时把命中的规则打到 stderr(解释)。")
    p.add_argument("--json", action="store_true", help="输出 JSON 记录。")
    p.add_argument("--encoding", default="cl100k_base", help="tiktoken 编码(默认 cl100k_base)。")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- 骨架化(独立子命令) ---
    if args.skeleton:
        if args.lang:
            ext = args.lang.lower().lstrip(".")
        else:
            ext = args.skeleton.rsplit(".", 1)[-1].lower() if "." in args.skeleton else ""
        with open(args.skeleton, encoding="utf-8", errors="replace") as f:
            code = f.read()
        skel = skeletonize(code, ext)
        if args.json:
            import json
            print(json.dumps({"file": args.skeleton, "skeleton": skel}, ensure_ascii=False))
        else:
            sys.stdout.write(skel)
        return 0

    text = _read_input(args)

    if args.count:
        n = count_tokens(text, args.encoding)
        if args.json:
            import json
            print(json.dumps({"tokens": n, "encoding": args.encoding}, ensure_ascii=False))
        else:
            print(f"{n}")
        return 0

    # --- 压缩(默认) ---
    if args.explain:
        info = explain(text)
        import json
        # token 统计用指定 encoding 重算一遍保持一致
        st = savings(text, info["output"], args.encoding)
        if args.json:
            print(json.dumps({**info, "stats": st}, ensure_ascii=False))
        else:
            sys.stdout.write(info["output"] + "\n")
            sys.stderr.write("\n[squeeze 明细] 命中 %d 条规则:\n" % info["rule_count"])
            for h in info["rules_applied"]:
                if h["before"] or h["after"]:
                    sys.stderr.write(f"  [{h['id']}] {h['category']}: "
                                     f"{h['before']!r} -> {h['after']!r}\n")
            sys.stderr.write(f"[squeeze] {st['original_tokens']} -> {st['compressed_tokens']} "
                             f"tokens ({st['encoding']}, -{st['reduction']*100:.1f}%, "
                             f"省 {st['saved_tokens']})\n")
        return 0

    out = compress(text)
    st = savings(text, out, args.encoding)
    if args.json:
        import json
        print(json.dumps({"output": out, "stats": st}, ensure_ascii=False))
    else:
        sys.stdout.write(out + "\n")
        sys.stderr.write(f"[squeeze] {st['original_tokens']} -> {st['compressed_tokens']} "
                         f"tokens ({st['encoding']}, -{st['reduction']*100:.1f}%)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
