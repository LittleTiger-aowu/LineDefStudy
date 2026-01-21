from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


ANCHOR_TYPES = [
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "switch_expression",
    "try_statement",
    "catch_clause",
    "throw_statement",
    "return_statement",
]

TYPE_TO_ID = {"UNK": 0, "window": 1}
for _idx, _t in enumerate(ANCHOR_TYPES, start=len(TYPE_TO_ID)):
    TYPE_TO_ID[_t] = _idx


@dataclass
class Block:
    block_id: int
    type_id: int
    type_str: str
    span: Tuple[int, int]
    text: str
    stats: dict
    parse_ok: bool
    parse_has_error: bool


def slice_text_by_span(src: str, span: Tuple[int, int]) -> str:
    lines = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start, end = span
    if not lines:
        return ""
    start_idx = max(start - 1, 0)
    end_idx = min(end, len(lines))
    return "\n".join(lines[start_idx:end_idx])


def _iter_nodes(root) -> Iterable:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _parse_with_tree_sitter(src: str, lang: str):
    try:
        from tree_sitter_languages import get_language, get_parser
    except Exception:
        return None

    try:
        _ = get_language(lang)
        parser = get_parser(lang)
    except Exception:
        return None

    tree = parser.parse(bytes(src, "utf-8"))
    return tree


def _collect_anchor_spans(tree, total_lines: int) -> List[Tuple[int, int, str]]:
    candidates = []
    root = tree.root_node
    for node in _iter_nodes(root):
        if node.type in ANCHOR_TYPES:
            start_line = int(node.start_point[0]) + 1
            end_line = int(node.end_point[0]) + 1
            if start_line < 1:
                start_line = 1
            if end_line > total_lines:
                end_line = total_lines
            if start_line <= end_line:
                candidates.append((start_line, end_line, node.type))
    return candidates


def _select_non_overlapping(candidates: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    selected: List[Tuple[int, int, str]] = []
    last_end = 0
    for start, end, node_type in candidates:
        if not selected:
            selected.append((start, end, node_type))
            last_end = end
            continue
        if start > last_end:
            selected.append((start, end, node_type))
            last_end = end
            continue
        prev_start, prev_end, prev_type = selected[-1]
        prev_len = prev_end - prev_start
        cur_len = end - start
        if cur_len > prev_len:
            selected[-1] = (start, end, node_type)
            last_end = end
    return selected


def _fill_holes(
    spans: List[Tuple[int, int, str]],
    total_lines: int,
    win_size_lines: int,
    parse_ok: bool,
) -> List[Tuple[int, int, str, dict, bool]]:
    blocks = []
    last_end = 0
    for start, end, node_type in spans:
        if start > last_end + 1:
            hole_start = last_end + 1
            hole_end = start - 1
            blocks.extend(_window_blocks(hole_start, hole_end, win_size_lines, parse_ok, is_fallback=0))
        blocks.append(
            (
                start,
                end,
                node_type,
                {"span_len": end - start + 1, "is_fallback": 0, "anchor_flag": 1},
                parse_ok,
            )
        )
        last_end = end
    if last_end < total_lines:
        blocks.extend(_window_blocks(last_end + 1, total_lines, win_size_lines, parse_ok, is_fallback=0))
    return blocks


def _window_blocks(
    start_line: int, end_line: int, win_size_lines: int, parse_ok: bool, is_fallback: int
) -> List[Tuple[int, int, str, dict, bool]]:
    blocks = []
    cur = start_line
    while cur <= end_line:
        win_end = min(cur + win_size_lines - 1, end_line)
        blocks.append(
            (
                cur,
                win_end,
                "window",
                {"span_len": win_end - cur + 1, "is_fallback": is_fallback, "anchor_flag": 0},
                parse_ok,
            )
        )
        cur = win_end + 1
    return blocks


def _tree_parse_flags(tree) -> tuple[bool, bool]:
    if tree is None:
        return False, False
    root = tree.root_node
    has_error = bool(getattr(root, "has_error", False))
    is_missing = bool(getattr(root, "is_missing", False))
    return True, bool(has_error or is_missing)


def extract_blocks(src: str, lang: str = "java", win_size_lines: int = 20) -> List[Block]:
    normalized = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    total_lines = max(1, len(lines))

    tree = _parse_with_tree_sitter(normalized, lang)
    parse_ok, parse_has_error = _tree_parse_flags(tree)

    if parse_ok:
        candidates = _collect_anchor_spans(tree, total_lines)
        spans = _select_non_overlapping(candidates)
        if spans:
            raw_blocks = _fill_holes(spans, total_lines, win_size_lines, parse_ok)
        else:
            raw_blocks = _window_blocks(1, total_lines, win_size_lines, parse_ok, is_fallback=0)
    else:
        raw_blocks = _window_blocks(1, total_lines, win_size_lines, parse_ok, is_fallback=1)

    raw_blocks = sorted(raw_blocks, key=lambda x: x[0])
    blocks: List[Block] = []
    for idx, (start, end, type_str, stats, blk_parse_ok) in enumerate(raw_blocks):
        text = slice_text_by_span(normalized, (start, end))
        type_id = TYPE_TO_ID.get(type_str, TYPE_TO_ID["UNK"])
        blocks.append(
            Block(
                block_id=idx,
                type_id=type_id,
                type_str=type_str,
                span=(start, end),
                text=text,
                stats=stats,
                parse_ok=blk_parse_ok,
                parse_has_error=parse_has_error,
            )
        )
    if not blocks:
        blocks.append(
            Block(
                block_id=0,
                type_id=TYPE_TO_ID["window"],
                type_str="window",
                span=(1, 1),
                text=normalized if normalized else "",
                stats={"span_len": 1, "is_fallback": 1, "anchor_flag": 0},
                parse_ok=False,
                parse_has_error=False,
            )
        )
    return blocks
