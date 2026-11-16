"""Deterministic reading of one model reply as JSON (no model call).

The reader drops markdown fences and surrounding prose, fixes common syntax
slips only outside double-quoted strings, salvages the complete array
elements of a reply cut off at the output cap (never closing it by guessing)
and decodes once more an object or array that arrived double-encoded as a
JSON string. It never raises for any ``str`` input.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
from typing import Any

__all__ = ["JsonReading", "Repair", "read_json"]

_OPENERS = "{["
_CLOSERS = "}]"
_LITERALS = {"True": "true", "False": "false", "None": "null"}
_DECODER = json.JSONDecoder()


class Repair(enum.StrEnum):
    """One member per deterministic repair step."""

    FENCES = "fences"
    OUTERMOST = "outermost"
    SYNTAX = "syntax"
    SALVAGE = "salvage"
    DOUBLE_ENCODED = "double_encoded"


@dataclasses.dataclass(frozen=True)
class JsonReading:
    """Outcome of reading one reply: the value, or a non-empty error."""

    ok: bool
    value: Any = None
    error: str | None = None
    truncated: bool = False
    repairs: tuple[Repair, ...] = ()


def _loads(text: str) -> tuple[bool, Any]:
    """Return ``(True, value)`` when ``text`` is strict JSON, else ``(False, None)``."""
    try:
        return True, json.loads(text)
    except (ValueError, RecursionError):
        return False, None


def _fail(error: str, *, truncated: bool = False) -> JsonReading:
    """Build a failed reading."""
    return JsonReading(ok=False, error=error, truncated=truncated)


def _string_end(text: str, start: int) -> int:
    """Index just past the string quoted at ``start``, or ``len(text)`` if unterminated."""
    quote, i = text[start], start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == quote:
            return i + 1
        else:
            i += 1
    return len(text)


def _first_opener(text: str) -> int:
    """Index of the first ``{`` or ``[``, or -1."""
    found = [i for i in (text.find(c) for c in _OPENERS) if i >= 0]
    return min(found, default=-1)


def _structure_end(text: str, start: int) -> int | None:
    """Index just past the structure opened at ``start``; None when still open at the end."""
    depth, i = 0, start
    while i < len(text):
        char = text[i]
        if char in "\"'":
            i = _string_end(text, i)
            continue
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _single_to_double(body: str) -> str:
    """Re-quote the content of a single-quoted string as a double-quoted one."""
    out, i = [], 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            out.append("'" if body[i + 1] == "'" else body[i : i + 2])
            i += 2
            continue
        out.append('\\"' if char == '"' else char)
        i += 1
    return "".join(out)


def _fix_syntax(text: str) -> str:
    """Fix trailing commas, single quotes and Python literals outside double-quoted strings."""
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"':
            end = _string_end(text, i)
            out.append(text[i:end])
            i = end
        elif char == "'":
            end = _string_end(text, i)
            closed = end > i + 1 and text[end - 1] == "'"
            body = text[i + 1 : end - 1] if closed else text[i + 1 : end]
            out.append('"' + _single_to_double(body) + ('"' if closed else ""))
            i = end
        elif char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] not in _CLOSERS:
                out.append(char)
            i += 1
        elif char.isalpha() or char == "_":
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            out.append(_LITERALS.get(word, word))
            i = j
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _salvage_start(text: str, salvage_key: str | None) -> int:
    """Index of the ``[`` of the array to salvage, or -1."""
    if salvage_key is not None:
        match = re.search('"' + re.escape(salvage_key) + r'"\s*:\s*\[', text)
        if match is not None:
            return match.end() - 1
    i = 0
    while i < len(text):
        if text[i] in "\"'":
            i = _string_end(text, i)
        elif text[i] == "[":
            return i
        else:
            i += 1
    return -1


def _salvage(text: str, salvage_key: str | None) -> list[Any]:
    """Decode the complete elements of the chosen array, stopping at the first incomplete one."""
    start = _salvage_start(text, salvage_key)
    if start < 0:
        return []
    items: list[Any] = []
    pos = start + 1
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            return items
        try:
            item, end = _DECODER.raw_decode(text, pos)
        except (ValueError, RecursionError):
            return items
        number = isinstance(item, int | float) and not isinstance(item, bool)
        if end >= len(text) and number:
            return items
        items.append(item)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == ",":
            pos += 1
        elif pos >= len(text) or text[pos] != "]":
            return items


def _strip_fence(text: str) -> str | None:
    """Content of a code fence opened before the JSON, or None when there is none."""
    fence = text.find("```")
    opener = _first_opener(text)
    if fence < 0 or (0 <= opener < fence):
        return None
    match = re.match(r"```[\w+-]*", text[fence:])
    content = text[fence + (match.end() if match else 3) :]
    close = content.find("```")
    return content if close < 0 else content[:close]


def _read_cut_off(text: str, salvage_key: str | None, repairs: list[Repair]) -> JsonReading:
    """Salvage the complete array elements of a structure still open at its end."""
    fixed = _fix_syntax(text)
    items = _salvage(fixed, salvage_key)
    if not items:
        return _fail("reply cut off with no complete array element", truncated=True)
    if fixed != text:
        repairs.append(Repair.SYNTAX)
    repairs.append(Repair.SALVAGE)
    return JsonReading(ok=True, value=items, truncated=True, repairs=tuple(repairs))


def _read_structure(text: str, salvage_key: str | None) -> JsonReading:
    """Read ``text`` without the double-encoded step."""
    body = text.strip()
    if not body:
        return _fail("empty reply")
    ok, value = _loads(body)
    if ok:
        return JsonReading(ok=True, value=value)
    repairs: list[Repair] = []
    fenced = _strip_fence(body)
    if fenced is not None:
        repairs.append(Repair.FENCES)
        body = fenced.strip()
    start = _first_opener(body)
    if start < 0:
        return _fail("no JSON object or array in reply")
    end = _structure_end(body, start)
    if end is None:
        return _read_cut_off(body[start:], salvage_key, repairs)
    candidate = body[start:end]
    if start > 0 or end < len(body):
        repairs.append(Repair.OUTERMOST)
    ok, value = _loads(candidate)
    if not ok:
        fixed = _fix_syntax(candidate)
        ok, value = _loads(fixed)
        if not ok:
            return _fail("reply is malformed JSON that no repair fixes")
        repairs.append(Repair.SYNTAX)
    return JsonReading(ok=True, value=value, repairs=tuple(repairs))


def read_json(text: str, *, salvage_key: str | None = None) -> JsonReading:
    """Read one model reply as JSON deterministically.

    ``salvage_key`` names the array inside a cut-off wrapper object whose
    elements are salvaged; when None or absent, the first ``[`` is used.
    """
    reading = _read_structure(text, salvage_key)
    if not (reading.ok and isinstance(reading.value, str)):
        return reading
    inner_text = reading.value.strip()
    if not inner_text or inner_text[0] not in _OPENERS:
        return reading
    inner = _read_structure(inner_text, None)
    if not inner.ok or inner.truncated or not isinstance(inner.value, dict | list):
        return reading
    repairs = (*reading.repairs, Repair.DOUBLE_ENCODED, *inner.repairs)
    return JsonReading(ok=True, value=inner.value, repairs=repairs)
