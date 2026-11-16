"""Deterministic reading of one model reply as JSON (no model call).

The reader drops markdown fences and surrounding prose, fixes common syntax
slips only outside double-quoted strings, salvages the complete array
elements of a reply cut off at the output cap (never closing it by guessing)
and decodes once more an object or array that arrived double-encoded as a
JSON string. It never raises for any ``str`` input.

On top of the reader, ``validate`` checks a value against a small JSON schema
subset applying lossless fixes only, ``parse_json`` combines both, and
``generate_json`` / ``agenerate_json`` drive a caller's model call through
read, validate and up to N corrective re-asks built from a caller template.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

__all__ = [
    "JsonReading",
    "JsonResult",
    "JsonStatus",
    "Repair",
    "agenerate_json",
    "generate_json",
    "parse_json",
    "read_json",
    "validate",
]

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


_INT_TEXT = r"\s*[+-]?\d+\s*"
_PLACEHOLDER = r"\{(prompt|error|reply|schema)\}"
_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "null": lambda v: v is None,
}
_MISSING = object()


def _to_boolean(value: Any) -> Any:
    """``True``/``False`` spelled by a string, else ``_MISSING``."""
    if not isinstance(value, str):
        return _MISSING
    return {"true": True, "false": False}.get(value.strip().lower(), _MISSING)


def _to_integer(value: Any) -> Any:
    """Int spelled by an integral string or held by an integral float, else ``_MISSING``."""
    if isinstance(value, str) and re.fullmatch(_INT_TEXT, value):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return _MISSING


def _to_number(value: Any) -> Any:
    """Finite number spelled by a string (an int for integral text), else ``_MISSING``."""
    if not isinstance(value, str):
        return _MISSING
    if re.fullmatch(_INT_TEXT, value):
        return int(value)
    try:
        number = float(value)
    except ValueError:
        return _MISSING
    return number if math.isfinite(number) else _MISSING


def _to_array(value: Any) -> Any:
    """Copy of the list held by the only key of an object, else ``_MISSING``."""
    if not (isinstance(value, dict) and len(value) == 1):
        return _MISSING
    inner = next(iter(value.values()))
    return list(inner) if isinstance(inner, list) else _MISSING


_FIXES: dict[str, Callable[[Any], Any]] = {
    "boolean": _to_boolean,
    "integer": _to_integer,
    "number": _to_number,
    "array": _to_array,
}


def _coerce(value: Any, name: str) -> Any:
    """Lossless conversion of ``value`` to the JSON type ``name``, or ``_MISSING``."""
    fix = _FIXES.get(name)
    if fix is None or isinstance(value, bool):
        return _MISSING
    return fix(value)


def _fix_type(value: Any, schema: Mapping[str, Any], path: str) -> tuple[Any, list[str]]:
    """Apply ``type``: keep a matching value, else try each lossless fix in order."""
    declared = schema.get("type")
    if declared is None:
        return value, []
    names = [declared] if isinstance(declared, str) else list(declared)
    known = [n for n in names if n in _TYPES]
    if not known or any(_TYPES[n](value) for n in known):
        return value, []
    for name in known:
        fixed = _coerce(value, name)
        if fixed is not _MISSING:
            return fixed, []
    return value, [f"{path}: expected {' or '.join(known)}, got {type(value).__name__}"]


def _norm(key: str) -> str:
    """Key reduced to its lowercase alphanumeric characters."""
    return "".join(c for c in key.lower() if c.isalnum())


def _fix_object(
    value: dict[str, Any], schema: Mapping[str, Any], path: str
) -> tuple[dict[str, Any], list[str]]:
    """Rename near-matching keys, validate declared properties, check ``required``."""
    props: Mapping[str, Any] = schema.get("properties") or {}
    out: dict[str, Any] = {}
    errors: list[str] = []
    for key, item in value.items():
        target = key
        if key not in props:
            matches = [p for p in props if _norm(p) == _norm(str(key))]
            if len(matches) == 1 and matches[0] not in value and matches[0] not in out:
                target = matches[0]
        fixed = item
        if target in props:
            fixed, sub = _check(item, props[target], f"{path}.{target}", ignore_size=False)
            errors.extend(sub)
        out[target] = fixed
    errors.extend(
        f"{path}: missing required key {key!r}"
        for key in schema.get("required") or ()
        if key not in out
    )
    return out, errors


def _fix_array(
    value: list[Any], schema: Mapping[str, Any], path: str, *, ignore_size: bool
) -> tuple[list[Any], list[str]]:
    """Validate ``items`` recursively and, unless ``ignore_size``, minItems/maxItems."""
    items = schema.get("items")
    out: list[Any] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        fixed = item
        if isinstance(items, Mapping):
            fixed, sub = _check(item, items, f"{path}[{index}]", ignore_size=False)
            errors.extend(sub)
        out.append(fixed)
    low, high = schema.get("minItems"), schema.get("maxItems")
    if not ignore_size and low is not None and len(out) < low:
        errors.append(f"{path}: {len(out)} items, at least {low} required")
    if not ignore_size and high is not None and len(out) > high:
        errors.append(f"{path}: {len(out)} items, at most {high} allowed")
    return out, errors


def _fix_enum(value: Any, schema: Mapping[str, Any], path: str) -> tuple[Any, list[str]]:
    """Accept an enum member, or a string equal to exactly one member but for case."""
    members = schema.get("enum")
    if members is None or value in list(members):
        return value, []
    if isinstance(value, str):
        folded = [m for m in members if isinstance(m, str) and m.casefold() == value.casefold()]
        if len(folded) == 1:
            return folded[0], []
    return value, [f"{path}: {value!r} is not one of {list(members)!r}"]


def _check(
    value: Any, schema: Mapping[str, Any], path: str, *, ignore_size: bool
) -> tuple[Any, list[str]]:
    """Validate ``value`` against ``schema``, returning a new fixed value and errors."""
    value, errors = _fix_type(value, schema, path)
    if errors:
        return value, errors
    if isinstance(value, dict):
        value, errors = _fix_object(value, schema, path)
    elif isinstance(value, list):
        value, errors = _fix_array(value, schema, path, ignore_size=ignore_size)
    value, enum_errors = _fix_enum(value, schema, path)
    return value, errors + enum_errors


def _same(left: Any, right: Any) -> bool:
    """Equality that also requires identical types and key order at every level."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(map(_same, left, right))
    return bool(left == right)


def validate(value: Any, schema: Mapping[str, Any] | None) -> tuple[Any, tuple[str, ...]]:
    """Check ``value`` against ``schema`` with lossless fixes; never mutates ``value``.

    Returns ``(fixed value, errors)``; ``errors == ()`` means valid. Understood
    keywords: type, properties, required, items, enum, minItems, maxItems.
    """
    if not schema:
        return value, ()
    fixed, errors = _check(value, schema, "$", ignore_size=False)
    return fixed, tuple(errors)


class JsonStatus(enum.StrEnum):
    """How the value of a ``JsonResult`` was obtained."""

    VALID = "valid"
    REPAIRED = "repaired"
    SALVAGED = "salvaged"
    REASKED = "reasked"
    UNREADABLE = "unreadable"


@dataclasses.dataclass(frozen=True)
class JsonResult:
    """Outcome of reading and validating replies; never holds an invented value."""

    ok: bool
    value: Any = None
    parsed: Any = None
    error: str | None = None
    raw: str = ""
    repairs: tuple[Repair, ...] = ()
    coerced: bool = False
    truncated: bool = False
    reasks: int = 0

    @property
    def status(self) -> JsonStatus:
        """Summarise how the value was obtained."""
        if not self.ok:
            return JsonStatus.UNREADABLE
        if self.reasks:
            return JsonStatus.REASKED
        if Repair.SALVAGE in self.repairs:
            return JsonStatus.SALVAGED
        if self.repairs or self.coerced:
            return JsonStatus.REPAIRED
        return JsonStatus.VALID

    def best(self) -> Any:
        """Return the value when ok, else the decoded but schema-invalid value."""
        return self.value if self.ok else self.parsed


def _salvage_schema(schema: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the schema to check a salvaged list with: an array schema, else None."""
    declared = (schema or {}).get("type")
    names = [declared] if isinstance(declared, str) else list(declared or ())
    return schema if "array" in names else None


def parse_json(
    text: str, schema: Mapping[str, Any] | None = None, *, salvage_key: str | None = None
) -> JsonResult:
    """Read ``text`` with ``read_json`` and validate it against ``schema`` (no model call).

    A salvaged list is checked against an array schema without minItems/maxItems,
    and not checked against any other (wrapper) schema.
    """
    reading = read_json(text, salvage_key=salvage_key)
    common: dict[str, Any] = {
        "raw": text,
        "repairs": reading.repairs,
        "truncated": reading.truncated,
    }
    if not reading.ok:
        return JsonResult(ok=False, error=reading.error or "unreadable reply", **common)
    errors: list[str] = []
    fixed = reading.value
    check = _salvage_schema(schema) if reading.truncated else schema
    if check:
        fixed, errors = _check(fixed, check, "$", ignore_size=reading.truncated)
    common["coerced"] = not _same(fixed, reading.value)
    if errors:
        return JsonResult(ok=False, parsed=fixed, error="; ".join(errors), **common)
    return JsonResult(ok=True, value=fixed, **common)


def _reply_text(reply: Any) -> str:
    """Text of a reply given as text, a ``(text, usage)`` tuple or None."""
    if isinstance(reply, tuple):
        reply = reply[0] if reply else None
    return "" if reply is None else str(reply)


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Replace each placeholder of ``template`` in one pass (inserted text is not re-scanned)."""
    return re.sub(_PLACEHOLDER, lambda m: values[m.group(1)], template)


@dataclasses.dataclass
class _Flow:
    """State shared by the sync and async drivers."""

    prompt: str
    schema: Mapping[str, Any] | None
    repair_template: str
    attempts: int
    salvage_key: str | None
    reask_truncated: bool
    result: JsonResult = dataclasses.field(default_factory=lambda: JsonResult(ok=False))

    def take(self, reply: Any, reasks: int) -> None:
        """Read one reply, keeping an earlier decoded value when this one decodes nothing."""
        new = parse_json(_reply_text(reply), self.schema, salvage_key=self.salvage_key)
        parsed = new.parsed
        if not new.ok and parsed is None:
            parsed = self.result.parsed
        self.result = dataclasses.replace(new, parsed=parsed, reasks=reasks)

    def next_prompt(self) -> str | None:
        """Return the re-ask prompt to send next, or None when the flow is over."""
        result = self.result
        if result.ok or result.reasks >= self.attempts:
            return None
        if result.truncated and not self.reask_truncated:
            return None
        values = {
            "prompt": self.prompt,
            "error": result.error or "",
            "reply": result.raw,
            "schema": json.dumps(self.schema or {}, ensure_ascii=False),
        }
        return _fill(self.repair_template, values)


def generate_json(
    ask: Callable[[str], Any],
    prompt: str,
    schema: Mapping[str, Any] | None = None,
    *,
    repair_template: str,
    attempts: int = 1,
    salvage_key: str | None = None,
    reask_truncated: bool = True,
) -> JsonResult:
    """Ask once, then re-ask with ``repair_template`` up to ``attempts`` times until valid."""
    flow = _Flow(prompt, schema, repair_template, attempts, salvage_key, reask_truncated)
    flow.take(ask(prompt), 0)
    while (reask := flow.next_prompt()) is not None:
        flow.take(ask(reask), flow.result.reasks + 1)
    return flow.result


async def agenerate_json(
    ask: Callable[[str], Any],
    prompt: str,
    schema: Mapping[str, Any] | None = None,
    *,
    repair_template: str,
    attempts: int = 1,
    salvage_key: str | None = None,
    reask_truncated: bool = True,
) -> JsonResult:
    """Run ``generate_json``'s flow; ``ask`` may return an awaitable, which is awaited."""

    async def call(text: str) -> Any:
        reply = ask(text)
        return await reply if isinstance(reply, Awaitable) else reply

    flow = _Flow(prompt, schema, repair_template, attempts, salvage_key, reask_truncated)
    flow.take(await call(prompt), 0)
    while (reask := flow.next_prompt()) is not None:
        flow.take(await call(reask), flow.result.reasks + 1)
    return flow.result
