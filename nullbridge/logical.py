"""Typed logical domain. This module has no dependency on an Arrow consumer."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math
from typing import Any

@dataclass(frozen=True)
class Type:
    kind: str
    fields: tuple[tuple[str, "Type"], ...] = ()

    def __post_init__(self):
        if self.kind not in {"int32", "int64", "float64", "bool", "string", "binary", "list", "struct"}:
            raise ValueError(f"unsupported type {self.kind}")
        if self.kind == "list" and len(self.fields) != 1:
            raise ValueError("a list must have exactly one child")
        if self.kind not in {"list", "struct"} and self.fields:
            raise ValueError("primitive type cannot have children")
        if self.kind == "struct" and len({n for n, _ in self.fields}) != len(self.fields):
            raise ValueError("duplicate struct field names")

    def label(self) -> str:
        if not self.fields:
            return self.kind
        return self.kind + "<" + ",".join(n + ":" + t.label() for n, t in self.fields) + ">"

    def to_json(self):
        return {"kind": self.kind, "fields": [[n, t.to_json()] for n, t in self.fields]}

    @staticmethod
    def from_json(x):
        return Type(x["kind"], tuple((n, Type.from_json(t)) for n, t in x.get("fields", [])))

I32, I64, F64, BOOL, STR, BIN = map(Type, ("int32", "int64", "float64", "bool", "string", "binary"))
def LIST(t: Type) -> Type:
    return Type("list", (("item", t),))
def STRUCT(**fields: Type) -> Type:
    return Type("struct", tuple(fields.items()))

def canonical(v: Any, *, sql: bool = False) -> tuple:
    """Type-tagged equality: NaNs coalesce, null/list/binary/bool never alias.

    SQL comparison deliberately quotients the sign of zero; the stronger
    representation certificate preserves it. NaN payloads are not values here.
    """
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        if math.isnan(v):
            return ("nan",)
        if sql and v == 0.0:
            return ("float", 0.0.hex())
        return ("float", v.hex())
    if isinstance(v, str):
        return ("string", v)
    if isinstance(v, bytes):
        return ("binary", v.hex())
    if isinstance(v, (list, tuple)):
        return ("list", tuple(canonical(x, sql=sql) for x in v))
    if isinstance(v, dict):
        return ("struct", tuple((k, canonical(x, sql=sql)) for k, x in v.items()))
    raise TypeError(type(v))

def same_values(xs, ys) -> bool:
    return [canonical(x) for x in xs] == [canonical(y) for y in ys]

def bag(rows) -> Counter:
    return Counter(tuple(canonical(x, sql=True) for x in row) for row in rows)

def tagged_json(v):
    """Strict JSON representation, including NaN, infinity, signed zero, bytes."""
    if isinstance(v, float):
        return {"$float": v.hex() if not math.isnan(v) else "nan"}
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, (list, tuple)):
        return [tagged_json(x) for x in v]
    if isinstance(v, dict):
        return {k: tagged_json(x) for k, x in v.items()}
    return v

def untagged_json(v):
    if isinstance(v, dict):
        if set(v) == {"$float"}:
            return float.fromhex(v["$float"])
        if set(v) == {"$bytes"}:
            return bytes.fromhex(v["$bytes"])
        return {k: untagged_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [untagged_json(x) for x in v]
    return v

def check_value(t: Type, v: Any) -> None:
    if v is None:
        return
    if t.kind in {"int32", "int64"}:
        bits = 32 if t.kind == "int32" else 64
        if type(v) is not int or not -(1 << (bits-1)) <= v < (1 << (bits-1)):
            raise ValueError(f"{v!r} is not {t.kind}")
    elif t.kind == "float64":
        if type(v) is not float:
            raise ValueError("float64 requires an explicit float")
    elif t.kind == "bool":
        if type(v) is not bool:
            raise ValueError("boolean required")
    elif t.kind == "string":
        if not isinstance(v, str):
            raise ValueError("string required")
        v.encode("utf-8")
    elif t.kind == "binary":
        if not isinstance(v, bytes):
            raise ValueError("bytes required")
    elif t.kind == "list":
        if not isinstance(v, list):
            raise ValueError("list required")
        for x in v:
            check_value(t.fields[0][1], x)
    elif t.kind == "struct":
        if not isinstance(v, dict) or tuple(v) != tuple(n for n, _ in t.fields):
            raise ValueError("struct fields/order do not match type")
        for n, child in t.fields:
            check_value(child, v[n])

@dataclass(frozen=True)
class LogicalTable:
    fields: tuple[tuple[str, Type], ...]
    rows: tuple[tuple[Any, ...], ...]

    def validate(self):
        if not self.fields or len({n for n, _ in self.fields}) != len(self.fields):
            raise ValueError("nonempty, distinct column names required")
        for row in self.rows:
            if len(row) != len(self.fields):
                raise ValueError("row width mismatch")
            for (_, t), v in zip(self.fields, row):
                check_value(t, v)

    def to_json(self):
        return {"fields": [[n, t.to_json()] for n, t in self.fields], "rows": tagged_json(self.rows)}

    @staticmethod
    def from_json(x):
        return LogicalTable(tuple((n, Type.from_json(t)) for n, t in x["fields"]),
                            tuple(tuple(r) for r in untagged_json(x["rows"])))
