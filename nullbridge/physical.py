"""Independent little-endian buffer interpreter for the admitted Arrow layouts.

No to_pylist(), numpy, pandas, Arrow compute kernel, or SQL is used here.
Descriptors own immutable bytes. This is not an IPC parser or a C-pointer reader.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import base64
import struct
from .logical import Type, LogicalTable, canonical

FORMATS = {"int8": "b", "int16": "h", "int32": "i", "int64": "q", "float64": "d"}
class InvalidRepresentation(ValueError):
    pass

@dataclass(frozen=True)
class Array:
    logical_type: Type
    length: int
    offset: int
    buffers: tuple[bytes | None, ...]
    children: tuple["Array", ...] = ()
    dictionary: "Array | None" = None
    index_type: str = "int32"

    def slice(self, start: int, length: int | None = None):
        length = self.length - start if length is None else length
        if start < 0 or length < 0 or start + length > self.length:
            raise IndexError("slice outside logical window")
        return replace(self, offset=self.offset + start, length=length)

    def to_json(self):
        return {"type": self.logical_type.to_json(), "length": self.length,
                "offset": self.offset, "buffers": [None if x is None else base64.b64encode(x).decode() for x in self.buffers],
                "children": [a.to_json() for a in self.children],
                "dictionary": None if self.dictionary is None else self.dictionary.to_json(),
                "index_type": self.index_type}

    @staticmethod
    def from_json(x):
        return Array(Type.from_json(x["type"]), x["length"], x["offset"],
                     tuple(None if v is None else base64.b64decode(v, validate=True) for v in x["buffers"]),
                     tuple(Array.from_json(v) for v in x["children"]),
                     None if x["dictionary"] is None else Array.from_json(x["dictionary"]), x["index_type"])

    def byte_size(self):
        return sum(len(x) for x in self.buffers if x is not None) + sum(a.byte_size() for a in self.children) + (self.dictionary.byte_size() if self.dictionary else 0)

def bit(buf: bytes | None, i: int) -> bool:
    if buf is None:
        return True
    if i < 0 or i // 8 >= len(buf):
        raise InvalidRepresentation("bitmap address outside buffer")
    return bool(buf[i // 8] & (1 << (i % 8)))

def number(buf: bytes | None, i: int, kind: str):
    if buf is None or i < 0:
        raise InvalidRepresentation("missing numerical buffer / negative index")
    fmt = "<" + FORMATS[kind]
    width = struct.calcsize(fmt)
    if (i + 1) * width > len(buf):
        raise InvalidRepresentation("numeric address outside buffer")
    return struct.unpack_from(fmt, buf, i * width)[0]

def valid_at(a: Array, i: int) -> bool:
    if i < 0 or i >= a.length:
        raise InvalidRepresentation("logical index outside array")
    return bit(a.buffers[0], a.offset + i)

def value_at(a: Array, i: int):
    """Validity is checked before any invalid-slot payload is interpreted."""
    if not valid_at(a, i):
        return None
    p = a.offset + i
    t = a.logical_type
    if a.dictionary is not None:
        k = number(a.buffers[1], p, a.index_type)
        if not 0 <= k < a.dictionary.length:
            raise InvalidRepresentation("valid dictionary index out of range")
        return value_at(a.dictionary, k)
    if t.kind in {"int32", "int64", "float64"}:
        return number(a.buffers[1], p, t.kind)
    if t.kind == "bool":
        return bit(a.buffers[1], p)
    if t.kind in {"string", "binary"}:
        start = number(a.buffers[1], p, "int32")
        end = number(a.buffers[1], p + 1, "int32")
        raw = a.buffers[2]
        if raw is None or not 0 <= start <= end <= len(raw):
            raise InvalidRepresentation("binary interval out of bounds")
        v = raw[start:end]
        return v.decode("utf-8", errors="strict") if t.kind == "string" else v
    if t.kind == "list":
        start = number(a.buffers[1], p, "int32")
        end = number(a.buffers[1], p + 1, "int32")
        if not 0 <= start <= end <= a.children[0].length:
            raise InvalidRepresentation("list interval out of bounds")
        return [value_at(a.children[0], j) for j in range(start, end)]
    if t.kind == "struct":
        return {name: value_at(child, p) for (name, _), child in zip(t.fields, a.children)}
    raise InvalidRepresentation("unsupported layout")

def decode(a: Array) -> list:
    return [value_at(a, i) for i in range(a.length)]

def validate(a: Array, _depth: int = 0) -> None:
    if _depth > 32:
        raise InvalidRepresentation("descriptor nesting exceeds local safety bound")
    if not isinstance(a.length, int) or not isinstance(a.offset, int) or a.length < 0 or a.offset < 0:
        raise InvalidRepresentation("invalid length/offset")
    t, end = a.logical_type, a.offset + a.length
    expected_buffers = 2 if a.dictionary else (3 if t.kind in {"string", "binary"} else 2 if t.kind != "struct" else 1)
    if len(a.buffers) != expected_buffers:
        raise InvalidRepresentation("buffer count mismatch")
    if any(b is not None and not isinstance(b, bytes) for b in a.buffers):
        raise InvalidRepresentation("buffers must be immutable bytes")
    if a.buffers[0] is not None and len(a.buffers[0]) * 8 < end:
        raise InvalidRepresentation("validity buffer too short")
    if a.dictionary is not None:
        if a.index_type not in {"int8", "int16", "int32"} or a.children:
            raise InvalidRepresentation("invalid dictionary descriptor")
        if a.dictionary.logical_type != t or a.dictionary.dictionary is not None:
            raise InvalidRepresentation("dictionary value type mismatch / nested dictionary")
        if a.buffers[1] is None or len(a.buffers[1]) < end * struct.calcsize(FORMATS[a.index_type]):
            raise InvalidRepresentation("index buffer too short")
        validate(a.dictionary, _depth + 1)
        for i in range(a.length):
            if valid_at(a, i):
                k = number(a.buffers[1], a.offset + i, a.index_type)
                if not 0 <= k < a.dictionary.length:
                    raise InvalidRepresentation("dictionary index out of bounds")
    elif t.kind in {"int32", "int64", "float64", "bool"}:
        if a.children:
            raise InvalidRepresentation("primitive has children")
        needed = (end + 7) // 8 if t.kind == "bool" else end * struct.calcsize(FORMATS[t.kind])
        if a.buffers[1] is None or len(a.buffers[1]) < needed:
            raise InvalidRepresentation("value buffer too short")
    elif t.kind in {"string", "binary", "list"}:
        if a.buffers[1] is None or len(a.buffers[1]) < (end + 1) * 4:
            raise InvalidRepresentation("offset buffer too short")
        offsets = [number(a.buffers[1], j, "int32") for j in range(a.offset, end + 1)]
        if any(x < 0 for x in offsets) or any(x > y for x, y in zip(offsets, offsets[1:])):
            raise InvalidRepresentation("offsets must be nonnegative and monotone")
        if t.kind == "list":
            if len(a.children) != 1 or a.children[0].logical_type != t.fields[0][1]:
                raise InvalidRepresentation("list child mismatch")
            validate(a.children[0], _depth + 1)
            bound = a.children[0].length
        else:
            if a.children or a.buffers[2] is None:
                raise InvalidRepresentation("binary buffer missing / extraneous child")
            bound = len(a.buffers[2])
        if offsets[-1] > bound:
            raise InvalidRepresentation("terminal offset outside data/child")
    elif t.kind == "struct":
        if len(a.children) != len(t.fields):
            raise InvalidRepresentation("struct arity mismatch")
        for (_, ct), child in zip(t.fields, a.children):
            if child.logical_type != ct or child.length < end:
                raise InvalidRepresentation("struct child type/length mismatch")
            validate(child, _depth + 1)
    try:
        decode(a)  # validates reachable UTF-8 and recursively selected values only
    except (UnicodeError, IndexError, struct.error) as exc:
        raise InvalidRepresentation(str(exc)) from exc

@dataclass(frozen=True)
class PhysicalTable:
    fields: tuple[tuple[str, Type], ...]
    columns: tuple[tuple[Array, ...], ...]

    def validate(self):
        if len(self.fields) != len(self.columns) or not self.columns:
            raise InvalidRepresentation("table width mismatch")
        lengths = []
        for (_, typ), chunks in zip(self.fields, self.columns):
            for a in chunks:
                if a.logical_type != typ:
                    raise InvalidRepresentation("column type mismatch")
                validate(a)
            lengths.append(sum(a.length for a in chunks))
        if len(set(lengths)) != 1:
            raise InvalidRepresentation("column lengths differ")

    def rows(self):
        cols = [[v for chunk in chunks for v in decode(chunk)] for chunks in self.columns]
        return tuple(zip(*cols))

    def to_json(self):
        return {"fields": [[n,t.to_json()] for n,t in self.fields],
                "columns": [[a.to_json() for a in chunks] for chunks in self.columns]}

    @staticmethod
    def from_json(x):
        return PhysicalTable(tuple((n,Type.from_json(t)) for n,t in x["fields"]),
                             tuple(tuple(Array.from_json(a) for a in c) for c in x["columns"]))

    def byte_size(self):
        return sum(a.byte_size() for c in self.columns for a in c)

def certify(logical: LogicalTable, physical: PhysicalTable) -> dict:
    logical.validate()
    physical.validate()
    if logical.fields != physical.fields:
        raise InvalidRepresentation("certificate schema differs")
    expected = [tuple(canonical(v) for v in r) for r in logical.rows]
    actual = [tuple(canonical(v) for v in r) for r in physical.rows()]
    if expected != actual:
        raise InvalidRepresentation("representation does not preserve ordered typed rows")
    return {"status": "certified_model", "rows": len(actual), "bytes": physical.byte_size(),
            "basis": "independent buffer interpreter; not a PyArrow validation result"}
