"""Minimal GGUF tensor-offset parser + in-place mmap patcher.

The Stage F+ rewrite path copies the source BF16 GGUF, then mutates a
small subset of tensors (Q/K/V/gate/up weights as BF16, attn_norm/ffn_norm
γ as F32) in place via mmap. This keeps non-modified bytes byte-identical
to the source while avoiding a full streaming GGUF rewrite.

GGUF format (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):

    magic     "GGUF" (4 bytes)
    version   u32
    tcount    u64  (tensor count)
    kvcount   u64  (KV metadata count)
    KV entries...   (key: string, type_tag: u32, value: <by type>)
    tensor info entries...
        name (string), n_dims (u32), dims (n_dims*u64),
        type (u32), offset (u64, RELATIVE to data start)
    [pad to alignment]
    tensor data

The header is the only thing we parse; tensor data byte-rewrite happens
through the returned (offset, n_bytes) tuples + the absolute data-start
offset.

ggml dim order: ne[0] = fastest-changing (= in_features for Linear
weights). So a `[out, in]` Linear surfaces as `dims=(in, out)`.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

# Bytes per element. Only the types we read/write in F+.
_TYPE_BYTES = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
}

# GGUF metadata-value type tags.
_GGUF_TYPE_TAG_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
    7: 1,                       # bool
    8: None, 9: None,           # string, array (variable)
    10: 8, 11: 8, 12: 8,
}


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]        # ggml order: dims[0] = fastest-changing
    ttype: int
    data_offset_rel: int         # offset relative to start of tensor data
    n_bytes: int                 # element count × bytes_per_elem


@dataclass
class GgufHeader:
    """Just enough to locate every tensor byte range and the default
    alignment for the tensor-data start."""
    path: Path
    version: int
    tensor_count: int
    kv_count: int
    alignment: int               # bytes; default 32 if KV doesn't specify
    data_start: int              # absolute byte offset of tensor data
    tensors: dict[str, TensorInfo]

    def abs_range(self, name: str) -> tuple[int, int]:
        ti = self.tensors[name]
        start = self.data_start + ti.data_offset_rel
        return (start, start + ti.n_bytes)


def parse_gguf_header(path: Path) -> GgufHeader:
    """Read a GGUF's header + tensor info table. Does not touch tensor
    data. Returns offsets/dims for every tensor in the file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF: {path}")
        version, tcount, kvcount = struct.unpack("<IQQ", f.read(20))

        alignment = 32  # GGUF default per spec
        # ----- KV metadata section -----
        def _read_string() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def _skip_value(tag: int) -> None:
            if tag == 8:        # string
                _read_string()
            elif tag == 9:      # array: <element_type:u32><n:u64><values...>
                etype, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    _skip_value(etype)
            else:
                sz = _GGUF_TYPE_TAG_SIZES.get(tag)
                if sz is None:
                    raise ValueError(f"unknown GGUF type tag {tag}")
                f.read(sz)

        def _read_value(tag: int):
            if tag == 4:        # u32
                return struct.unpack("<I", f.read(4))[0]
            if tag == 5:        # i32
                return struct.unpack("<i", f.read(4))[0]
            if tag == 8:        # string
                return _read_string()
            _skip_value(tag)
            return None

        for _ in range(kvcount):
            key = _read_string()
            (tag,) = struct.unpack("<I", f.read(4))
            val = _read_value(tag)
            if key == "general.alignment" and isinstance(val, int):
                alignment = int(val)

        # ----- Tensor info section -----
        tensors: dict[str, TensorInfo] = {}
        for _ in range(tcount):
            name = _read_string()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            ttype, rel_off = struct.unpack("<IQ", f.read(12))
            n_elems = 1
            for d in dims:
                n_elems *= d
            bpe = _TYPE_BYTES.get(ttype)
            if bpe is None:
                # Unknown / quantized type — we won't patch it; still record
                # so the parser doesn't fail on mixed GGUFs. n_bytes set to 0
                # signals "do not touch."
                n_bytes = 0
            else:
                n_bytes = n_elems * bpe
            tensors[name] = TensorInfo(
                name=name, dims=dims, ttype=ttype,
                data_offset_rel=rel_off, n_bytes=n_bytes)

        # ----- Pad to alignment -----
        header_end = f.tell()
        pad = (-header_end) % alignment
        data_start = header_end + pad

    return GgufHeader(
        path=path, version=version, tensor_count=tcount, kv_count=kvcount,
        alignment=alignment, data_start=data_start, tensors=tensors)


# ─────────────────────────────────────────────────────────────────────────────
# bf16 ↔ fp32 conversions (NumPy doesn't have a native bf16 dtype).
# fp32 -> bf16 uses IEEE round-to-nearest-even, matching llama.cpp's
# `ggml_fp32_to_bf16` convention. Truncation alone would bias error.
# ─────────────────────────────────────────────────────────────────────────────

def bf16_bytes_to_fp32(buf: bytes | memoryview) -> np.ndarray:
    """Bit-cast a BF16 byte buffer to an fp32 ndarray (zero-padded mantissa).

    NumPy ≥1.25 has a `bfloat16` extension dtype but it's not always
    available, so we go through uint16 → uint32<<16 → view as float32.
    """
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def fp32_to_bf16_bytes(x: np.ndarray) -> bytes:
    """Pack an fp32 ndarray into BF16 bytes with round-to-nearest-even.

    RNE: add `(((u32 >> 16) & 1) + 0x7fff)` before right-shifting. This is
    the standard "ties-to-even" bias derivation for halving the mantissa.
    NaN is canonicalized to 0x7fc0 (quiet bf16 NaN) so a truncation-induced
    flip to ±inf cannot happen.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    nan_mask = np.isnan(x)
    u32 = x.view(np.uint32).copy()
    rounding = ((u32 >> 16) & 1).astype(np.uint32) + np.uint32(0x7fff)
    bf16 = ((u32 + rounding) >> 16).astype(np.uint16)
    if nan_mask.any():
        bf16[nan_mask] = 0x7fc0
    return bf16.tobytes()


def numpy_shape(dims: tuple[int, ...]) -> tuple[int, ...]:
    """ggml stores dims in fastest-axis-first order (ne[0] = fastest). NumPy
    is row-major (last-axis-fastest). So the natural numpy view of a ggml
    tensor with `dims=(in, out)` is `(out, in)` — i.e. `reversed(dims)`.
    Apply this to read/write so 2-D Linear weights surface as the
    intuitive `[out, in]` shape.
    """
    return tuple(reversed(dims))


def read_tensor_fp32(hdr: GgufHeader, mm, name: str) -> np.ndarray:
    """Pull a tensor out of `mm` (mmap or file-like supporting [start:end])
    as fp32 ndarray. Shape is `reversed(ggml_dims)` — i.e. `[out, in]` for a
    2-D Linear weight, `[in]` for a 1-D RMSNorm γ.

    Returns a copy (independent of mm). Caller is expected to do math in
    fp32 and write back via `write_tensor_fp32`.
    """
    ti = hdr.tensors[name]
    start, end = hdr.abs_range(name)
    buf = mm[start:end]
    if ti.ttype == GGML_TYPE_F32:
        arr = np.frombuffer(buf, dtype=np.float32).copy()
    elif ti.ttype == GGML_TYPE_BF16:
        arr = bf16_bytes_to_fp32(buf)
    else:
        raise NotImplementedError(
            f"tensor {name!r} has ggml type {ti.ttype}; F+ only handles "
            f"F32 ({GGML_TYPE_F32}) and BF16 ({GGML_TYPE_BF16})")
    return arr.reshape(numpy_shape(ti.dims))


def write_tensor_fp32(hdr: GgufHeader, mm, name: str, arr: np.ndarray) -> None:
    """Write `arr` (fp32, shape == reversed(ggml_dims)) back to the same
    byte range, re-encoding to the tensor's stored dtype. Total byte count
    is preserved so this never grows or shrinks the GGUF.

    `mm` must be an mmap opened in write mode (or any bytearray-backed
    object supporting slice assignment). Caller is responsible for
    `mm.flush()` after a batch of writes.
    """
    ti = hdr.tensors[name]
    start, end = hdr.abs_range(name)
    expected = numpy_shape(ti.dims)
    if tuple(arr.shape) != expected:
        raise ValueError(
            f"shape mismatch on {name}: got {arr.shape}, expected {expected}")
    if ti.ttype == GGML_TYPE_F32:
        buf = arr.astype(np.float32, copy=False).tobytes()
    elif ti.ttype == GGML_TYPE_BF16:
        buf = fp32_to_bf16_bytes(arr)
    else:
        raise NotImplementedError(
            f"write of ggml type {ti.ttype} not implemented (tensor {name!r})")
    if len(buf) != end - start:
        raise RuntimeError(
            f"encoded size mismatch on {name}: {len(buf)} vs slot {end-start}")
    mm[start:end] = buf
