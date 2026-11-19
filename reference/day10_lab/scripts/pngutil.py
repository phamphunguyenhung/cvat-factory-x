"""Read and write 8-bit RGB PNGs. COCO panoptic masks use this layout.

Segment id is R + G * 256 + B * 65536. No third-party packages.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    if abs(estimate - left) <= abs(estimate - up) and abs(estimate - left) <= abs(estimate - up_left):
        return left
    if abs(estimate - up) <= abs(estimate - up_left):
        return up
    return up_left


def read_rgb(path: Path) -> tuple[int, int, bytearray]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos = 8
    width = height = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit, color, comp, filt, inter = struct.unpack(">IIBBBBB", chunk)
            if (bit, color, comp, filt, inter) != (8, 2, 0, 0, 0):
                raise ValueError(f"{path} is not 8-bit RGB")
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    if width is None:
        raise ValueError(f"{path} has no IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    pixels = bytearray()
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filt = raw[cursor]
        row = bytearray(raw[cursor + 1 : cursor + 1 + stride])
        cursor += 1 + stride
        if filt == 1:
            for index in range(stride):
                left = row[index - 3] if index >= 3 else 0
                row[index] = (row[index] + left) & 255
        elif filt == 2:
            for index in range(stride):
                row[index] = (row[index] + previous[index]) & 255
        elif filt == 3:
            for index in range(stride):
                left = row[index - 3] if index >= 3 else 0
                row[index] = (row[index] + ((left + previous[index]) // 2)) & 255
        elif filt == 4:
            for index in range(stride):
                left = row[index - 3] if index >= 3 else 0
                row[index] = (row[index] + _paeth(left, previous[index], previous[index - 3] if index >= 3 else 0)) & 255
        elif filt != 0:
            raise ValueError(f"unsupported PNG filter {filt}")
        pixels += row
        previous = row
    return width, height, pixels


def write_rgb(path: Path, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height * 3:
        raise ValueError("pixel buffer has the wrong size")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += pixels[y * stride : (y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def segment_id(pixels: bytes, index: int) -> int:
    return pixels[index] + pixels[index + 1] * 256 + pixels[index + 2] * 65536


def segment_rgb(segment: int) -> tuple[int, int, int]:
    return segment & 255, (segment >> 8) & 255, (segment >> 16) & 255
