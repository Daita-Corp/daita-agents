"""Capture immutable output bytes only while the native parent proves suspension."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".parquet": "application/vnd.apache.parquet",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".sql": "application/sql",
}
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
INPUT_MEDIA_TYPES = {**MEDIA_TYPES, ".arrow": "application/vnd.apache.arrow.file"}


@dataclass(frozen=True, slots=True)
class OutputCandidate:
    name: str
    filename: str
    media_type: str
    content: bytes
    sha256: str


def capture(
    scratch: Path,
    descriptors: object,
    *,
    max_bytes: int = MAX_OUTPUT_BYTES,
    max_count: int = 4,
) -> tuple[OutputCandidate, ...]:
    if not isinstance(descriptors, list) or len(descriptors) > max_count:
        raise ValueError("Invalid output candidate manifest")
    candidates = []
    for item in descriptors:
        if not isinstance(item, dict) or set(item) != {"name", "path"}:
            raise ValueError("Invalid output candidate descriptor")
        name, filename = item["name"], item["path"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None
            or not isinstance(filename, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", filename) is None
            or Path(filename).suffix not in MEDIA_TYPES
        ):
            raise ValueError("Output names and formats must use the admitted contract")
        descriptor = os.open(
            scratch / filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > max_bytes
            ):
                raise ValueError("Output candidate is not one bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                content = source.read(max_bytes + 1)
            after = os.fstat(descriptor)
            identity = lambda facts: (
                facts.st_dev,
                facts.st_ino,
                facts.st_mode,
                facts.st_nlink,
                facts.st_size,
                facts.st_mtime_ns,
                facts.st_ctime_ns,
            )
            if len(content) != before.st_size or identity(after) != identity(before):
                raise ValueError("Output candidate changed during capture")
        finally:
            os.close(descriptor)
        candidates.append(
            OutputCandidate(
                name,
                filename,
                MEDIA_TYPES[Path(filename).suffix],
                content,
                "sha256:" + sha256(content).hexdigest(),
            )
        )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Output candidate names must be distinct")
    return tuple(candidates)


VALIDATION_CODE = {
    ".arrow": "import pyarrow.ipc as ipc\nwith ipc.open_file('candidate.bin') as reader:\n if reader.num_record_batches > 4096 or len(reader.schema) > 256: raise ValueError('table bound exceeded')\n rows=0; size=0\n for i in range(reader.num_record_batches):\n  batch=reader.get_batch(i); rows+=batch.num_rows; size+=batch.nbytes\n  if rows > 1000000 or size > 134217728: raise ValueError('decoded table bound exceeded')",
    ".json": "import json\njson.loads(open('candidate.bin', encoding='utf-8').read())",
    ".csv": "import csv\nwith open('candidate.bin', newline='', encoding='utf-8') as f:\n for index, row in enumerate(csv.reader(f)):\n  if index >= 1000000 or len(row) > 256: raise ValueError('table bound exceeded')",
    ".png": "from PIL import Image\nwith Image.open('candidate.bin') as image:\n if image.format != 'PNG' or image.width*image.height > 16777216: raise ValueError('image bound exceeded')\n image.verify()\nwith Image.open('candidate.bin') as image: image.load()",
    ".parquet": "import pyarrow.parquet as pq\np = pq.ParquetFile('candidate.bin')\nm=p.metadata\nif m.num_rows > 1000000 or m.num_columns > 256 or sum(m.row_group(i).total_byte_size for i in range(m.num_row_groups)) > 134217728: raise ValueError('decoded table bound exceeded')\nt=p.read()\nif t.nbytes > 134217728: raise ValueError('decoded table bound exceeded')",
    ".md": "open('candidate.bin', encoding='utf-8').read()",
    ".txt": "open('candidate.bin', encoding='utf-8').read()",
    ".py": "open('candidate.bin', encoding='utf-8').read()",
    ".sql": "open('candidate.bin', encoding='utf-8').read()",
}
