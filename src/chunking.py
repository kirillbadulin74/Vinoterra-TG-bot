from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    source_file: str
    section_index: int
    heading_path: tuple[str, ...]
    heading_level: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_file: str
    section_index: int
    chunk_index: int
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    content: str

    @property
    def section_path(self) -> str:
        return " > ".join(self.heading_path)

    def metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["section_path"] = self.section_path
        data.pop("content")
        data.pop("text")
        return data


def iter_markdown_files(base_dir: str | Path, pattern: str = "*.md") -> list[Path]:
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {base_path}")
    return sorted(path for path in base_path.glob(pattern) if path.is_file())


def parse_markdown_sections(path: str | Path) -> list[MarkdownSection]:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()
    current_level = 0
    current_start_line = 1
    section_index = 0

    def flush(end_line: int) -> None:
        nonlocal section_index, current_lines, current_path, current_level, current_start_line
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        sections.append(
            MarkdownSection(
                source_file=source_path.name,
                section_index=section_index,
                heading_path=current_path or (source_path.stem,),
                heading_level=current_level,
                start_line=current_start_line,
                end_line=end_line,
                text=section_text,
            )
        )
        section_index += 1

    for line_number, line in enumerate(lines, start=1):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = _clean_heading_title(heading_match.group(2))
            heading_stack = [(lvl, txt) for lvl, txt in heading_stack if lvl < level]
            heading_stack.append((level, title))
            current_path = tuple(title for _, title in heading_stack)
            current_level = level
            current_lines = []
            current_start_line = line_number + 1
            continue
        current_lines.append(line)

    flush(len(lines))
    return sections


def load_knowledge_chunks(
    base_dir: str | Path,
    *,
    chunk_size: int = 350,
    chunk_overlap: int = 0,
    min_chunk_chars: int = 40,
    pattern: str = "*.md",
) -> list[KnowledgeChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[KnowledgeChunk] = []
    for markdown_path in iter_markdown_files(base_dir, pattern):
        sections = parse_markdown_sections(markdown_path)
        for section in sections:
            section_chunks = split_text(section.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk_index, chunk_text in enumerate(section_chunks):
                if len(chunk_text.strip()) < min_chunk_chars:
                    continue
                chunk_id = f"{section.source_file}::s{section.section_index:04d}::c{chunk_index:02d}"
                content = build_chunk_content(section, chunk_text)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_file=section.source_file,
                        section_index=section.section_index,
                        chunk_index=chunk_index,
                        heading_path=section.heading_path,
                        start_line=section.start_line,
                        end_line=section.end_line,
                        text=chunk_text,
                        content=content,
                    )
                )
    return chunks


def split_text(text: str, *, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    units: list[str] = []
    for block in blocks:
        units.extend(_split_oversized_block(block, chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current.strip())
        current = unit

    if current.strip():
        chunks.append(current.strip())

    if chunk_overlap:
        chunks = _add_overlap(chunks, chunk_overlap)
    return chunks


def build_chunk_content(section: MarkdownSection, chunk_text: str) -> str:
    section_path = " > ".join(section.heading_path)
    return (
        f"Файл-источник: {section.source_file}\n"
        f"Раздел: {section_path}\n\n"
        f"{chunk_text.strip()}"
    )


def chunks_to_jsonl(chunks: Iterable[KnowledgeChunk], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "text": chunk.text,
                "metadata": chunk.metadata(),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean_heading_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("**", "").strip())


def _split_oversized_block(block: str, chunk_size: int) -> list[str]:
    if len(block) <= chunk_size:
        return [block]

    if "\n" in block:
        return _pack_units(block.splitlines(), chunk_size)

    sentences = re.split(r"(?<=[.!?])\s+", block)
    if len(sentences) > 1:
        return _pack_units(sentences, chunk_size)

    return _hard_wrap(block, chunk_size)


def _pack_units(units: Iterable[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_wrap(unit, chunk_size))
            continue
        separator = "\n" if current else ""
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            chunks.append(current.strip())
            current = unit
    if current:
        chunks.append(current.strip())
    return chunks


def _hard_wrap(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(word), chunk_size):
                chunks.append(word[start : start + chunk_size])
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            chunks.append(current.strip())
            current = word
    if current:
        chunks.append(current.strip())
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        tail = _overlap_tail(previous, overlap)
        overlapped.append(f"{tail}\n\n{current}" if tail else current)
    return overlapped


def _overlap_tail(text: str, overlap: int) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= overlap:
        return compact
    tail = compact[-overlap:]
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]
    return tail.strip()

