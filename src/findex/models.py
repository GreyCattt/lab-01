from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int

@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    length: int
    path: str