from dataclasses import dataclass, field
from functools import total_ordering

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: tuple[int, ...] = field(default_factory=tuple)

@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    length: int
    path: str
    
@total_ordering  
@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: int
    score: float
    
    def __eq__(self, other):
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score