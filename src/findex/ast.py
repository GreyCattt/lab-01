from abc import ABC, abstractmethod
from dataclasses import dataclass
from findex.index import Index

class QueryNode(ABC):
    """Базовий клас для всіх вузлів синтаксичного дерева запиту."""
    
    @abstractmethod
    def evaluate(self, index: Index) -> set[int]:
        """Оцінює вузол відносно індексу і повертає множину doc_id."""
        pass
    def __and__(self, other: 'QueryNode') -> 'QueryNode':
        return And(self, other)

    def __or__(self, other: 'QueryNode') -> 'QueryNode':
        return Or(self, other)

    def __invert__(self) -> 'QueryNode':
        return Not(self)


@dataclass(frozen=True)
class Term(QueryNode):
    word: str

    def evaluate(self, index: Index) -> set[int]:
        """Повертає документи, де зустрічається термін."""
        return {p.doc_id for p in index[self.word]}


@dataclass(frozen=True)
class Phrase(QueryNode):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        """Фразовий пошук: терміни мають йти строго один за одним."""
        if not self.words:
            return set()
            
        postings_by_word = {word: {p.doc_id: p for p in index[word]} for word in self.words}
        # type: ignore - Pyright не може вивести тип після динамічного розпакування генератора в set.intersection
        common_docs: set[int] = set.intersection(*(set(pw.keys()) for pw in postings_by_word.values())) # type: ignore
        common_docs = set.intersection(*(set(pw.keys()) for pw in postings_by_word.values())) # type: ignore
        
        result: set[int] = set()
        for doc_id in common_docs:
            first_word_positions = postings_by_word[self.words[0]][doc_id].positions
            
            for start_pos in first_word_positions:
                match = True
                for i, word in enumerate(self.words[1:], 1):
                    if (start_pos + i) not in postings_by_word[word][doc_id].positions:
                        match = False
                        break
                        
                if match:
                    result.add(doc_id)
                    break 
                    
        return result


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(QueryNode):
    child: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        all_docs = set(range(index.num_docs))
        return all_docs - self.child.evaluate(index)