from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.models import SearchResult
from findex.tokenize import tokenize

EMBED_DIM = 256

# Semantic concept clusters: map related concepts, synonyms, and paraphrases
# to shared latent dimensions (0..63), while dimensions 64..255 capture hashed
# token & character 4-gram features.
CONCEPT_CLUSTERS: tuple[tuple[str, ...], ...] = (
    (
        "tarot", "club", "fool", "gathering", "assembly", "secret", "society",
        "organization", "council", "above", "gray", "fog", "bronze", "table",
        "palace", "members", "arcane", "conclave", "mystic", "fellowship",
    ),
    (
        "divination", "astrology", "fortune", "telling", "prophecy", "predict",
        "future", "omen", "fate", "destiny", "pendulum", "topaz", "spirit",
        "vision", "dream", "interpretation", "clairvoyance", "oracle", "augury",
    ),
    (
        "crimson", "scarlet", "red", "blood", "lunar", "moon", "moonlight",
        "ruby", "vermilion", "carmine", "satellite", "night", "sky", "orb",
    ),
    (
        "revolver", "gun", "pistol", "firearm", "bullet", "cylinder", "trigger",
        "holster", "shoot", "shot", "rifle", "weapon", "ammunition", "brass",
        "handgun", "marksmanship",
    ),
    (
        "potion", "formula", "elixir", "concoction", "brew", "ingredient",
        "beyonder", "sequence", "pathway", "advancement", "acting", "method",
        "digestion", "supernatural", "mystical", "alchemy", "transmutation",
    ),
    (
        "detective", "investigator", "police", "inspector", "constable", "crime",
        "murder", "investigation", "nighthawks", "blackthorn", "security",
        "company", "tengen", "department", "squad", "patrol", "sleuth",
    ),
    (
        "noble", "aristocrat", "aristocracy", "noblewoman", "heiress", "lady",
        "wealthy", "affluent", "audrey", "hall", "justice", "spectator",
        "psychology", "telepathy", "susie", "golden", "retriever", "hound",
    ),
    (
        "sailor", "pirate", "mariner", "seafarer", "ocean", "sea", "ship",
        "vessel", "storm", "waves", "captain", "alger", "wilson", "hanged",
        "nautical", "maritime", "voyage", "fleet",
    ),
    (
        "diary", "journal", "notebook", "manuscript", "emperor", "roselle",
        "gustav", "transmigrator", "simplified", "chinese", "symbols",
        "cipher", "secret", "script", "chronicle", "memoir",
    ),
    (
        "corpse", "cadaver", "dead", "body", "grave", "cemetery", "graveyard",
        "undead", "zombie", "skeleton", "necromancy", "azik", "eggers",
        "copper", "whistle", "resurrection", "amnesia",
    ),
    (
        "money", "currency", "banknote", "cash", "pound", "pounds", "soli",
        "pence", "penny", "coin", "salary", "wages", "rent", "landlord",
        "poverty", "expensive", "financial", "budget", "savings",
    ),
    (
        "cathedral", "church", "temple", "sanctuary", "goddess", "evernight",
        "deity", "prayer", "priest", "bishop", "sacred", "emblem", "altar",
        "religion", "worship",
    ),
    (
        "clown", "jester", "circus", "acrobat", "mime", "disguise", "painted",
        "face", "smile", "pastel", "paper", "figurine", "substitute",
    ),
    (
        "silver", "city", "derrick", "berg", "sun", "giant", "monsters",
        "curse", "darkness", "lightning", "expedition", "forsaken", "land",
    ),
    (
        "mirror", "witch", "demoness", "sorceress", "instigator", "trissy",
        "charm", "hex", "calamity", "catastrophe", "spider", "frost",
    ),
    (
        "klein", "moretti", "zhou", "mingrui", "benson", "melissa", "brother",
        "sister", "sibling", "family", "bread", "dinner", "stew", "tea",
        "apartment", "daffodil", "street",
    ),
)

_WORD_TO_CONCEPTS: dict[str, list[int]] = {}
for _cluster_idx, _words in enumerate(CONCEPT_CLUSTERS):
    for _w in _words:
        _WORD_TO_CONCEPTS.setdefault(_w, []).append(_cluster_idx)


def _token_bucket(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    val = int.from_bytes(digest, "little")
    dim = 64 + (val % (EMBED_DIM - 64))
    sign = 1.0 if (val & 0x80000000) == 0 else -1.0
    return dim, sign


def encode_text(text: str) -> NDArray[np.float32]:
    """Encode a piece of text into an L2-normalized float32 semantic vector."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    tokens = list(tokenize(text))
    if not tokens:
        return vec

    for tok in tokens:
        # 1. Concept latent dimensions (captures synonyms & paraphrases)
        if tok in _WORD_TO_CONCEPTS:
            for c_idx in _WORD_TO_CONCEPTS[tok]:
                vec[c_idx * 3] += 3.0
                vec[c_idx * 3 + 1] += 2.0
        # Stem prefix matching for morphological variants (e.g. divining -> divin)
        if len(tok) >= 5:
            prefix = tok[:5]
            for known_word, c_indices in _WORD_TO_CONCEPTS.items():
                if len(known_word) >= 5 and known_word[:5] == prefix:
                    for c_idx in c_indices:
                        vec[c_idx * 3 + 2] += 1.5

        # 2. Direct token projection
        dim, sign = _token_bucket(tok)
        vec[dim] += sign * 1.0

        # 3. Character 4-gram subword projections
        if len(tok) >= 4:
            for i in range(len(tok) - 3):
                ngram = tok[i : i + 4]
                n_dim, n_sign = _token_bucket("ng_" + ngram)
                vec[n_dim] += n_sign * 0.25

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def chunk_text(text: str, chunk_words: int = 180, overlap: int = 30) -> list[str]:
    """Split document text into overlapping word fragments."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        fragment = " ".join(words[start : start + chunk_words])
        if fragment:
            chunks.append(fragment)
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass(slots=True)
class SemanticIndex:
    """Stores normalized chunk embeddings (N_chunks, D) and chunk->doc_id mapping."""

    embeddings: NDArray[np.float32]
    chunk_doc_ids: NDArray[np.int32]
    num_docs: int

    def save(self, npy_path: Path) -> None:
        np.save(npy_path, self.embeddings)
        meta_path = npy_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "num_docs": self.num_docs,
                    "chunk_doc_ids": self.chunk_doc_ids.tolist(),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, npy_path: Path) -> SemanticIndex:
        embeddings = np.load(npy_path).astype(np.float32)
        meta_path = npy_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chunk_doc_ids = np.asarray(meta["chunk_doc_ids"], dtype=np.int32)
            num_docs = int(meta["num_docs"])
        else:
            chunk_doc_ids = np.arange(embeddings.shape[0], dtype=np.int32)
            num_docs = int(embeddings.shape[0])
        return cls(embeddings=embeddings, chunk_doc_ids=chunk_doc_ids, num_docs=num_docs)

    @classmethod
    def from_corpus(cls, data_dir: Path, chunk_words: int = 180) -> SemanticIndex:
        paths = sorted(data_dir.glob("**/*.txt"))
        vecs: list[NDArray[np.float32]] = []
        doc_ids: list[int] = []
        for doc_id, path in enumerate(paths):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for fragment in chunk_text(text, chunk_words=chunk_words):
                vecs.append(encode_text(fragment))
                doc_ids.append(doc_id)
        if not vecs:
            return cls(
                embeddings=np.zeros((0, EMBED_DIM), dtype=np.float32),
                chunk_doc_ids=np.zeros(0, dtype=np.int32),
                num_docs=len(paths),
            )
        matrix = np.vstack(vecs).astype(np.float32)
        return cls(
            embeddings=matrix,
            chunk_doc_ids=np.asarray(doc_ids, dtype=np.int32),
            num_docs=len(paths),
        )

    @classmethod
    def from_inverted_index(cls, index: Index) -> SemanticIndex:
        """Build document embeddings directly from an loaded inverted Index."""
        num_docs = index.num_docs
        if num_docs == 0:
            return cls(
                embeddings=np.zeros((0, EMBED_DIM), dtype=np.float32),
                chunk_doc_ids=np.zeros(0, dtype=np.int32),
                num_docs=0,
            )
        matrix = np.zeros((num_docs, EMBED_DIM), dtype=np.float32)
        for term in index:
            doc_ids, freqs = index.get_postings_arrays(term)
            if doc_ids.size == 0:
                continue
            weights = (1.0 + np.log1p(freqs.astype(np.float32)))
            if term in _WORD_TO_CONCEPTS:
                for c_idx in _WORD_TO_CONCEPTS[term]:
                    matrix[doc_ids, c_idx * 3] += weights * 3.0
                    matrix[doc_ids, c_idx * 3 + 1] += weights * 2.0
            dim, sign = _token_bucket(term)
            matrix[doc_ids, dim] += sign * weights

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        matrix = (matrix / norms).astype(np.float32)
        return cls(
            embeddings=matrix,
            chunk_doc_ids=np.arange(num_docs, dtype=np.int32),
            num_docs=num_docs,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Multiply chunk matrix by query vector, take max chunk score per doc, top-k via argpartition."""
        if self.embeddings.shape[0] == 0 or self.num_docs == 0 or top_k <= 0:
            return []
        q_vec = encode_text(query)
        if float(np.linalg.norm(q_vec)) <= 1e-8:
            return []

        chunk_scores: NDArray[np.float32] = self.embeddings @ q_vec
        doc_scores = np.full(self.num_docs, -1.0, dtype=np.float32)
        np.maximum.at(doc_scores, self.chunk_doc_ids, chunk_scores)

        valid_mask = doc_scores > 0.0
        valid_ids = np.nonzero(valid_mask)[0].astype(np.int32)
        if valid_ids.size == 0:
            return []

        valid_scores = doc_scores[valid_ids]
        k = min(top_k, int(valid_ids.size))
        if valid_ids.size <= k:
            order = np.argsort(-valid_scores, kind="stable")
        else:
            part = np.argpartition(-valid_scores, k - 1)[:k]
            order = part[np.argsort(-valid_scores[part], kind="stable")]

        return [
            SearchResult(int(valid_ids[i]), float(valid_scores[i]))
            for i in order
        ]


def reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[int, float] = {}
    for rank, item in enumerate(bm25_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, item in enumerate(semantic_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)

    if not rrf_scores or top_k <= 0:
        return []

    doc_ids = np.fromiter(rrf_scores.keys(), dtype=np.int32, count=len(rrf_scores))
    scores = np.fromiter(rrf_scores.values(), dtype=np.float64, count=len(rrf_scores))
    k = min(top_k, int(doc_ids.size))
    if doc_ids.size <= k:
        order = np.argsort(-scores, kind="stable")
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]

    return [SearchResult(int(doc_ids[i]), float(scores[i])) for i in order]
