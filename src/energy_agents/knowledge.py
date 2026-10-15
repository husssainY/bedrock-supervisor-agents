"""Knowledge base retrieval.

`LocalKnowledgeBase` indexes markdown files by `##` section with a small BM25
scorer, enough for local runs and tests with no services. `OpenSearchKnowledgeBase`
queries an OpenSearch index with the same interface for production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from how i in is it my of on or the to what when why with "
                  "you your do does did can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str
    score: float


class KnowledgeBase(Protocol):
    def search(self, query: str, k: int = 3) -> list[Passage]: ...


class LocalKnowledgeBase:
    def __init__(self, sections: list[tuple[str, str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._sections = sections
        self._docs = [Counter(tokenize(f"{title} {title} {text}")) for _, title, text in sections]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        df = Counter(term for d in self._docs for term in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = k1, b

    @classmethod
    def from_directory(cls, path: str | Path) -> LocalKnowledgeBase:
        sections: list[tuple[str, str, str]] = []
        for file in sorted(Path(path).glob("*.md")):
            for block in re.split(r"^## ", file.read_text(), flags=re.MULTILINE)[1:]:
                title, _, body = block.partition("\n")
                sections.append((file.name, title.strip(), body.strip()))
        return cls(sections)

    def search(self, query: str, k: int = 3) -> list[Passage]:
        terms = tokenize(query)
        scored = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg_len)
                    score += self._idf[term] * tf * (self._k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [
            Passage(self._sections[i][0], self._sections[i][1], self._sections[i][2], round(s, 3))
            for s, i in scored[:k]
        ]


class OpenSearchKnowledgeBase:
    def __init__(self, client: Any, index: str) -> None:
        """`client` is an `opensearchpy.OpenSearch` instance."""
        self._client = client
        self._index = index

    def search(self, query: str, k: int = 3) -> list[Passage]:
        response = self._client.search(index=self._index, body={
            "size": k,
            "query": {"multi_match": {"query": query, "fields": ["title^2", "text"]}},
        })
        return [
            Passage(hit["_source"].get("source", self._index), hit["_source"]["title"],
                    hit["_source"]["text"], float(hit["_score"]))
            for hit in response["hits"]["hits"]
        ]
