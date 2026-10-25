"""The five retrieval strategies, each one thing different from the last.

Every strategy has the same signature:

    search(question, k) -> [(source_file, passage_text), ...]   best first

They are written to be read top to bottom. baseline is four lines; each one
after it adds exactly one idea.
"""
import csv
import functools
import os
from typing import Literal, Optional

import cohere
from pydantic import BaseModel, Field

from . import config
from .util import retry


# ---------------------------------------------------------------- the stores

@functools.lru_cache(maxsize=1)
def vector_store():
    from langchain_pinecone import PineconeVectorStore

    emb = config.embeddings()
    dim = len(emb.embed_query("probe"))
    return PineconeVectorStore(index_name=config.index_name(dim), embedding=emb)


@functools.lru_cache(maxsize=1)
def local_chunks():
    """The same chunks that went into Pinecone, kept in memory as text.

    BM25 counts words, so it needs the words. There is no way to ask a vector
    database for them, which is the first practical thing you learn about
    hybrid search: you end up maintaining two indexes.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    with open(config.MANIFEST) as f:
        names = sorted(row["file"] for row in csv.DictReader(f))

    chunks = []
    for name in names:
        page = TextLoader(os.path.join(config.SECTIONS, name)).load()[0]
        for piece in splitter.split_documents([page]):
            chunks.append((name, " ".join(piece.page_content.split())))
    return chunks


@functools.lru_cache(maxsize=1)
def bm25():
    from rank_bm25 import BM25Okapi

    return BM25Okapi([text.lower().split() for _, text in local_chunks()])


# ------------------------------------------------------- reading the question
#
# TODO 2.1 (2 pts) — read the company and year out of the question.

class Filter(BaseModel):
    """Which filings the question is about, and what to look for in them."""
    ticker: Optional[Literal["NVDA", "AMD", "MU"]] = Field(
        default=None, description="The company the question refers to. it could be either Micron,Nvidia or AMD. If the question states  either 0 names/tickers or multiple names then leave it blank. Output MU,NVDA,or AMD the ticker symbols and not the name")
    fiscal_year: Optional[int] = Field(
        default=None, description="What year is the question reffering to. If multiple years mentioned in the question leave blank")
    search: str = Field(description="what is the topic of the question without company name or year")


def extract_filter(question):
    model = config.chat().with_structured_output(Filter)
    prompt = "rewrite the question but without the ticker or year if applicable and fill in the ticker and fiscal year vairable with those values. If the question states  either 0 names/tickers or multiple names then leave it blank. If there are 0 or multiple fiscal years then leave it blank as well  " + question
    return retry(model.invoke, prompt)


def where_of(f):
    """Turn a Filter into a Pinecone filter dict.

    Pinecone wants operators, so ticker NVDA becomes {"ticker": {"$eq": "NVDA"}}.
    Watch the empty case: {} is not the same as no filter, and passing {} will
    not do what you want. Return None when there is nothing to filter on.
    """
    where = {}
    if f.ticker:
        where["ticker"] = {"$eq": f.ticker}
    if f.fiscal_year:
        where["fiscal_year"] = {"$eq": f.fiscal_year}
    if where:
        return where
    return None


# --------------------------------------------------------- many queries, fused
#
# TODO 2.2 (2 pts) — multi-query and reciprocal rank fusion.

class Queries(BaseModel):
    """Several ways of asking the same thing."""
    queries: list[str] = Field(description="Different search phrases that all look for the same information, each worded differently")


def write_queries(text, n=4):
    model = config.chat().with_structured_output(Queries)
    prompt = f"Write {n} different search queries that look for the same information as this text, using different wording: {text}"
    out = retry(model.invoke, prompt)
    return out.queries[:n] or [text]


def rrf(rankings, k=config.RRF_K):
    """Fuse ranked lists. Takes a list of lists, returns one list, best first.

    Every item must be hashable, which is why the strategies pass
    (source, text) tuples around rather than Document objects.
    """
    scores = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + position + 1)
    return sorted(scores, key=scores.get, reverse=True)


# --------------------------------------------------------------- the retrieval
# dense() is given. It is the one you already know.

def dense(query, k, where=None):
    hits = vector_store().similarity_search(query, k=k, filter=where)
    return [(h.metadata["source"], h.page_content) for h in hits]


def lexical(query, k):
    """BM25. Scores every chunk once, then takes the top k."""
    scores = bm25().get_scores(query.lower().split())
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [local_chunks()[i] for i in best]


# ---------------------------------------------------------------- reranking
#
# TODO 2.3 (2 pts) — reorder a pool with a cross-encoder.

def rerank_cohere(question, passages, top_n):
    client = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    response = client.rerank(
        model="rerank-v3.5",
        query=question,
        documents=[text for _, text in passages],
        top_n=top_n,
    )
    return [passages[r.index] for r in response.results]


def rerank(question, passages, top_n=config.K):
    """Cohere reranks the pool. Needs COHERE_API_KEY in .env."""
    return rerank_cohere(question, passages, top_n)


# -------------------------------------------------------------- the strategies
#
# All five return the same shape:  [(source_file, passage_text), ...]

def baseline(question, k=config.K):
    """Given. Embed the question as typed, one search, no filter."""
    return dense(question, k)


def with_filter(question, k=config.K):
    f = extract_filter(question)
    return dense(f.search, k, where=where_of(f))


def filter_rrf(question, k=config.K):
    f = extract_filter(question)
    where = where_of(f)
    lists = []
    for q in write_queries(f.search):
        lists.append(dense(q, k, where=where))
    return rrf(lists)[:k]


def filter_rrf_rerank(question, k=config.K):
    f = extract_filter(question)
    where = where_of(f)
    lists = []
    for q in write_queries(f.search):
        lists.append(dense(q, k, where=where))
    pool = rrf(lists)[:config.POOL]
    return rerank(question, pool, top_n=k)


def hybrid_rrf(question, k=config.K):
    """Dense and BM25, fused by rank. Catches the exact-term questions that
    embeddings paraphrase away."""
    f = extract_filter(question)
    dense_hits = dense(f.search, k, where=where_of(f))
    lexical_hits = lexical(f.search, k)
    return rrf([dense_hits, lexical_hits])[:k]


STRATEGIES = {
    "baseline": baseline,
    "filter": with_filter,
    "filter+rrf": filter_rrf,
    "filter+rrf+rerank": filter_rrf_rerank,
    "hybrid+rrf": hybrid_rrf,
}