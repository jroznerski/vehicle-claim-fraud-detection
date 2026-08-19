"""
Pełny, minimalny pipeline RAG. Dwie fazy, o których musisz umieć mówić
osobno na rozmowie:

FAZA 1 -- INDEXING (offline, robisz raz albo przy aktualizacji dokumentów)
    dokumenty -> chunking -> wektoryzacja (fit_transform) -> "baza wektorowa"
    (tu: macierz TF-IDF trzymana w pamięci; w produkcji: FAISS/Chroma/pgvector)

FAZA 2 -- RETRIEVAL + GENERATION (online, przy każdym zapytaniu użytkownika)
    zapytanie -> wektoryzacja (transform, TYM SAMYM wektoryzatorem)
              -> cosine similarity vs baza -> top-k chunków
              -> wstrzyknięcie do promptu -> LLM -> odpowiedź
"""

from chunking import chunk_text
from retriever import TfidfRetriever
from generation import build_prompt, generate_answer


# --- Przykładowa "baza wiedzy" firmowej (symulacja) ---
DOCUMENTS = [
    """Polityka urlopowa. Każdy pracownik zatrudniony na pełny etat ma prawo
    do 26 dni urlopu wypoczynkowego rocznie, jeśli jego staż pracy (łącznie
    z poprzednimi pracodawcami) przekracza 10 lat. Dla stażu poniżej 10 lat
    przysługuje 20 dni. Urlop należy zgłosić w systeme HR minimum 3 dni
    robocze przed planowanym wyjazdem, chyba że jest to sytuacja losowa.""",

    """Polityka zwrotu kosztów podróży służbowych. Wszystkie wydatki
    związane z podróżą służbową (hotel, transport, wyżywienie) wymagają
    wcześniejszego zatwierdzenia przez bezpośredniego managera przed
    wyjazdem. Limit dzienny na wyżywienie wynosi 150 zł. Faktury należy
    dostarczyć do działu księgowości w ciągu 14 dni od powrotu.""",

    """Polityka pracy zdalnej. Pracownicy mogą pracować zdalnie do 3 dni
    w tygodniu po uzgodnieniu z bezpośrednim przełożonym. W dniach pracy
    zdalnej obowiązuje dostępność w standardowych godzinach pracy zespołu.
    Praca zdalna z zagranicy wymaga dodatkowej zgody działu HR i prawnego.""",
]


def build_index(documents: list[str], chunk_size: int = 60, overlap: int = 15) -> TfidfRetriever:
    """FAZA 1: indexing. Dzielimy każdy dokument na chunki, spłaszczamy
    do jednej listy i budujemy na niej retriever (fit TF-IDF)."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))

    print(f"[Indexing] {len(documents)} dokumentów -> {len(all_chunks)} chunków")
    return TfidfRetriever(all_chunks)


def answer_query(retriever: TfidfRetriever, query: str, k: int = 2) -> str:
    """FAZA 2: retrieval + generation dla jednego zapytania."""
    retrieved = retriever.retrieve(query, k=k)

    print(f"\n[Retrieval] Top-{k} dla: '{query}'")
    for chunk, score in retrieved:
        print(f"  [{score:.3f}] {chunk[:70]}...")

    retrieved_chunks = [chunk for chunk, _ in retrieved]
    prompt = build_prompt(query, retrieved_chunks)
    return generate_answer(prompt)


if __name__ == "__main__":
    retriever = build_index(DOCUMENTS)

    for question in [
        "Ile dni urlopu mi przysługuje po 12 latach pracy?",
        "Jaki jest dzienny limit na wyżywienie w delegacji?",
        "Czy mogę pracować zdalnie z Hiszpanii?",
    ]:
        answer = answer_query(retriever, question)
        print(f"\n{'='*70}\n{answer}\n{'='*70}\n")
