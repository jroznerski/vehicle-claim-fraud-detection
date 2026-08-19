"""
Retriever: zamienia chunki na wektory TF-IDF i wyszukuje te najbardziej
podobne do zapytania na podstawie cosine similarity.

TF-IDF (Term Frequency - Inverse Document Frequency):
  - Term Frequency: jak często słowo występuje w danym dokumencie
  - Inverse Document Frequency: karze słowa występujące w wielu dokumentach
    (bo są mało informacyjne — np. spójniki)
  - Wynik: sparse wektor (długość = rozmiar słownika), gdzie ważne, rzadkie
    słowa dostają wysoką wagę.

To KLASYCZNA (nie neuronowa) metoda reprezentacji tekstu. W produkcyjnym
RAG zwykle używa się dense embeddingów (np. sentence-transformers, OpenAI
text-embedding-3), bo łapią znaczenie, a nie tylko dopasowanie słów.
TF-IDF jest tu świadomym uproszczeniem: w pełni offline, deterministyczne,
zero zależności od zewnętrznego API czy pobierania modelu — ale nie złapie
synonimów ani parafraz.

Cosine similarity: mierzy kąt między dwoma wektorami (a nie ich długość).
Dwa dokumenty o różnej długości, ale podobnej "treści proporcjonalnej",
wyjdą jako podobne.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        # KLUCZOWE: fit_transform tylko raz, na chunkach (indexing offline).
        # Zapytanie później tylko `.transform()` -- musi użyć TEGO SAMEGO
        # wektoryzatora (tego samego "słownika"), inaczej wymiary wektorów
        # się nie zgadzają i porównanie nie ma sensu.
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, k: int = 2) -> list[tuple[str, float]]:
        """Zwraca top-k chunków najbardziej podobnych do zapytania,
        posortowane malejąco po score."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_vectors)[0]

        # Indeksy posortowane malejąco po score
        top_k_idx = scores.argsort()[::-1][:k]

        return [(self.chunks[i], float(scores[i])) for i in top_k_idx]


if __name__ == "__main__":
    docs = [
        "Polityka urlopowa: pracownik ma prawo do 26 dni urlopu wypoczynkowego rocznie.",
        "Zwrot kosztów podróży służbowych wymaga zatwierdzenia przez managera przed wyjazdem.",
        "Praca zdalna jest dozwolona do 3 dni w tygodniu, po ustaleniu z zespołem.",
        "Benefity pracownicze obejmują prywatną opiekę medyczną i kartę sportową.",
    ]

    retriever = TfidfRetriever(docs)
    results = retriever.retrieve("ile dni urlopu mi przysługuje?", k=2)

    for chunk, score in results:
        print(f"[{score:.3f}] {chunk}")
