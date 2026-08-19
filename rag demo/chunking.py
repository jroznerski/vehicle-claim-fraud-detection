"""
Chunking: dzielimy długi tekst na mniejsze, częściowo nachodzące na siebie
fragmenty (chunki). Robimy to, bo:
  1) LLM ma ograniczone okno kontekstu (i płacisz za każdy token),
  2) retrieval jest dokładniejszy na małych, tematycznie spójnych
     fragmentach niż na całych dokumentach (mniej "szumu" w jednym wektorze),
  3) zjawisko "lost in the middle" — modele gorzej wykorzystują informacje
     leżące w środku bardzo długiego kontekstu.

Overlap (zachodzenie chunków) zapobiega ucięciu zdania/myśli dokładnie na
granicy dwóch chunków — fragment na styku dwóch kawałków pojawia się w obu.
"""

def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    """Dzieli tekst na chunki po `chunk_size` słów, z `overlap` słowami
    powtórzonymi na początku kolejnego chunka.

    Przykład: chunk_size=50, overlap=10 -> chunk 2 zaczyna się 10 słów
    przed końcem chunka 1.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap musi być mniejszy niż chunk_size, inaczej pętla się nie skończy")

    words = text.split()
    chunks = []
    step = chunk_size - overlap  # o ile "przesuwamy się" do przodu przy każdym kroku

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    sample = " ".join([f"słowo{i}" for i in range(1, 121)])  # 120 "słów"
    result = chunk_text(sample, chunk_size=50, overlap=10)
    print(f"Liczba chunków: {len(result)}")
    for i, c in enumerate(result):
        print(f"\nChunk {i}: {c[:60]}...")
