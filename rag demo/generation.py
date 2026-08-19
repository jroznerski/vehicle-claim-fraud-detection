"""
Ta część to właściwe "Augmented Generation" w RAG: bierzemy chunki
znalezione przez retriever i wstrzykujemy je do promptu jako kontekst,
instruując model żeby:
  1) odpowiadał WYŁĄCZNIE na podstawie podanego kontekstu (nie z własnej
     "pamięci" wyuczonej podczas treningu) -- to jest grounding,
  2) jasno powiedział "nie wiem", jeśli kontekst nie zawiera odpowiedzi,
     zamiast zmyślać (to główna dźwignia ograniczania halucynacji w RAG).

W produkcji ten prompt trafiłby do prawdziwego LLM (Claude, GPT, itd.)
przez API. Tu sandbox nie ma skonfigurowanego klucza API, więc funkcja
`generate_answer` ma dwa tryby:
  - jeśli jest dostępny klucz ANTHROPIC_API_KEY w środowisku, faktycznie
    wywołuje model,
  - w przeciwnym razie zwraca zbudowany prompt + informację, że to właśnie
    ten tekst zostałby wysłany do LLM.
"""

import os


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Fragment {i+1}]\n{c}" for i, c in enumerate(retrieved_chunks))

    prompt = f"""Odpowiedz na pytanie użytkownika WYŁĄCZNIE na podstawie poniższego kontekstu.
Jeśli kontekst nie zawiera odpowiedzi, napisz "Nie znalazłem odpowiedzi w dostępnych dokumentach" -- nie zgaduj i nie korzystaj z wiedzy spoza kontekstu.

Kontekst:
{context}

Pytanie: {query}

Odpowiedź:"""
    return prompt


def generate_answer(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[Brak klucza API w tym środowisku -- w produkcji ten prompt "
            "trafiłby do LLM przez API. Poniżej dokładnie ten tekst, który "
            "zostałby wysłany:]\n\n" + prompt
        )

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    fake_chunks = [
        "Polityka urlopowa: pracownik ma prawo do 26 dni urlopu wypoczynkowego rocznie.",
    ]
    prompt = build_prompt("ile dni urlopu mi przysługuje?", fake_chunks)
    print(generate_answer(prompt))
