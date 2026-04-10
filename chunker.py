def chunk_text(text: str, chunk_size: int = 160, overlap: int = 30) -> list[str]:
    words = text.split(" ")
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

