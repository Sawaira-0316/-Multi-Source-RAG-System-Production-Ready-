# data_ingestion/audio_loader.py

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from openai import OpenAI

client = OpenAI()


def transcribe_audio_file(path: str) -> List[Document]:
    """
    Transcribe a single audio/video file to text and wrap as a Document.
    Supported: .mp3, .wav, .m4a, .mp4, etc.
    """
    p = Path(path)
    if not p.exists():
        return []

    with p.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # or current transcription model
            file=f,
        )

    return [
        Document(
            page_content=transcript.text,
            metadata={"source": str(p), "type": "audio_transcript"},
        )
    ]

