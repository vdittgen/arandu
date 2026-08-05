"""Local voice transcription via Qwen3-ASR (MLX) with faster-whisper fallback.

Primary backend: ``mlx-qwen3-asr`` — runs Qwen3-ASR-0.6B on Apple Silicon
via MLX with 8-bit quantization (~0.7 GB RAM). Provides 52-language support
and significantly better accuracy than Whisper.

Fallback backend: ``faster-whisper`` (CTranslate2) — used when MLX is not
available (e.g., non-macOS platforms).

The module is designed for graceful degradation: if neither backend is
installed, ``is_available()`` returns False and all transcription calls
raise without crashing the app.

sensitivity_tier: 3 (processes voice/audio data)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Backend constants
BACKEND_QWEN = "qwen"
BACKEND_WHISPER = "whisper"

# Qwen model mapping — on 8GB M1, only 0.6B is practical
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


@dataclass(frozen=True)
class TranscriptionSegment:
    """A timestamped segment of transcribed text.

    sensitivity_tier: 3
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Full transcription result with metadata.

    sensitivity_tier: 3
    """

    text: str
    language: str
    duration: float
    segments: list[TranscriptionSegment]


def is_qwen_asr_available() -> bool:
    """Check if mlx-qwen3-asr is installed.

    sensitivity_tier: 1
    """
    try:
        import mlx_qwen3_asr  # noqa: F401

        return True
    except ImportError:
        return False


def is_whisper_available() -> bool:
    """Check if faster-whisper is installed (fallback backend).

    sensitivity_tier: 1
    """
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def is_available() -> bool:
    """Check if any ASR backend is installed.

    Returns True if either mlx-qwen3-asr (preferred) or
    faster-whisper (fallback) is available.

    sensitivity_tier: 1
    """
    return is_qwen_asr_available() or is_whisper_available()


class VoiceTranscriber:
    """Local speech-to-text transcription with dual-backend support.

    Prefers Qwen3-ASR via MLX (Apple Silicon optimized) and falls back
    to faster-whisper (CTranslate2) when MLX is unavailable.

    The model is loaded lazily on first ``transcribe()`` call to avoid
    unnecessary memory usage. Once loaded, the model stays in memory
    for subsequent calls.

    Args:
        model_size: Model variant (``"tiny"``, ``"base"``, ``"small"``).
            For Qwen backend, all sizes map to Qwen3-ASR-0.6B.
            For Whisper fallback, maps directly to Whisper model sizes.
        compute_type: CTranslate2 compute type for Whisper fallback.
            Ignored when using Qwen backend.

    sensitivity_tier: 3
    """

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._model: Any | None = None
        self._session: Any | None = None
        self._backend: str | None = None

    def _ensure_model(self) -> Any:
        """Lazily load the ASR model on first use.

        Tries Qwen3-ASR (MLX) first, falls back to faster-whisper.

        sensitivity_tier: 1
        """
        if self._backend is not None:
            return self._model or self._session

        # Try Qwen3-ASR via MLX first (preferred)
        if is_qwen_asr_available():
            from mlx_qwen3_asr import Session

            logger.info(
                "Loading Qwen3-ASR model: %s (MLX backend)",
                QWEN_MODEL_ID,
            )
            self._session = Session(model=QWEN_MODEL_ID)
            self._backend = BACKEND_QWEN
            return self._session

        # Fall back to faster-whisper
        if is_whisper_available():
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model: size=%s compute=%s (fallback backend)",
                self._model_size,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type=self._compute_type,
            )
            self._backend = BACKEND_WHISPER
            return self._model

        raise RuntimeError(
            "No ASR backend available. "
            "Install mlx-qwen3-asr (recommended) or faster-whisper: "
            "pip install 'arandu[voice]'"
        )

    def _transcribe_qwen(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using Qwen3-ASR via MLX.

        sensitivity_tier: 3
        """
        if language:
            try:
                results = self._session.transcribe(audio_path, language=language)
            except TypeError:
                # Older mlx_qwen3_asr builds don't accept the kwarg.
                results = self._session.transcribe(audio_path)
        else:
            results = self._session.transcribe(audio_path)
        result = results if not isinstance(results, list) else results[0]

        text = result.text.strip() if hasattr(result, "text") else ""
        language = result.language if hasattr(result, "language") else "unknown"

        # Qwen3-ASR returns segments if available
        segments: list[TranscriptionSegment] = []
        duration = 0.0
        if hasattr(result, "segments") and result.segments:
            for seg in result.segments:
                segments.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                    )
                )
            duration = segments[-1].end if segments else 0.0
        elif hasattr(result, "duration"):
            duration = result.duration

        # If no segments returned, create a single segment for the full text
        if not segments and text:
            segments = [
                TranscriptionSegment(start=0.0, end=duration, text=text)
            ]

        return TranscriptionResult(
            text=text,
            language=language,
            duration=duration,
            segments=segments,
        )

    def _transcribe_whisper(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper (fallback).

        sensitivity_tier: 3
        """
        kwargs: dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        segments_iter, info = self._model.transcribe(audio_path, **kwargs)

        segments: list[TranscriptionSegment] = []
        text_parts: list[str] = []
        for seg in segments_iter:
            segments.append(
                TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
            )
            text_parts.append(seg.text.strip())

        full_text = " ".join(text_parts)
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration=info.duration,
            segments=segments,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (WAV, OGG, MP3, WebM, etc.).
            language: Optional BCP-47 / ISO-639-1 language hint (e.g. ``"en"``,
                ``"es"``). When ``None``, the backend auto-detects.

        Returns:
            TranscriptionResult with full text, detected language,
            duration, and timestamped segments.

        Raises:
            RuntimeError: If no ASR backend is installed.
            FileNotFoundError: If audio_path does not exist.

        sensitivity_tier: 3
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_model()

        if self._backend == BACKEND_QWEN:
            return self._transcribe_qwen(str(path), language=language)
        return self._transcribe_whisper(str(path), language=language)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio from raw bytes.

        Writes bytes to a temporary file, transcribes, then cleans up.

        Args:
            audio_bytes: Raw audio data (any format supported by ffmpeg).
            language: Optional language hint forwarded to ``transcribe``.

        Returns:
            TranscriptionResult with full text and metadata.

        sensitivity_tier: 3
        """
        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=True,
        ) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe(tmp.name, language=language)
