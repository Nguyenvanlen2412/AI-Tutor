"""
main.py – interactive demo / programmatic entry point for the AI Tutor.

CLI usage:
    # Text mode (default)
    python main.py

    # Voice mode  (pipe a WAV file)
    python main.py --voice path/to/audio.wav --output voice

    # Headless / one-shot
    python main.py --query "Hãy giải thích định lý Pythagoras" --user-id student_42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

from state import State
from graph import tutor_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_text_turn(
    query: str,
    session_id: str | None = None,
    user_id: str = "user",
) -> dict:
    """
    Single text-in / text-out turn.
    Returns the final State dict so callers can inspect every field.
    """
    initial: State = {
        "input_format":  "text",
        "output_format": "text_and_voice",
        "user_query":    query,
        "session_id":    session_id or str(uuid.uuid4()),
        "user_id":       user_id,
        "regenerated_count": 0,
        "conversation_history": [],
    }
    t0 = time.monotonic()
    final = tutor_graph.invoke(initial)
    final["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return final


def run_voice_turn(
    audio_bytes: bytes,
    session_id: str | None = None,
    user_id: str = "user",
    output_format: str = "text_and_voice",
) -> dict:
    """
    Returns the final State dict.
    """
    initial: State = {
        "input_format":  "voice",
        "output_format": output_format,
        "raw_audio":     audio_bytes,
        "session_id":    session_id or str(uuid.uuid4()),
        "user_id":       user_id,
        "regenerated_count": 0,
        "conversation_history": [],
    }
    t0 = time.monotonic()
    final = tutor_graph.invoke(initial)
    final["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _print_result(result: dict) -> None:
    print("\n" + "═" * 60)
    print(f"  RESPONSE  (latency: {result.get('latency_ms', '?')} ms)")
    print("═" * 60)
    print(result.get("llm_response", ""))

    if result.get("sources"):
        print("\n  Sources:", ", ".join(result["sources"]))
    if result.get("error_message"):
        print(f"  [Error] {result['error_message']}")

    audio = result.get("audio_response")
    if audio:
        out_file = f"response_{int(time.time())}.wav"
        Path(out_file).write_bytes(audio)
        print(f"  Audio saved to: {out_file}")
    print()


def _interactive_loop(session_id: str, user_id: str) -> None:
    print("\n🎓  AI Tutor  (type 'exit' to quit)\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query or query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        result = run_text_turn(query, session_id=session_id, user_id=user_id)
        _print_result(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Tutor powered by LangGraph.")
    parser.add_argument("--query",      type=str,  default=None,   help="One-shot text query.")
    parser.add_argument("--voice",      type=str,  default=None,   help="Path to a WAV audio file.")
    parser.add_argument("--output",     type=str,  default="text_and_voice",
                        help="Desired output format.")
    parser.add_argument("--user-id",    type=str,  default="cli_user", help="User identifier.")
    parser.add_argument("--session-id", type=str,  default=None,   help="Resume an existing session.")
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())
    user_id    = args.user_id

    if args.voice:
        audio_path = Path(args.voice)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            sys.exit(1)
        audio_bytes = audio_path.read_bytes()
        result = run_voice_turn(
            audio_bytes,
            session_id=session_id,
            user_id=user_id,
            output_format=args.output,
        )
        _print_result(result)

    elif args.query:
        result = run_text_turn(args.query, session_id=session_id, user_id=user_id)
        _print_result(result)

    else:
        _interactive_loop(session_id=session_id, user_id=user_id)


if __name__ == "__main__":
    main()
