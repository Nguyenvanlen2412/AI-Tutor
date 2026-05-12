"""
graph.py – assembles the AI Tutor LangGraph StateGraph.

Full pipeline topology:

  START
    │
    ▼
  get_user_input
    │
    ├─ (voice) ──► speech_to_text
    │                     │
    │ (text) ──────────────┘
    │
    ▼
  check_input_vulnerability
    │
    ├─ (unsafe) ──► handle_input_vulnerability ──┐
    │                                             │
    │ (safe) ──────────────────────────────────────┐
    ▼                                               │
  retrieve_context                                  │
    │                                               │
    ├─ (cache hit) ──────────────────────────────┐  │
    │                                            │  │
    │ (cache miss) ──► create_response ──────────┘  │
    │                                               │
    ▼                                               │
  check_output_vulnerability  ◄──────────────────────┘
    │
    ├─ (unsafe & retries) ──► handle_output_vulnerability
    │                                  │
    │                     ┌─ (retries left) ──► create_response (loop)
    │                     └─ (exhausted)   ──► text_to_speech | save_context
    │
    │ (safe, voice) ──► text_to_speech ──► save_context ──► END
    │ (safe, text)  ──────────────────────► save_context ──► END
"""

from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    get_user_input,
    speech_to_text,
    check_input_vulnerability,
    handle_input_vulnerability,
    retrieve_context,
    create_response,
    check_output_vulnerability,
    handle_output_vulnerability,
    text_to_speech,
    save_context,
    # routing functions
    route_after_input,
    route_after_input_safety,
    route_after_retrieval,
    route_after_output_safety,
    route_after_handle_output,
)


def build_graph() -> StateGraph:
    """Construct and compile the AI Tutor LangGraph."""
    builder = StateGraph(State)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node("get_user_input",            get_user_input)
    builder.add_node("speech_to_text",            speech_to_text)
    builder.add_node("check_input_vulnerability", check_input_vulnerability)
    builder.add_node("handle_input_vulnerability",handle_input_vulnerability)
    builder.add_node("retrieve_context",          retrieve_context)
    builder.add_node("create_response",           create_response)
    builder.add_node("check_output_vulnerability",check_output_vulnerability)
    builder.add_node("handle_output_vulnerability",handle_output_vulnerability)
    builder.add_node("text_to_speech",            text_to_speech)
    builder.add_node("save_context",              save_context)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.add_edge(START, "get_user_input")

    # ── Conditional: text vs voice input ──────────────────────────────────────
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "speech_to_text":            "speech_to_text",
            "check_input_vulnerability": "check_input_vulnerability",
        },
    )

    # ── STT always feeds into safety check ───────────────────────────────────
    builder.add_edge("speech_to_text", "check_input_vulnerability")

    # ── Conditional: safe vs unsafe input ────────────────────────────────────
    builder.add_conditional_edges(
        "check_input_vulnerability",
        route_after_input_safety,
        {
            "handle_input_vulnerability": "handle_input_vulnerability",
            "retrieve_context":           "retrieve_context",
        },
    )

    # ── Blocked input: bypass LLM, go straight to TTS or save ────────────────
    # (handle_input_vulnerability sets output_safety_status="safe" on the fallback)
    builder.add_edge(
        "handle_input_vulnerability",            "text_to_speech"
    )

    # ── Conditional: cache hit → skip LLM ────────────────────────────────────
    builder.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "check_output_vulnerability": "check_output_vulnerability",
            "create_response":            "create_response",
        },
    )

    # ── LLM → output safety ───────────────────────────────────────────────────
    builder.add_edge("create_response", "check_output_vulnerability")

    # ── Conditional: safe output → TTS/save  |  unsafe → regen loop ─────────
    builder.add_conditional_edges(
        "check_output_vulnerability",
        route_after_output_safety,
        {
            "handle_output_vulnerability": "handle_output_vulnerability",
            "text_to_speech":              "text_to_speech",        },
    )

    # ── Regen loop: try again OR fall through ─────────────────────────────────
    builder.add_conditional_edges(
        "handle_output_vulnerability",
        route_after_handle_output,
        {
            "create_response":  "create_response",
            "text_to_speech":   "text_to_speech",        },
    )

    # ── TTS → save ────────────────────────────────────────────────────────────
    builder.add_edge("text_to_speech", "save_context")

    # ── Terminal ──────────────────────────────────────────────────────────────
    builder.add_edge("save_context", END)

    return builder.compile()


# Pre-compiled singleton
tutor_graph = build_graph()

# Save graph visualization
graph_image = tutor_graph.get_graph(xray=True).draw_mermaid_png()
with open("tutor_graph.png", "wb") as f:
    f.write(graph_image)
print("Graph saved to tutor_graph.png")

