# Libraries to create our MCP host application
import os
import re
import gradio as gr
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastmcp.client import Client, PythonStdioTransport
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load environment variables (e.g., GEMINI_API_KEY) from .env
load_dotenv()

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
SYSTEM_PROMPT = """You are Connoisseur Companion, a friendly and knowledgeable food and dining assistant.

You have access to a database of California restaurants and can use the following tools to help users:

- get_restaurant_info: Use this to look up specific restaurants by name and return their structured details (cuisine, rating, price range, signature dishes, etc.).
- recommend_by_vibe: Use this to find restaurants that match a mood, atmosphere, or vibe the user describes (e.g., "romantic", "casual", "lively").
- get_review: Use this to retrieve detailed user reviews for a specific restaurant.

Choose the most appropriate tool based on what the user is asking for, and use the information returned by the tools to craft helpful, conversational responses. If a user's request is ambiguous, ask a clarifying question before calling a tool."""

# Initializing the Gemini LLM
def make_model():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.7,
    )

# Rate-Limit Handling
def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini quota/rate-limit errors (HTTP 429 / RESOURCE_EXHAUSTED)
    anywhere in the exception chain (including the original google.genai
    ClientError wrapped by langchain_google_genai)."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "code", None) == 429:
            return True
        text = str(current)
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            return True
        current = getattr(current, "__cause__", None)
    return False


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    """Walk the exception chain looking for Google's RetryInfo.retryDelay
    (e.g. "32s"), falling back to parsing "Please retry in Ns" from the
    error text if the structured field isn't available."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        details = getattr(current, "details", None)
        if isinstance(details, dict):
            error_details = details.get("error", {}).get("details", [])
            for item in error_details:
                if isinstance(item, dict) and "RetryInfo" in item.get("@type", ""):
                    match = re.match(r"([\d.]+)s", str(item.get("retryDelay", "")))
                    if match:
                        return float(match.group(1))

        match = re.search(r"retry in ([\d.]+)s", str(current))
        if match:
            return float(match.group(1))

        current = getattr(current, "__cause__", None)
    return None


def _format_rate_limit_message(exc: Exception) -> str:
    """Build a user-friendly message telling the user exactly when
    (in hours and minutes from now) they'll be able to chat again."""
    delay_seconds = _extract_retry_delay_seconds(exc)
    if delay_seconds is None:
        delay_seconds = 60  # sensible fallback if Google didn't report a delay

    reset_time = datetime.now() + timedelta(seconds=delay_seconds)
    total_minutes = max(1, round(delay_seconds / 60))
    hours, minutes = divmod(total_minutes, 60)

    if hours and minutes:
        wait_str = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    elif hours:
        wait_str = f"{hours} hour{'s' if hours != 1 else ''}"
    else:
        wait_str = f"{minutes} minute{'s' if minutes != 1 else ''}"

    return (
        "⏳ **Token limit reached.** Connoisseur Companion has hit its usage limit for now.\n\n"
        f"Please try again in **{wait_str}** (around **{reset_time.strftime('%I:%M %p')}**)."
    )

# Maps each MCP tool to the specialized agent (from the Module 3 multi-agent
# design) conceptually responsible for that kind of reasoning. The production
# chatbot runs a single ReAct loop for latency/cost reasons (see README), but
# each tool still maps 1:1 to a specialist role — this label makes that
# multi-agent lineage visible in the live trace.
TOOL_AGENT_MAP = {
    "get_restaurant_info": ("🔎", "RAG Retriever"),
    "recommend_by_vibe": ("🎨", "Food Style Expert"),
    "get_review": ("📰", "Food Trend Analyst"),
}
FINAL_AGENT = ("🧑‍🍳", "Recommendation Expert")


def _format_tool_args(args: dict) -> str:
    return ", ".join(f'{k}="{v}"' for k, v in args.items()) if args else ""


def _render_turn_trace(steps: list[dict]) -> str:
    """Render one turn's ReAct steps as a readable markdown trace, labeling
    each tool call with the specialized agent role responsible for it."""
    if not steps:
        return "_No tool calls were needed — the Recommendation Expert answered directly._"

    lines = []
    for i, step in enumerate(steps, start=1):
        if step["type"] == "tool_call":
            emoji, agent_name = TOOL_AGENT_MAP.get(step["name"], ("🛠️", "Agent"))
            lines.append(
                f"**Step {i} — {emoji} {agent_name}** called `{step['name']}({_format_tool_args(step['args'])})`\n\n"
                f"> {step['result']}"
            )
        elif step["type"] == "final":
            emoji, agent_name = FINAL_AGENT
            lines.append(f"**Step {i} — {emoji} {agent_name}** synthesized the final answer from the results above.")
    return "\n\n".join(lines)


def _render_full_trace(turns: list[dict]) -> str:
    """Render the trace for every turn in the session so far, each clearly
    labeled with the user message it responsds to, so the whole conversation's
    agent reasoning can be scrolled through from the beginning."""
    if not turns:
        return "_Ask a question to see which specialized agent/tool handles it, step by step._"

    blocks = []
    for i, turn in enumerate(turns, start=1):
        preview = turn["user_message"]
        if len(preview) > 80:
            preview = preview[:80] + "…"
        blocks.append(f"#### 💬 Turn {i} — \"{preview}\"\n\n{_render_turn_trace(turn['steps'])}")
    return "\n\n---\n\n".join(blocks)


# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list, trace_steps: list) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response. Each tool call is
    recorded into trace_steps so the UI can display the agent's reasoning."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        # Discover available tools from the MCP server
        mcp_tools = await client.list_tools()

        # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = make_model().bind_tools(openai_tools)

        # Build the message list from chat history and the new user message
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # ReAct loop — call tools until the LLM returns a plain text reply
        try:
            for _ in range(10):
                response = await model.ainvoke(messages)
                messages.append(response)

                # No tool calls means the LLM is done — return the final response
                if not response.tool_calls:
                    trace_steps.append({"type": "final"})
                    raw = response.content
                    if isinstance(raw, list):
                        return " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in raw
                        )
                    return str(raw)

                # Execute each tool call via the MCP server and feed results back
                for tool_call in response.tool_calls:
                    result = await client.call_tool(tool_call["name"], tool_call["args"])
                    tool_output = " ".join(
                        item.text if hasattr(item, "text") else str(item)
                        for item in result.content
                    ) if result.content else "(no result)"
                    trace_steps.append(
                        {
                            "type": "tool_call",
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                            "result": tool_output[:400] + ("…" if len(tool_output) > 400 else ""),
                        }
                    )
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

            return "I wasn't able to complete that request. Please try again."
        except Exception as e:
            if _is_rate_limit_error(e):
                return _format_rate_limit_message(e)
            raise

# Gradio Event Handler
async def handle_chat(user_message, history, trace_history):
    if history is None:
        history = []
    if trace_history is None:
        trace_history = []
    if not user_message or not user_message.strip():
        yield history, trace_history, gr.update()
        return

    # Show a thinking placeholder while the agent runs
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "_Thinking..._"},
    ]
    pending_trace = trace_history + [{"user_message": user_message, "steps": []}]
    yield history, trace_history, gr.update(value=_render_full_trace(pending_trace))

    trace_steps: list[dict] = []
    response_text = await chat_with_agent(user_message, history[:-2], trace_steps)
    history[-1] = {"role": "assistant", "content": response_text}
    trace_history = trace_history + [{"user_message": user_message, "steps": trace_steps}]
    yield history, trace_history, gr.update(value=_render_full_trace(trace_history))


CUSTOM_CSS = """
:root {
    --cc-radius: 14px;
}
html, body {
    height: 100%;
    overflow: hidden;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0.5rem 1rem !important;
    box-sizing: border-box !important;
}
#cc-header {
    text-align: center;
    padding: 0.25rem 0 0.5rem 0;
    flex-shrink: 0;
}
#cc-header h1 {
    font-size: 1.5rem;
    margin-bottom: 0.15rem;
}
#cc-header p {
    opacity: 0.75;
    font-size: 0.9rem;
    margin: 0;
}
#cc-main-row {
    flex: 1 1 auto;
    min-height: 0;
}
#cc-chat-col, #cc-trace-col {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
#cc-chatbot {
    border-radius: var(--cc-radius) !important;
    flex: 1 1 auto;
    min-height: 0;
}
#cc-msg-input {
    flex-shrink: 0;
    margin-top: 0.4rem;
}
#cc-msg-input label span {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
#cc-msg-input textarea, #cc-msg-input input {
    border: 2px solid var(--color-accent) !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 0.65rem 0.9rem !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
}
#cc-msg-input textarea::placeholder, #cc-msg-input input::placeholder {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#cc-trace-col > div {
    min-height: 0;
}
#cc-trace {
    border-radius: var(--cc-radius) !important;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto !important;
    border: 1px solid var(--border-color-primary);
    padding: 0.75rem;
}
.cc-examples {
    flex-shrink: 0;
}
.cc-examples button {
    border-radius: 999px !important;
    font-size: 0.8rem !important;
}
/* Stack the chat and trace panels on narrow / mobile screens */
@media (max-width: 900px) {
    html, body {
        overflow: auto;
    }
    .gradio-container {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }
    #cc-main-row {
        flex-direction: column !important;
    }
    #cc-chatbot {
        height: 420px !important;
    }
    #cc-trace {
        max-height: 320px;
    }
    #cc-header h1 {
        font-size: 1.3rem;
    }
}
"""

# Gradio Interface
with gr.Blocks(title="Connoisseur Companion") as demo:
    with gr.Column(elem_id="cc-header"):
        gr.Markdown(
            "# 🍽️ Connoisseur Companion\n"
            "Your AI guide to California's restaurant scene — powered by a multi-tool agent "
            "over a Retrieval-Augmented restaurant &amp; review dataset."
        )

    with gr.Row(elem_id="cc-main-row", equal_height=True):
        with gr.Column(scale=3, min_width=320, elem_id="cc-chat-col"):
            chatbot = gr.Chatbot(
                elem_id="cc-chatbot",
                buttons=["copy"],
                label="Chat",
            )
            msg_input = gr.Textbox(
                elem_id="cc-msg-input",
                label="Type your message",
                show_label=True,
                placeholder='e.g. "Find me a moody spot in DTLA"',
                autofocus=True,
            )
            with gr.Row(elem_classes="cc-examples"):
                btn1 = gr.Button("🍝 Recommend a restaurant for a special occasion", size="sm")
                btn2 = gr.Button("💸 Find a good casual spot that won't break the bank", size="sm")
                btn3 = gr.Button("⭐ What's your top-rated pick right now?", size="sm")

        with gr.Column(scale=2, min_width=280, elem_id="cc-trace-col"):
            gr.Markdown("### 🔍 Multi-Agent Reasoning Trace")
            gr.Markdown(
                "_Shows which specialized agent/tool handled each message, for the whole conversation. Scroll to review earlier turns._",
                elem_id="cc-trace-caption",
            )
            trace_panel = gr.Markdown(
                "_Ask a question to see which specialized agent/tool handles it, step by step._",
                elem_id="cc-trace",
            )

    trace_state = gr.State([])  # Accumulates {"user_message", "steps"} for every turn this session

    msg_input.submit(handle_chat, [msg_input, chatbot, trace_state], [chatbot, trace_state, trace_panel])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Recommend a restaurant for a special occasion"), chatbot, trace_state], [chatbot, trace_state, trace_panel])
    btn2.click(handle_chat, [gr.State("Find a good casual spot that won't break the bank"), chatbot, trace_state], [chatbot, trace_state, trace_panel])
    btn3.click(handle_chat, [gr.State("What's your top-rated pick right now?"), chatbot, trace_state], [chatbot, trace_state, trace_panel])

# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    # Hugging Face Spaces sets SPACE_ID and Render sets PORT/RENDER automatically,
    # and both already expose the app on a public URL, so the extra gradio.live
    # share tunnel is only needed when running locally.
    running_on_host = bool(os.environ.get("SPACE_ID") or os.environ.get("RENDER"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=not running_on_host,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    )
