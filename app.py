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

def _format_tool_args(args: dict) -> str:
    return ", ".join(f'{k}="{v}"' for k, v in args.items()) if args else ""


def _render_trace(steps: list[dict]) -> str:
    """Render the collected ReAct steps as a readable markdown trace showing
    each reasoning/tool-call step the agent took, for transparency in the UI."""
    if not steps:
        return "_No tool calls were needed for this response — the agent answered directly._"

    lines = []
    for i, step in enumerate(steps, start=1):
        if step["type"] == "tool_call":
            lines.append(
                f"**Step {i} — Tool call:** `{step['name']}({_format_tool_args(step['args'])})`\n\n"
                f"> {step['result']}"
            )
        elif step["type"] == "final":
            lines.append(f"**Step {i} — Final answer generated** using the information above.")
    return "\n\n---\n\n".join(lines)


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
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history, gr.update()
        return

    # Show a thinking placeholder while the agent runs
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "_Thinking..._"},
    ]
    yield history, gr.update(value="_Waiting for the agent to reason and call tools..._")

    trace_steps: list[dict] = []
    response_text = await chat_with_agent(user_message, history[:-2], trace_steps)
    history[-1] = {"role": "assistant", "content": response_text}
    yield history, gr.update(value=_render_trace(trace_steps))


CUSTOM_CSS = """
:root {
    --cc-radius: 14px;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
#cc-header {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
}
#cc-header h1 {
    font-size: 1.9rem;
    margin-bottom: 0.25rem;
}
#cc-header p {
    opacity: 0.75;
    font-size: 0.95rem;
}
#cc-chatbot {
    border-radius: var(--cc-radius) !important;
}
#cc-trace {
    border-radius: var(--cc-radius) !important;
    max-height: 520px;
    overflow-y: auto;
}
.cc-examples button {
    border-radius: 999px !important;
    font-size: 0.85rem !important;
}
/* Stack the chat and trace panels on narrow / mobile screens */
@media (max-width: 900px) {
    #cc-main-row {
        flex-direction: column !important;
    }
    #cc-header h1 {
        font-size: 1.5rem;
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
        with gr.Column(scale=3, min_width=320):
            chatbot = gr.Chatbot(
                elem_id="cc-chatbot",
                height=480,
                buttons=["copy"],
                avatar_images=(None, "🍽️"),
                label="Chat",
            )
            msg_input = gr.Textbox(
                label=None,
                show_label=False,
                placeholder='Ask about a restaurant, cuisine, or vibe — e.g. "Find me a moody spot in DTLA"',
                autofocus=True,
            )
            with gr.Row(elem_classes="cc-examples"):
                btn1 = gr.Button("🌙 Find moody restaurants", size="sm")
                btn2 = gr.Button("🥩 Tell me about Iron & Embers", size="sm")
                btn3 = gr.Button("🍵 Zen dining in Little Tokyo?", size="sm")

        with gr.Column(scale=2, min_width=280):
            gr.Markdown("### 🔍 Agent reasoning trace")
            trace_panel = gr.Markdown(
                "_Ask a question to see which tools the agent calls and what data it retrieves, step by step._",
                elem_id="cc-trace",
            )

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot, trace_panel])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot, trace_panel])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot, trace_panel])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot, trace_panel])

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
