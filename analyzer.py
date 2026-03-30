import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib import error, request


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
CONVERSATION_LOG_PATH = LOG_DIR / "conversation_log.jsonl"
TASKS_LOG_PATH = LOG_DIR / "tasks.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:12b"
SYSTEM_PROMPT = """You are an expert in analyzing work-related conversations.
Extract the list of tasks the user attempted to perform from the conversation below.
task_name must be in Korean only.
Do not include English translation or romanization.
Respond only in the following JSON format. Do not include any other text.
{
  "tasks": [
    {
      "task_name": "task name in Korean",
      "status": "completed | in_progress | abandoned",
      "start_turn": 1
    }
  ]
}
"""


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_conversation_records():
    if not CONVERSATION_LOG_PATH.is_file():
        raise FileNotFoundError(f"Conversation log not found: {CONVERSATION_LOG_PATH}")

    records = []
    with CONVERSATION_LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in conversation log at line {line_number}: {exc}"
                ) from exc

    return records


def select_session_id(records, session_id=None):
    session_ids = [record.get("session_id") for record in records if record.get("session_id")]
    if not session_ids:
        raise ValueError("No session_id found in conversation log.")

    if session_id:
        if session_id not in session_ids:
            raise ValueError(f"Session not found: {session_id}")
        return session_id

    return max(session_ids)


def build_conversation_text(records, session_id):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        raise ValueError(f"No records found for session: {session_id}")

    session_records.sort(key=lambda item: (item.get("turn", 0), item.get("timestamp", ""), item.get("role", "")))

    lines = []
    for record in session_records:
        turn = record.get("turn", "")
        role = record.get("role", "unknown")
        content = record.get("content", "")
        lines.append(f"Turn {turn} [{role}]: {content}")

    return "\n".join(lines)


def call_ollama(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request) as response:
        body = response.read().decode("utf-8")

    parsed = json.loads(body)
    return parsed["message"]["content"]


def append_tasks_log(record):
    with TASKS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_code_block(text):
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def main() -> None:
    ensure_log_dir()

    session_id_arg = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        records = load_conversation_records()
        session_id = select_session_id(records, session_id_arg)
        conversation_text = build_conversation_text(records, session_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to prepare conversation: {exc}")
        sys.exit(1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conversation_text},
    ]

    try:
        raw_response = call_ollama(messages)
    except error.URLError as exc:
        print(f"Ollama request failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Ollama request failed: {exc}")
        sys.exit(1)

    result = {
        "session_id": session_id,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": [],
    }

    try:
        cleaned_response = strip_code_block(raw_response)
        parsed_response = json.loads(cleaned_response)
        tasks = parsed_response.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("`tasks` field is missing or is not a list.")
        result["tasks"] = tasks
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to parse Ollama JSON response: {exc}")
        result["raw_response"] = raw_response

    append_tasks_log(result)
    print(f"Saved analysis for session {session_id} to {TASKS_LOG_PATH}")


if __name__ == "__main__":
    main()
