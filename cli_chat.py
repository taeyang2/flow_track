import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai")
    print("Install it with: python3 -m pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency: python-dotenv")
    print("Install it with: python3 -m pip install python-dotenv")
    sys.exit(1)


MODEL_NAME = "gpt-4o-mini"
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = Path.home() / "toy" / "flow_track" / "logs" / "conversation_log.jsonl"
GOAL_LOG_PATH = Path.home() / "toy" / "flow_track" / "logs" / "goal.jsonl"

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_goal_log(session_id: str, goals: list[str]) -> None:
    record = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "goals": goals,
    }
    with GOAL_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_log(session_id: str, turn: int, role: str, content: str) -> None:
    record = {
        "session_id": session_id,
        "turn": turn,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(messages):
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    ensure_log_dir()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = []
    turn = 1
    goals_input = input("오늘 목표를 입력하세요 (쉼표로 구분): ").strip()
    goals = [goal.strip() for goal in goals_input.split(",") if goal.strip()]
    append_goal_log(session_id, goals)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing.")
        print(f"Set it in the shell or add it to {BASE_DIR / '.env'}")
        sys.exit(1)

    print(f"CLI chat started. Model: {MODEL_NAME}")
    print("Exit with /quit")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Exiting.")
            break

        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        append_log(session_id, turn, "user", user_input)

        try:
            assistant_reply = call_llm(messages)
        except Exception as exc:
            print(f"API call failed: {exc}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": assistant_reply})
        append_log(session_id, turn, "assistant", assistant_reply)

        print(f"Assistant: {assistant_reply}")
        turn += 1


if __name__ == "__main__":
    main()
