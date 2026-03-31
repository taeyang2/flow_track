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
EXIT_TRIGGERS = {
    "/quit",
    "quit",
    "/exit",
    "exit",
    "q",
    "종료",
    "대화 종료",
    "안녕",
    "끝",
    "그만",
    "bye",
    "goodbye",
    "see you",
}
GOAL_COMMANDS = {"/goal"}

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


def confirm_exit() -> bool:
    while True:
        answer = input("대화를 종료하시겠습니까? (y/n): ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False


def confirm_program_exit() -> bool:
    while True:
        answer = input("프로그램을 종료하시겠습니까? (y/n): ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False


def print_goals(goals: list[str]) -> None:
    print("=============================")
    print("오늘의 목표")
    print("=============================")
    if goals:
        for index, goal in enumerate(goals, start=1):
            print(f"{index}. {goal}")
    else:
        print("(목표 없음 - 자동 추출 예정)")
    print("=============================")


def main() -> str | None:
    ensure_log_dir()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = []
    has_conversation = False
    turn = 1
    while True:
        goals_input = input("오늘 목표를 입력하세요 (쉼표로 구분): ").strip()
        if goals_input.lower() in EXIT_TRIGGERS:
            if confirm_program_exit():
                print("Exiting.")
                return None
            continue
        break

    goals = [goal.strip() for goal in goals_input.split(",") if goal.strip()]
    append_goal_log(session_id, goals)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing.")
        print(f"Set it in the shell or add it to {BASE_DIR / '.env'}")
        sys.exit(1)

    print_goals(goals)
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

        if user_input.lower() in GOAL_COMMANDS:
            print_goals(goals)
            continue

        if user_input.lower() in EXIT_TRIGGERS:
            if confirm_exit():
                print("Exiting.")
                if not has_conversation:
                    print("대화 내용이 없어 분석을 건너뜁니다.")
                    return None
                break
            continue

        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        append_log(session_id, turn, "user", user_input)
        has_conversation = True

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

    return session_id


if __name__ == "__main__":
    main()
