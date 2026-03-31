import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib import error

from analyzer import (
    detect_goal_change,
    get_active_goal_record,
    get_next_goal_version,
    get_session_goal_history,
    load_goal_records,
    restore_previous_goal_version,
    save_goal_version,
)

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


def print_goals(
    goals: list[str], title: str = "오늘의 목표", auto_extracted: bool = False
) -> None:
    print("=============================")
    if auto_extracted:
        print(f"{title} (자동 추출)")
    else:
        print(title)
    print("=============================")
    if goals:
        for index, goal in enumerate(goals, start=1):
            print(f"{index}. {goal}")
    else:
        print("(목표 없음 - 자동 추출 예정)")
    print("=============================")


def format_goal_timestamp(timestamp: str) -> str:
    if not timestamp:
        return "0000-00-00 00:00"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp[:16].replace("T", " ")


def format_goal_list(goals: list[str]) -> str:
    return ", ".join(goal for goal in goals if goal) if goals else "(없음)"


def print_goal_history(goal_history: list[dict]) -> None:
    print("=============================")
    print("목표 변경 이력")
    print("=============================")
    if not goal_history:
        print("(이력 없음)")
    else:
        current_version = None
        for record in goal_history:
            if record.get("superseded") is False:
                current_version = record.get("version")
                break
        for record in goal_history:
            version = record.get("version", "?")
            timestamp = format_goal_timestamp(record.get("timestamp", ""))
            goals_text = format_goal_list(record.get("goals", []))
            source_label = "자동 감지" if record.get("auto_extracted") else "직접 입력"
            current_marker = " \u2190 현재 목표" if version == current_version else ""
            print(f"[{version}] {timestamp}  {goals_text}")
            print(f"                      ({source_label}){current_marker}")
    print("=============================")


def prompt_goal_history_action() -> str:
    print("현재 목표로 유지하시겠습니까?")
    while True:
        answer = input("(y: 유지 / n: 목표 수정 / undo: 이전 목표로 복원): ").strip().lower()
        if answer in {"y", "n", "undo"}:
            return answer


def prompt_goal_change_update(
    current_version: int, current_goals: list[str], next_version: int, new_goals: list[str]
) -> bool:
    print("=============================")
    print("업무 방향이 바뀐 것 같습니다.")
    print(f"현재 목표 (v{current_version}): {format_goal_list(current_goals)}")
    print(f"새로운 목표 (v{next_version}): {format_goal_list(new_goals)}")
    print("목표를 업데이트할까요? (y/n):")
    print("=============================")

    while True:
        answer = input().strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False


def main() -> str | None:
    ensure_log_dir()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = []
    session_records = []
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
    active_goal_record = save_goal_version(session_id, goals, auto_extracted=False)
    goals = active_goal_record.get("goals", [])

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
            goal_history = get_session_goal_history(load_goal_records(), session_id)
            print_goal_history(goal_history)
            action = prompt_goal_history_action()
            if action == "n":
                new_goals_input = input(
                    "새로운 목표를 입력하세요 (쉼표로 구분): "
                ).strip()
                goals = [
                    goal.strip()
                    for goal in new_goals_input.split(",")
                    if goal.strip()
                ]
                saved_record = save_goal_version(session_id, goals, auto_extracted=False)
                goals = saved_record.get("goals", [])
                print("목표가 업데이트됐습니다.")
            elif action == "undo":
                restored_record = restore_previous_goal_version(session_id)
                if restored_record is None:
                    print("복원할 이전 목표가 없습니다.")
                else:
                    goals = restored_record.get("goals", [])
                    print("목표가 복원됐습니다.")
            else:
                active_goal_record = get_active_goal_record(load_goal_records(), session_id)
                goals = active_goal_record.get("goals", []) if active_goal_record else goals
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
        session_records.append(
            {
                "session_id": session_id,
                "turn": turn,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "role": "user",
                "content": user_input,
            }
        )
        has_conversation = True

        try:
            assistant_reply = call_llm(messages)
        except Exception as exc:
            print(f"API call failed: {exc}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": assistant_reply})
        append_log(session_id, turn, "assistant", assistant_reply)
        session_records.append(
            {
                "session_id": session_id,
                "turn": turn,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "role": "assistant",
                "content": assistant_reply,
            }
        )

        print(f"Assistant: {assistant_reply}")

        if turn % 5 == 0:
            try:
                detection_result = detect_goal_change(session_id, goals, session_records)
            except error.URLError as exc:
                print(f"Goal change detection failed: {exc}")
                detection_result = {"goal_changed": False, "new_goals": []}
            except Exception as exc:
                print(f"Goal change detection failed: {exc}")
                detection_result = {"goal_changed": False, "new_goals": []}

            new_goals = detection_result.get("new_goals", [])
            if detection_result.get("goal_changed") and new_goals:
                goal_records = load_goal_records()
                active_goal_record = get_active_goal_record(goal_records, session_id)
                current_version = (
                    active_goal_record.get("version", 1) if active_goal_record else 1
                )
                next_version = get_next_goal_version(goal_records, session_id)
                if prompt_goal_change_update(
                    current_version, goals, next_version, new_goals
                ):
                    saved_record = save_goal_version(
                        session_id, new_goals, auto_extracted=True
                    )
                    goals = saved_record.get("goals", [])
                    print("목표가 업데이트됐습니다.")

        turn += 1

    return session_id


if __name__ == "__main__":
    main()
