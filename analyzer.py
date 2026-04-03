import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
CONVERSATION_LOG_PATH = LOG_DIR / "conversation_log.jsonl"
CONVERSATION_SNAPSHOT_PATH = LOG_DIR / "conversation_log.snapshot.jsonl"
TASKS_LOG_PATH = LOG_DIR / "tasks.jsonl"
GOAL_LOG_PATH = LOG_DIR / "goal.jsonl"
GOAL_TMP_PATH = LOG_DIR / "goal.jsonl.tmp"
#OLLAMA_URL = "http://localhost:11434/api/chat"
#OLLAMA_MODEL = "gemma2:12b"
OLLAMA_URL = "http://localhost:11444/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
# SYSTEM_PROMPT = """You are an expert in analyzing work-related conversations.
# Extract the list of tasks the user attempted to perform from the conversation below.
# task_name must be in Korean only.
# Do not include English translation or romanization.
# Respond only in the following JSON format. Do not include any other text.
# {
#   "tasks": [
#     {
#       "task_name": "task name in Korean",
#       "status": "completed | in_progress | abandoned",
#       "start_turn": 1
#     }
#   ]
# }
# """
SYSTEM_PROMPT = """Extract tasks from the conversation below.
Respond only in JSON format:
{
  "tasks": [
    {
      "task_name": "task name in Korean",
      "status": "completed | in_progress | abandoned",
      "start_turn": 1
    }
  ]
}
task_name must be in Korean only.
Do not include any other text."""
GOAL_RELEVANCE_SYSTEM_PROMPT = """You are a work flow analyst.
Given a goal list, a task, and the surrounding conversation context,
determine if the task is relevant to achieving the goals.

Use the conversation context to understand why the user performed
this task and whether it naturally follows from the goals.

Classify as relevant (true) only if:
1. The task is directly related to the goals.
2. The task is a necessary sub-task to achieve the goals,
   based on the conversation context.

Classify as irrelevant (false) if:
1. The task is casual conversation or off-topic chatting.
2. The task is exploring interesting but unnecessary information
   such as trivia, history, advertisements, or background knowledge.
3. The task is clearly unrelated to the goals.

When in doubt, classify as irrelevant (false).

Respond only in JSON format:
{"relevant": true} or {"relevant": false}
"""
GOAL_EXTRACTION_SYSTEM_PROMPT = """You are a work flow analyst.
Look at the first few turns of this conversation and extract
the user's main work goals.
goals must be in Korean only.
Do not include English translation or romanization.
Respond only in JSON format:
{"goals": ["goal1", "goal2", ...]}
Do not include more than 3 goals.
"""
GOAL_CHANGE_DETECTION_SYSTEM_PROMPT = """You are a work flow analyst.
Look at the recent conversation and the current goals.
Determine if the user has shifted to a completely different work focus.

Classify as goal_changed (true) only if:
1. The recent conversation is clearly about a different work topic
   that has continued for multiple turns.
2. The user shows no intention of returning to the original goals.

Classify as goal_changed (false) if:
1. The conversation is a temporary detour (1~2 turns).
2. The conversation is still related to the original goals.
3. The conversation is exploring sub-topics or details
   within the same category as the current goals.
   (e.g. goal: "research restaurants in Suwon"
    → conversation about "famous chicken restaurants"
    → conversation about "reviews of a specific restaurant"
    These are all sub-topics of the same goal, NOT a goal change.)
4. The user is narrowing down or drilling deeper into
   the current goal topic.

Only classify as goal_changed (true) if the conversation
has completely shifted to an unrelated domain for multiple turns
and shows no sign of returning to the original goal.

Respond only in JSON format:
{
  "goal_changed": true or false,
  "new_goals": ["새로운 목표1", "새로운 목표2"]
}
new_goals must be in Korean only.
If goal_changed is false, new_goals should be an empty list.
"""
CYCLE_DETECTION_SYSTEM_PROMPT = """Determine if the new task is a return to a previous task (cycle).
Respond only in JSON: {"is_cycle": true, "cycle_target_turn": 1} or {"is_cycle": false, "cycle_target_turn": null}
"""


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def create_conversation_snapshot():
    if not CONVERSATION_LOG_PATH.is_file():
        raise FileNotFoundError(f"Conversation log not found: {CONVERSATION_LOG_PATH}")

    raw_lines = CONVERSATION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    last_index = len(raw_lines) - 1

    with CONVERSATION_SNAPSHOT_PATH.open("w", encoding="utf-8") as snapshot_file:
        for index, line in enumerate(raw_lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                snapshot_file.write(stripped + "\n")
            except json.JSONDecodeError:
                if index == last_index:
                    continue
                raise ValueError(
                    f"Invalid JSON in conversation log at line {index + 1}"
                )


def load_conversation_records():
    if not CONVERSATION_SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(f"Conversation snapshot not found: {CONVERSATION_SNAPSHOT_PATH}")

    records = []
    with CONVERSATION_SNAPSHOT_PATH.open("r", encoding="utf-8") as snapshot_file:
        for line_number, line in enumerate(snapshot_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in conversation snapshot at line {line_number}: {exc}"
                ) from exc

    return records


def _parse_goal_records_from_file():
    records = []
    with GOAL_LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in goal log at line {line_number}: {exc}"
                ) from exc
    return records


def load_goal_records():
    if not GOAL_LOG_PATH.is_file():
        return []

    try:
        return _parse_goal_records_from_file()
    except ValueError:
        time.sleep(0.1)
        if not GOAL_LOG_PATH.is_file():
            return []
        return _parse_goal_records_from_file()


def select_session_id(records, session_id=None):
    session_ids = [record.get("session_id") for record in records if record.get("session_id")]
    if not session_ids:
        raise ValueError("No session_id found in conversation log.")

    if session_id:
        if session_id not in session_ids:
            raise ValueError(f"Session not found: {session_id}")
        return session_id

    return max(session_ids)


def build_conversation_text(records, session_id, max_turns=10):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        raise ValueError(f"No records found for session: {session_id}")

    session_records.sort(key=lambda item: (item.get("turn", 0), item.get("timestamp", ""), item.get("role", "")))

    all_turns = sorted({record.get("turn") for record in session_records})
    recent_turns = set(all_turns[-max_turns:])
    session_records = [record for record in session_records if record.get("turn") in recent_turns]

    lines = []
    for record in session_records:
        if record.get("role") != "user":
            continue
        turn = record.get("turn", "")
        content = record.get("content", "")
        lines.append(f"Turn {turn} [user]: {content}")

    return "\n".join(lines)


def build_initial_turns_text(records, session_id, max_turn=5):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        raise ValueError(f"No records found for session: {session_id}")

    session_records.sort(key=lambda item: (item.get("turn", 0), item.get("timestamp", ""), item.get("role", "")))

    lines = []
    for record in session_records:
        turn = record.get("turn", 0)
        if turn > max_turn:
            continue

        role = record.get("role", "unknown")
        content = record.get("content", "")
        lines.append(f"Turn {turn} [{role}]: {content}")

    return "\n".join(lines)


def build_recent_turns_text(records, session_id, last_n_turns=5):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        return ""

    session_records.sort(
        key=lambda item: (item.get("turn", 0), item.get("timestamp", ""), item.get("role", ""))
    )
    turn_numbers = sorted(
        {
            record.get("turn")
            for record in session_records
            if isinstance(record.get("turn"), int)
        }
    )
    recent_turns = set(turn_numbers[-last_n_turns:])
    if not recent_turns:
        return ""

    lines = []
    for record in session_records:
        turn = record.get("turn", 0)
        if turn not in recent_turns:
            continue
        role = record.get("role", "unknown")
        content = record.get("content", "")
        lines.append(f"Turn {turn} [{role}]: {content}")

    return "\n".join(lines)


def call_ollama(messages):
    patched_messages = []
    for message in messages:
        if message.get("role") == "system":
            patched_messages.append({
                "role": "system",
                "content": "You are a JSON-only responder. Always respond with valid JSON and nothing else. No explanations, no markdown, no prose.\n\n" + message["content"],
            })
        else:
            patched_messages.append(message)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": patched_messages,
        "stream": False,
        "format": "json",
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


def write_goal_records(records):
    with GOAL_TMP_PATH.open("w", encoding="utf-8") as tmp_file:
        for record in records:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(GOAL_TMP_PATH, GOAL_LOG_PATH)


def normalize_session_goal_records(session_goal_records):
    if not session_goal_records:
        return []

    sorted_records = sorted(
        session_goal_records,
        key=lambda item: (item.get("timestamp", ""), item.get("version", 0)),
    )
    normalized_records = []
    total_records = len(sorted_records)
    for index, record in enumerate(sorted_records, start=1):
        normalized_record = dict(record)
        goals = normalized_record.get("goals", [])
        normalized_record["goals"] = goals if isinstance(goals, list) else []
        normalized_record["version"] = index
        normalized_record["auto_extracted"] = normalized_record.get("auto_extracted") is True
        normalized_record["superseded"] = index != total_records
        normalized_records.append(normalized_record)

    return normalized_records


def get_session_goal_history(goal_records, session_id):
    session_goal_records = [
        record for record in goal_records if record.get("session_id") == session_id
    ]
    return normalize_session_goal_records(session_goal_records)


def get_active_goal_record(goal_records, session_id):
    session_goal_records = get_session_goal_history(goal_records, session_id)
    active_records = [
        record for record in session_goal_records if record.get("superseded") is False
    ]
    if not active_records:
        return None
    return active_records[-1]


def get_next_goal_version(goal_records, session_id):
    session_goal_records = get_session_goal_history(goal_records, session_id)
    if not session_goal_records:
        return 1
    return session_goal_records[-1].get("version", 0) + 1


def save_goal_version(session_id, goals, auto_extracted=False):
    goal_records = load_goal_records()
    other_records = [
        record for record in goal_records if record.get("session_id") != session_id
    ]
    session_goal_records = get_session_goal_history(goal_records, session_id)

    updated_session_records = []
    for record in session_goal_records:
        updated_record = dict(record)
        updated_record["superseded"] = True
        updated_session_records.append(updated_record)

    updated_session_records.append(
        {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "goals": goals,
            "auto_extracted": auto_extracted,
            "version": len(session_goal_records) + 1,
            "superseded": False,
        }
    )

    write_goal_records(other_records + updated_session_records)
    return updated_session_records[-1]


def restore_previous_goal_version(session_id):
    goal_records = load_goal_records()
    session_goal_records = get_session_goal_history(goal_records, session_id)
    if len(session_goal_records) < 2:
        return None

    previous_record = session_goal_records[-2]
    restored_goals = previous_record.get("goals", [])
    return save_goal_version(session_id, restored_goals, auto_extracted=False)


def load_task_records():
    if not TASKS_LOG_PATH.is_file():
        raise FileNotFoundError(f"Tasks log not found: {TASKS_LOG_PATH}")

    records = []
    with TASKS_LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in tasks log at line {line_number}: {exc}"
                ) from exc

    return records


def load_existing_tasks(session_id):
    """Return latest task state for the session.

    existing_tasks: list of task dicts already saved (may be empty)
    existing_cycle_edges: list of cycle edge dicts already saved (may be empty)
    last_start_turn: highest start_turn seen, or 0 if none
    """
    if not TASKS_LOG_PATH.is_file():
        return [], [], 0

    try:
        records = load_task_records()
    except (FileNotFoundError, ValueError):
        return [], [], 0

    session_records = [r for r in records if r.get("session_id") == session_id]
    if not session_records:
        return [], [], 0

    session_records.sort(key=lambda item: item.get("analyzed_at", ""))
    latest_record = session_records[-1]
    tasks = latest_record.get("tasks", [])
    cycle_edges = latest_record.get("cycle_edges", [])
    if not isinstance(tasks, list):
        tasks = []
    if not isinstance(cycle_edges, list):
        cycle_edges = []

    last_start_turn = max(
        (t.get("start_turn", 0) for t in tasks if isinstance(t.get("start_turn"), int)),
        default=0,
    )
    return tasks, cycle_edges, last_start_turn


def build_conversation_text_after_turn(records, session_id, after_turn):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        raise ValueError(f"No records found for session: {session_id}")

    session_records.sort(key=lambda item: (item.get("turn", 0), item.get("timestamp", ""), item.get("role", "")))

    lines = []
    for record in session_records:
        turn = record.get("turn", 0)
        if isinstance(turn, int) and turn <= after_turn:
            continue
        if record.get("role") != "user":
            continue
        content = record.get("content", "")
        lines.append(f"Turn {turn} [user]: {content}")

    return "\n".join(lines)


def append_tasks_log(record):
    with TASKS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_code_block(text):
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def load_session_goals(goal_records, session_id):
    active_record = get_active_goal_record(goal_records, session_id)
    if not active_record:
        return []
    goals = active_record.get("goals", [])
    return goals if isinstance(goals, list) else []


def extract_goals_from_initial_turns(initial_turns_text):
    messages = [
        {"role": "system", "content": GOAL_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": initial_turns_text},
    ]
    raw_response = call_ollama(messages)
    cleaned_response = strip_code_block(raw_response)
    parsed_response = json.loads(cleaned_response)
    goals = parsed_response.get("goals")
    if not isinstance(goals, list):
        raise ValueError("`goals` field is missing or is not a list.")

    normalized_goals = []
    for goal in goals[:3]:
        if isinstance(goal, str):
            stripped_goal = goal.strip()
            if stripped_goal:
                normalized_goals.append(stripped_goal)

    return normalized_goals


def detect_goal_change(session_id, current_goals, conversation_records):
    if not current_goals:
        return {"goal_changed": False, "new_goals": []}

    recent_conversation = build_recent_turns_text(
        conversation_records, session_id, last_n_turns=5
    )
    if not recent_conversation:
        return {"goal_changed": False, "new_goals": []}

    payload = {
        "current_goals": current_goals,
        "recent_conversation": recent_conversation,
    }
    messages = [
        {"role": "system", "content": GOAL_CHANGE_DETECTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw_response = call_ollama(messages)
    cleaned_response = strip_code_block(raw_response)
    parsed_response = json.loads(cleaned_response)

    goal_changed = parsed_response.get("goal_changed")
    new_goals = parsed_response.get("new_goals")

    if not isinstance(goal_changed, bool):
        raise ValueError("`goal_changed` field is missing or is not a boolean.")
    if not isinstance(new_goals, list):
        raise ValueError("`new_goals` field is missing or is not a list.")

    normalized_goals = []
    for goal in new_goals[:3]:
        if isinstance(goal, str):
            stripped_goal = goal.strip()
            if stripped_goal:
                normalized_goals.append(stripped_goal)

    if not goal_changed:
        return {"goal_changed": False, "new_goals": []}

    return {"goal_changed": True, "new_goals": normalized_goals}


def build_task_context(records, start_turn, turn_window=1):
    if not isinstance(start_turn, int):
        return []

    min_turn = max(1, start_turn - turn_window)
    max_turn = start_turn + turn_window
    session_records = [
        record
        for record in records
        if min_turn <= record.get("turn", 0) <= max_turn
        and record.get("role") == "user"
    ]
    session_records.sort(key=lambda item: (item.get("turn", 0), item.get("timestamp", "")))

    return [record.get("content", "") for record in session_records]


def evaluate_task_deviation(task, goals, conversation_records):
    if not goals:
        return False

    start_turn = task.get("start_turn", 0)
    context = build_task_context(conversation_records, start_turn)
    goals_str = json.dumps(goals, ensure_ascii=False)
    context_str = json.dumps(context, ensure_ascii=False)
    user_content = f"goals: {goals_str}, task: \"{task.get('task_name', '')}\", context: {context_str}"
    messages = [
        {"role": "system", "content": GOAL_RELEVANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw_response = call_ollama(messages)
    cleaned_response = strip_code_block(raw_response)
    parsed_response = json.loads(cleaned_response)
    relevant = parsed_response.get("relevant")
    if not isinstance(relevant, bool):
        print(f"[DEBUG] evaluate_task_deviation raw response:\n{raw_response}")
        raise ValueError("`relevant` field is missing or is not a boolean.")
    return not relevant


def detect_cycle_edges(new_tasks, existing_tasks, conversation_records):
    if not new_tasks or not existing_tasks:
        return []

    existing_task_payload = []
    valid_existing_turns = set()
    for task in existing_tasks:
        start_turn = task.get("start_turn")
        if not isinstance(start_turn, int):
            continue
        valid_existing_turns.add(start_turn)
        existing_task_payload.append(
            {
                "task_name": task.get("task_name", ""),
                "status": task.get("status", ""),
                "start_turn": start_turn,
            }
        )

    if not existing_task_payload:
        return []

    cycle_edges = []
    seen_edges = set()
    for task in new_tasks:
        from_turn = task.get("start_turn")
        if not isinstance(from_turn, int):
            continue

        previous_tasks_str = [
            f"{t['task_name']}(turn {t['start_turn']})"
            for t in existing_task_payload
        ]
        user_content = f"new_task: \"{task.get('task_name', '')}\", previous_tasks: {previous_tasks_str}"
        messages = [
            {"role": "system", "content": CYCLE_DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw_response = call_ollama(messages)
        cleaned_response = strip_code_block(raw_response)
        parsed_response = json.loads(cleaned_response)

        is_cycle = parsed_response.get("is_cycle")
        cycle_target_turn = parsed_response.get("cycle_target_turn")

        if not isinstance(is_cycle, bool):
            print(f"[DEBUG] detect_cycle_edges raw response:\n{raw_response}")
            raise ValueError("`is_cycle` field is missing or is not a boolean.")

        if is_cycle:
            if not isinstance(cycle_target_turn, int):
                raise ValueError(
                    "`cycle_target_turn` must be an integer when `is_cycle` is true."
                )
            if cycle_target_turn not in valid_existing_turns:
                raise ValueError(
                    f"`cycle_target_turn` {cycle_target_turn} does not match any existing task."
                )
            edge_key = (from_turn, cycle_target_turn)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            cycle_edges.append(
                {
                    "from_turn": from_turn,
                    "to_turn": cycle_target_turn,
                }
            )
        elif cycle_target_turn is not None:
            raise ValueError(
                "`cycle_target_turn` must be null when `is_cycle` is false."
            )

    return cycle_edges


def main(session_id=None) -> str:
    ensure_log_dir()

    if CONVERSATION_SNAPSHOT_PATH.exists():
        CONVERSATION_SNAPSHOT_PATH.unlink()

    session_id_arg = session_id
    if session_id_arg is None and len(sys.argv) > 1:
        session_id_arg = sys.argv[1]

    try:
        create_conversation_snapshot()
        records = load_conversation_records()
        session_id = select_session_id(records, session_id_arg)
        initial_turns_text = build_initial_turns_text(records, session_id)
        goal_records = load_goal_records()
        goals = load_session_goals(goal_records, session_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to prepare conversation: {exc}")
        sys.exit(1)

    if not goals and initial_turns_text:
        try:
            goals = extract_goals_from_initial_turns(initial_turns_text)
            save_goal_version(session_id, goals, auto_extracted=True)
        except error.URLError as exc:
            print(f"Ollama request failed during goal extraction: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"Ollama request failed during goal extraction: {exc}")
            sys.exit(1)

    existing_tasks, existing_cycle_edges, last_start_turn = load_existing_tasks(session_id)

    if existing_tasks:
        conversation_text = build_conversation_text_after_turn(records, session_id, last_start_turn)
    else:
        conversation_text = build_conversation_text(records, session_id)

    if not conversation_text.strip():
        print(f"No new turns to analyze for session {session_id}.")
        append_tasks_log({
            "session_id": session_id,
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "tasks": existing_tasks,
            "cycle_edges": existing_cycle_edges,
        })
        print(f"Saved analysis for session {session_id} to {TASKS_LOG_PATH}")
        return session_id

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conversation_text},
    ]

    print(f"[DEBUG] conversation_text (first 500 chars):\n{conversation_text[:500]}")

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
        "cycle_edges": existing_cycle_edges,
    }

    try:
        cleaned_response = strip_code_block(raw_response)
        parsed_response = json.loads(cleaned_response)
        tasks = parsed_response.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("`tasks` field is missing or is not a list.")
        session_records = [
            record for record in records if record.get("session_id") == session_id
        ]
        new_tasks = []
        for task in tasks:
            normalized_task = dict(task)
            normalized_task["deviation"] = False
            if goals:
                try:
                    normalized_task["deviation"] = evaluate_task_deviation(
                        normalized_task, goals, session_records
                    )
                except (json.JSONDecodeError, ValueError, error.URLError) as deviation_exc:
                    print(f"Failed to evaluate deviation for task '{normalized_task.get('task_name')}': {deviation_exc}")
            new_tasks.append(normalized_task)
        result["tasks"] = existing_tasks + new_tasks
        try:
            result["cycle_edges"] = existing_cycle_edges + detect_cycle_edges(
                new_tasks, existing_tasks, session_records
            )
        except (json.JSONDecodeError, ValueError, error.URLError) as cycle_exc:
            print(f"Failed to detect cycle edges: {cycle_exc}")
            result["cycle_edges"] = existing_cycle_edges
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to parse Ollama JSON response: {exc}")
        print(f"[DEBUG] Raw Ollama response:\n{raw_response}")
        result["raw_response"] = raw_response
        result["tasks"] = existing_tasks
        result["cycle_edges"] = existing_cycle_edges
    except error.URLError as exc:
        print(f"Ollama request failed during deviation/cycle analysis: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Ollama request failed during deviation/cycle analysis: {exc}")
        sys.exit(1)

    append_tasks_log(result)
    print(f"Saved analysis for session {session_id} to {TASKS_LOG_PATH}")

    if CONVERSATION_SNAPSHOT_PATH.exists():
        CONVERSATION_SNAPSHOT_PATH.unlink()

    return session_id


if __name__ == "__main__":
    main()
