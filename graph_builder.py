import json
import sys
from datetime import datetime
from pathlib import Path


ALLOW_CYCLES = False
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
TASKS_LOG_PATH = LOG_DIR / "tasks.jsonl"
GRAPH_PATH = LOG_DIR / "graph.json"


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


def select_session_id(records, session_id=None):
    session_ids = [record.get("session_id") for record in records if record.get("session_id")]
    if not session_ids:
        raise ValueError("No session_id found in tasks log.")

    if session_id:
        if session_id not in session_ids:
            raise ValueError(f"Session not found: {session_id}")
        return session_id

    return max(session_ids)


def select_latest_session_record(records, session_id):
    session_records = [record for record in records if record.get("session_id") == session_id]
    if not session_records:
        raise ValueError(f"No task records found for session: {session_id}")

    session_records.sort(key=lambda item: item.get("analyzed_at", ""))
    return session_records[-1]


def build_nodes(tasks):
    sorted_tasks = sorted(
        tasks,
        key=lambda item: (
            item.get("start_turn", 0),
            item.get("task_name", ""),
            item.get("status", ""),
        ),
    )

    nodes = []
    for index, task in enumerate(sorted_tasks, start=1):
        nodes.append(
            {
                "id": index,
                "task_name": task.get("task_name", ""),
                "status": task.get("status", ""),
                "start_turn": task.get("start_turn", 0),
                "deviation": task.get("deviation", False),
            }
        )

    return nodes


def build_edges(nodes):
    grouped_nodes = {}
    for node in nodes:
        grouped_nodes.setdefault(node["start_turn"], []).append(node)

    turns = sorted(grouped_nodes)
    edges = []

    for index in range(len(turns) - 1):
        current_turn_nodes = grouped_nodes[turns[index]]
        next_turn_nodes = grouped_nodes[turns[index + 1]]

        for current_node in current_turn_nodes:
            for next_node in next_turn_nodes:
                edges.append(
                    {
                        "from": current_node["id"],
                        "to": next_node["id"],
                        "type": "next",
                    }
                )

    return edges


def save_graph(graph):
    with GRAPH_PATH.open("w", encoding="utf-8") as graph_file:
        json.dump(graph, graph_file, ensure_ascii=False, indent=2)


def main(session_id=None) -> str:
    session_id_arg = session_id
    if session_id_arg is None and len(sys.argv) > 1:
        session_id_arg = sys.argv[1]

    try:
        records = load_task_records()
        session_id = select_session_id(records, session_id_arg)
        session_record = select_latest_session_record(records, session_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to prepare graph data: {exc}")
        sys.exit(1)

    tasks = session_record.get("tasks", [])
    if not isinstance(tasks, list):
        print("Failed to prepare graph data: `tasks` field is not a list.")
        sys.exit(1)

    nodes = build_nodes(tasks)
    edges = build_edges(nodes)

    graph = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "allow_cycles": ALLOW_CYCLES,
        "nodes": nodes,
        "edges": edges,
    }

    save_graph(graph)
    print(f"Saved graph for session {session_id} to {GRAPH_PATH}")
    return session_id


if __name__ == "__main__":
    main()
