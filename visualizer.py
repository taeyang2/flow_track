import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
GRAPH_PATH = LOG_DIR / "graph.json"
HTML_PATH = LOG_DIR / "graph.html"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"
STATUS_CLASSES = {
    "completed": "completed",
    "in_progress": "inProgress",
    "abandoned": "abandoned",
}


def load_graph():
    if not GRAPH_PATH.is_file():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    with GRAPH_PATH.open("r", encoding="utf-8") as graph_file:
        try:
            return json.load(graph_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in graph file: {exc}") from exc


def validate_session_id(graph, session_id=None):
    graph_session_id = graph.get("session_id")
    if not graph_session_id:
        raise ValueError("`session_id` is missing in graph.json.")

    if session_id and session_id != graph_session_id:
        raise ValueError(
            f"Requested session_id {session_id} does not match graph session {graph_session_id}."
        )

    return graph_session_id


def escape_mermaid_label(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_mermaid(graph):
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    lines = ["flowchart TD"]

    for node in nodes:
        node_id = node["id"]
        task_name = node.get("task_name", "")
        if node.get("deviation", False):
            task_name = f"⚠ {task_name}"
        label = escape_mermaid_label(task_name)
        lines.append(f'  {node_id}["{label}"]')

    for edge in edges:
        lines.append(f'  {edge["from"]} --> {edge["to"]}')

    lines.append("  classDef completed fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;")
    lines.append("  classDef inProgress fill:#fff3bf,stroke:#f9a825,stroke-width:2px,color:#8a6d00;")
    lines.append("  classDef abandoned fill:#e0e0e0,stroke:#757575,stroke-width:2px,color:#424242;")
    lines.append("  classDef deviation fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#8e0000;")

    for node in nodes:
        if node.get("deviation", False):
            class_name = "deviation"
        else:
            status = node.get("status", "")
            class_name = STATUS_CLASSES.get(status)
        if class_name:
            lines.append(f'  class {node["id"]} {class_name};')

    return "\n".join(lines)


def build_html(graph, mermaid_text):
    session_id = graph.get("session_id", "")
    created_at = graph.get("created_at", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Flow Track Graph</title>
  <script src="{MERMAID_CDN}"></script>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: "base",
      themeVariables: {{
        primaryColor: "#ffffff",
        primaryTextColor: "#222222",
        primaryBorderColor: "#999999",
        lineColor: "#666666",
        fontFamily: "Arial, sans-serif"
      }}
    }});
  </script>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: Arial, sans-serif;
      background: #f7f7f7;
      color: #222;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .meta {{
      margin-bottom: 24px;
      color: #555;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid #ddd;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .graph-card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 16px;
      padding: 24px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <h1>Task Graph</h1>
  <div class="meta">session_id: {session_id} | created_at: {created_at}</div>
  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background:#c8e6c9;"></span>completed</div>
    <div class="legend-item"><span class="swatch" style="background:#fff3bf;"></span>in_progress</div>
    <div class="legend-item"><span class="swatch" style="background:#e0e0e0;"></span>abandoned</div>
    <div class="legend-item"><span class="swatch" style="background:#ffcdd2;"></span>deviation</div>
  </div>
  <div class="graph-card">
    <pre class="mermaid">
{mermaid_text}
    </pre>
  </div>
</body>
</html>
"""


def save_html(html):
    with HTML_PATH.open("w", encoding="utf-8") as html_file:
        html_file.write(html)


def main(session_id=None, print_mermaid=False) -> str:
    session_id_arg = session_id
    if session_id_arg is None and len(sys.argv) > 1:
        session_id_arg = sys.argv[1]

    try:
        graph = load_graph()
        session_id = validate_session_id(graph, session_id_arg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to prepare visualization: {exc}")
        sys.exit(1)

    mermaid_text = build_mermaid(graph)
    if print_mermaid:
        print(mermaid_text)

    html = build_html(graph, mermaid_text)
    save_html(html)
    print(f"Saved HTML visualization to {HTML_PATH}")
    return session_id


if __name__ == "__main__":
    main(print_mermaid=True)
