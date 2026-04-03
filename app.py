# 실행: streamlit run app.py --server.port 8501
# 로컬 포트포워딩: ssh -L 8501:localhost:8501 tylee@서버주소

from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from cli_chat import append_log, call_llm
from analyzer import (
    get_active_goal_record,
    get_session_goal_history,
    load_goal_records,
    main as analyzer_main,
    restore_previous_goal_version,
    save_goal_version,
)
import graph_builder
import visualizer


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "logs" / "conversation_log.jsonl"
GRAPH_HTML_PATH = BASE_DIR / "logs" / "graph.html"


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.session_state.goals = []
        st.session_state.turn = 1
        st.session_state.pipeline_notice = False
        st.session_state.prev_goal_input = ""
        if GRAPH_HTML_PATH.exists():
            GRAPH_HTML_PATH.unlink()

    if "goal_input" not in st.session_state:
        st.session_state.goal_input = ""
    if "tracking_enabled" not in st.session_state:
        st.session_state.tracking_enabled = True
    if "show_goal_editor" not in st.session_state:
        st.session_state.show_goal_editor = False
    if "goal_edit_input" not in st.session_state:
        st.session_state.goal_edit_input = ""


def ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def parse_goals(raw_text: str) -> list[str]:
    return [goal.strip() for goal in raw_text.split(",") if goal.strip()]


def sync_active_goals() -> tuple[list[dict], dict | None]:
    goal_records = load_goal_records()
    goal_history = get_session_goal_history(goal_records, st.session_state.session_id)
    active_goal_record = get_active_goal_record(goal_records, st.session_state.session_id)
    if active_goal_record:
        st.session_state.goals = active_goal_record.get("goals", [])
        st.session_state.goal_input = ", ".join(st.session_state.goals)
        st.session_state.prev_goal_input = st.session_state.goal_input
    return goal_history, active_goal_record


def format_goal_type(record: dict | None) -> str:
    if not record:
        return "-"
    version = record.get("version", 1)
    source = "자동 감지" if record.get("auto_extracted") else "직접 입력"
    return f"v{version} · {source}"


def format_goal_timestamp(timestamp: str) -> str:
    if not timestamp:
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp


def render_global_css() -> None:
    st.markdown(
        """
<style>

div.stButton > button {
    background-color: #534AB7;
    color: white;
    border: none;
}
div.stButton > button:hover {
    background-color: #3C3489;
    color: white;
}
div.stButton > button:disabled {
    background-color: #AFA9EC;
    color: white;
    border: none;
}

section.main > div.block-container {
    max-width: 1200px;
    padding: 2rem 2rem 0 2rem;
    margin: 0 auto;
}

div[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08),
                0 1px 4px rgba(0,0,0,0.04) !important;
    padding: 4px 8px !important;
}

div[data-testid="stChatInput"] > div {
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stChatInput"]:focus-within {
    border: 1px solid #e0e0e0 !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08),
                0 1px 4px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

div[data-testid="stChatInput"] > div:focus,
div[data-testid="stChatInput"] > div:focus-visible,
div[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    font-size: 15px !important;
    min-height: 24px !important;
    max-height: 24px !important;
    padding: 6px 8px !important;
    resize: none !important;
    overflow: hidden !important;
}

div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInput"] textarea:focus-visible,
div[data-testid="stChatInput"] textarea:active {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stChatInput"] button {
    background-color: transparent !important;
    border: none !important;
    color: #888 !important;
}
div[data-testid="stChatInput"] button:hover {
    color: #534AB7 !important;
}

</style>
""",
        unsafe_allow_html=True,
    )


def render_sidebar(goal_history: list[dict], active_goal_record: dict | None) -> None:
    with st.sidebar:
        has_goal = bool(st.session_state.get("goals"))

        st.title("flow_track")
        st.text_input(
            "목표",
            placeholder="목표를 입력하세요 (선택사항)",
            key="goal_input",
        )
        st.toggle("목표 추적", value=True, key="tracking_enabled")

        st.subheader("현재 목표")
        if has_goal:
            st.write(", ".join(st.session_state.goals))
            st.caption(format_goal_type(active_goal_record))
        else:
            st.write("(목표 없음)")

        st.subheader("목표 변경 이력")
        if goal_history:
            for record in reversed(goal_history):
                version = record.get("version", "?")
                goals_text = ", ".join(record.get("goals", [])) or "(없음)"
                source = "자동 감지" if record.get("auto_extracted") else "직접 입력"
                current = " · 현재" if record.get("superseded") is False else ""
                st.write(f"v{version} · {format_goal_timestamp(record.get('timestamp', ''))}")
                st.caption(f"{goals_text} | {source}{current}")
        else:
            st.write("이력이 없습니다.")

        if st.button("목표 수정", use_container_width=True, disabled=not has_goal):
            st.session_state.show_goal_editor = not st.session_state.show_goal_editor

        if st.session_state.show_goal_editor:
            st.session_state.goal_edit_input = st.text_input(
                "새 목표 입력",
                value=st.session_state.goal_edit_input,
                placeholder="예: 대시보드 만들기, 그래프 검증",
            )
            if st.button("목표 저장", use_container_width=True):
                new_goals = parse_goals(st.session_state.goal_edit_input)
                if new_goals:
                    saved_record = save_goal_version(
                        st.session_state.session_id,
                        new_goals,
                        auto_extracted=False,
                    )
                    st.session_state.goals = saved_record.get("goals", [])
                    st.session_state.goal_input = ", ".join(st.session_state.goals)
                    st.session_state.prev_goal_input = st.session_state.goal_input
                    st.session_state.goal_edit_input = ""
                    st.session_state.show_goal_editor = False
                    st.rerun()
                else:
                    st.warning("새 목표를 입력하세요.")

        if st.button("목표 되돌리기", use_container_width=True, disabled=not has_goal):
            restored_record = restore_previous_goal_version(st.session_state.session_id)
            if restored_record is None:
                st.warning("복원할 이전 목표가 없습니다.")
            else:
                st.session_state.goals = restored_record.get("goals", [])
                st.session_state.goal_input = ", ".join(st.session_state.goals)
                st.session_state.prev_goal_input = st.session_state.goal_input
                st.rerun()


def handle_goal_input_change() -> None:
    current_input = st.session_state.get("goal_input", "").strip()
    prev_input = st.session_state.get("prev_goal_input", "").strip()

    if current_input and current_input != prev_input:
        goals = parse_goals(current_input)
        save_goal_version(st.session_state.session_id, goals, auto_extracted=False)
        st.session_state.goals = goals
        st.session_state.prev_goal_input = current_input
        st.rerun()


def run_pipeline(session_id: str) -> None:
    analyzer_main(session_id)
    graph_builder.main(session_id)
    visualizer.main(session_id)


def build_responsive_graph_html(graph_html: str) -> str:
    resize_script = """
<script>
(function () {
  let animationFrameId = null;

  function resizeFrame() {
    const frame = window.frameElement;
    if (!frame) {
      return;
    }
    const parentWindow = window.parent || window;
    const parentDocument = parentWindow.document;
    const viewportHeight = parentWindow.innerHeight || window.innerHeight;
    const frameRect = frame.getBoundingClientRect();
    const bottomGap = 24;
    const nextHeight = Math.max(520, viewportHeight - frameRect.top - bottomGap);

    frame.style.height = nextHeight + "px";
    frame.style.width = "100%";

    const body = parentDocument.body;
    if (body) {
      body.style.overflowAnchor = "none";
    }
  }

  function scheduleResize() {
    if (animationFrameId !== null) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(() => {
      animationFrameId = null;
      resizeFrame();
    });
  }

  window.addEventListener("load", scheduleResize);
  window.addEventListener("resize", scheduleResize);
  if (window.parent) {
    window.parent.addEventListener("resize", scheduleResize);
  }

  const parentDocument = window.parent ? window.parent.document : document;
  if (parentDocument && "ResizeObserver" in window) {
    const observer = new ResizeObserver(scheduleResize);
    if (parentDocument.body) {
      observer.observe(parentDocument.body);
    }
    if (parentDocument.documentElement) {
      observer.observe(parentDocument.documentElement);
    }
  }

  scheduleResize();
})();
</script>
"""
    if "</body>" in graph_html:
        return graph_html.replace("</body>", resize_script + "\n</body>")
    return graph_html + resize_script


def render_chat_column() -> None:
    st.markdown('<div id="flowtrack-chat-anchor"></div>', unsafe_allow_html=True)
    st.subheader("채팅")

    if not st.session_state.display_messages:
        st.markdown(
            """
    <div style="
        height: 45vh;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <h2 style="font-size: 26px; font-weight: 500; color: inherit;">
            무슨 작업을 하고 계세요?
        </h2>
    </div>
    """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["time"])

    if st.session_state.pipeline_notice:
        st.success("워크플로우가 업데이트됐습니다.")
        st.session_state.pipeline_notice = False

    user_input = st.chat_input("메시지를 입력하세요...")
    if not user_input:
        return

    now_text = datetime.now().strftime("%H:%M:%S")
    st.session_state.display_messages.append(
        {"role": "user", "content": user_input, "time": now_text}
    )
    st.session_state.messages.append({"role": "user", "content": user_input})
    append_log(st.session_state.session_id, st.session_state.turn, "user", user_input)

    assistant_reply = call_llm(st.session_state.messages)
    assistant_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.display_messages.append(
        {"role": "assistant", "content": assistant_reply, "time": assistant_time}
    )
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    append_log(
        st.session_state.session_id,
        st.session_state.turn,
        "assistant",
        assistant_reply,
    )

    if st.session_state.tracking_enabled and st.session_state.turn % 5 == 0:
        run_pipeline(st.session_state.session_id)
        st.session_state.pipeline_notice = True

    st.session_state.turn += 1
    st.rerun()


def render_graph_column() -> None:
    st.markdown('<div id="flowtrack-graph-anchor"></div>', unsafe_allow_html=True)
    st.markdown("**워크플로우 그래프**")
    st.caption("3초 갱신")

    if not st.session_state.get("tracking_enabled", True):
        st.info("목표 추적이 비활성화됐습니다.")
    else:
        if GRAPH_HTML_PATH.exists():
            graph_html = GRAPH_HTML_PATH.read_text(encoding="utf-8")
            responsive_html = build_responsive_graph_html(graph_html)
            components.html(responsive_html, height=700, scrolling=True)
        else:
            st.info("아직 그래프가 없습니다. 대화를 시작하세요.")

    st_autorefresh(interval=3000, key="graph_refresh")


def render_resizable_columns_script() -> None:
    components.html(
        """
<script>
(function () {
  const STORAGE_KEY = "flowtrack_split_ratio";
  const HANDLE_ID = "flowtrack-column-resizer";
  const MIN_RATIO = 0.35;
  const MAX_RATIO = 0.75;
  const DEFAULT_RATIO = 1.2 / (1.2 + 1);

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function getColumns() {
    const parentDoc = window.parent.document;
    const chatAnchor = parentDoc.getElementById("flowtrack-chat-anchor");
    const graphAnchor = parentDoc.getElementById("flowtrack-graph-anchor");
    if (!chatAnchor || !graphAnchor) {
      return null;
    }

    const chatColumn = chatAnchor.closest('[data-testid="column"]');
    const graphColumn = graphAnchor.closest('[data-testid="column"]');
    if (!chatColumn || !graphColumn || !chatColumn.parentElement) {
      return null;
    }

    const wrapper = chatColumn.parentElement;
    if (wrapper !== graphColumn.parentElement) {
      return null;
    }

    return { parentDoc, wrapper, chatColumn, graphColumn };
  }

  function applyRatio(chatColumn, graphColumn, ratio) {
    const safeRatio = clamp(ratio, MIN_RATIO, MAX_RATIO);
    const graphRatio = 1 - safeRatio;

    chatColumn.style.flex = `0 0 calc(${safeRatio * 100}% - 6px)`;
    chatColumn.style.width = `calc(${safeRatio * 100}% - 6px)`;
    chatColumn.style.maxWidth = `calc(${safeRatio * 100}% - 6px)`;
    chatColumn.style.minWidth = "320px";

    graphColumn.style.flex = `0 0 calc(${graphRatio * 100}% - 6px)`;
    graphColumn.style.width = `calc(${graphRatio * 100}% - 6px)`;
    graphColumn.style.maxWidth = `calc(${graphRatio * 100}% - 6px)`;
    graphColumn.style.minWidth = "280px";

    window.parent.localStorage.setItem(STORAGE_KEY, String(safeRatio));
  }

  function ensureHandle(wrapper, graphColumn) {
    let handle = wrapper.querySelector("#" + HANDLE_ID);
    if (handle) {
      return handle;
    }

    handle = window.parent.document.createElement("div");
    handle.id = HANDLE_ID;
    handle.setAttribute("title", "드래그해서 패널 크기 조정");
    handle.style.flex = "0 0 12px";
    handle.style.width = "12px";
    handle.style.minWidth = "12px";
    handle.style.cursor = "col-resize";
    handle.style.position = "relative";
    handle.style.alignSelf = "stretch";
    handle.style.background = "transparent";
    handle.style.zIndex = "10";
    handle.innerHTML = '<div style="position:absolute;left:5px;top:0;bottom:0;width:2px;background:#d6d2fb;border-radius:999px;"></div>';

    wrapper.insertBefore(handle, graphColumn);
    return handle;
  }

  function boot() {
    const result = getColumns();
    if (!result) {
      window.setTimeout(boot, 150);
      return;
    }

    const { wrapper, chatColumn, graphColumn } = result;
    const savedRatio = Number.parseFloat(
      window.parent.localStorage.getItem(STORAGE_KEY) || String(DEFAULT_RATIO)
    );
    applyRatio(chatColumn, graphColumn, Number.isNaN(savedRatio) ? DEFAULT_RATIO : savedRatio);

    const handle = ensureHandle(wrapper, graphColumn);
    if (handle.dataset.bound === "true") {
      return;
    }
    handle.dataset.bound = "true";

    let dragging = false;

    const onMouseMove = (event) => {
      if (!dragging) {
        return;
      }
      const rect = wrapper.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / rect.width;
      applyRatio(chatColumn, graphColumn, ratio);
    };

    const stopDragging = () => {
      dragging = false;
      window.parent.document.body.style.userSelect = "";
      window.parent.document.body.style.cursor = "";
    };

    handle.addEventListener("mousedown", () => {
      dragging = true;
      window.parent.document.body.style.userSelect = "none";
      window.parent.document.body.style.cursor = "col-resize";
    });

    window.parent.addEventListener("mousemove", onMouseMove);
    window.parent.addEventListener("mouseup", stopDragging);
  }

  boot();
})();
</script>
        """,
        height=0,
    )


def main() -> None:
    st.set_page_config(layout="wide")
    ensure_log_dir()
    init_session_state()
    render_global_css()
    goal_history, active_goal_record = sync_active_goals()
    render_sidebar(goal_history, active_goal_record)
    handle_goal_input_change()

    col_chat, col_graph = st.columns([1.2, 1])
    with col_chat:
        render_chat_column()
    with col_graph:
        render_graph_column()
    render_resizable_columns_script()


if __name__ == "__main__":
    main()
