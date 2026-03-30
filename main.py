from pathlib import Path

import analyzer
import cli_chat
import graph_builder
import visualizer


HTML_PATH = Path.home() / "toy" / "flow_track" / "logs" / "graph.html"


def main() -> None:
    session_id = cli_chat.main()

    print("분석 중...")
    analyzer.main(session_id)

    print("그래프 변환 중...")
    graph_builder.main(session_id)

    print("시각화 생성 중...")
    visualizer.main(session_id)

    print(f"완료! {HTML_PATH} 을 브라우저에서 확인하세요.")


if __name__ == "__main__":
    main()
