# Flow Track

## 프로젝트 소개

Flow Track은 업무 대화를 분석해 워크플로우를 시각화하는 도구입니다. 대화에서 수행한 태스크를 추출하고, 목표 대비 흐름 이탈을 감지해 메타인지와 작업 회고를 지원합니다.

## 시스템 구조

- `cli_chat.py`: OpenAI API 기반 CLI 채팅
- `analyzer.py`: Ollama 기반 태스크 추출 및 이탈 감지
- `graph_builder.py`: DAG 그래프 변환
- `visualizer.py`: Mermaid 기반 HTML 시각화
- `main.py`: 전체 파이프라인 오케스트레이션

## 실행 환경

- Python 3.10+
- Ollama (`gemma3:12b`)
- OpenAI API 키

## 설치 및 실행 방법

1. 환경변수를 설정합니다.

```bash
cp .env_secrets .env
```

`.env` 또는 `.env_secrets`에는 최소한 아래 값이 필요합니다.

```env
OPENAI_API_KEY=your_api_key_here
```

2. Ollama 서버를 실행합니다.

```bash
ollama serve
```

필요하면 모델도 미리 받아둡니다.

```bash
ollama pull gemma3:12b
```

3. 파이프라인을 실행합니다.

```bash
python3 main.py
```

## 로그 파일 구조

- `logs/conversation_log.jsonl`: 세션별 대화 로그
- `logs/goal.jsonl`: 세션 목표 및 자동 추출 목표 로그
- `logs/tasks.jsonl`: 태스크 추출 결과와 `deviation` 플래그
- `logs/graph.json`: 그래프 구성용 중간 산출물
- `logs/graph.html`: 최종 시각화 결과

## 다음 개발 계획

- 프롬프트 튜닝: 맥락적 이탈 감지 개선
- 실시간 분석
- 웹 대시보드 전환
- DCG 확장: 순환 구조 지원
