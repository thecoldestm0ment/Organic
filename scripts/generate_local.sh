#!/usr/bin/env bash
# GPU 서버에서 config.yaml 의 로컬(localhost) 생성기를 하나씩 vLLM으로 띄우고, 생성하고, 내립니다.
#
#   bash scripts/generate_local.sh --pilot 20          # 모델마다 시험 생성 20개씩 (프롬프트 확인용)
#   bash scripts/generate_local.sh                     # 전체 생성 (끊겨도 다시 실행하면 이어서 함)
#   bash scripts/generate_local.sh exaone kanana       # 특정 생성기만
#   CUDA_VISIBLE_DEVICES=1 bash scripts/generate_local.sh   # 비어 있는 GPU 지정 (nvidia-smi 로 확인)
#
# 처음 실행할 때는 모델을 내려받느라 모델당 몇 분 더 걸립니다. 서버 로그는 logs/vllm_<이름>.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
PILOT=()
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot) PILOT=(--pilot "$2"); shift 2 ;;
    *) ONLY+=("$1"); shift ;;
  esac
done
mkdir -p logs

# name, model, port, serve args (tab separated) for every localhost generator
mapfile -t ROWS < <("$PY" - <<'EOF'
import yaml
from urllib.parse import urlparse
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
for g in cfg["generation"]["generators"]:
    u = urlparse(g["base_url"])
    if u.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        print(g["name"], g["model"], u.port or 8000, " ".join(map(str, g.get("serve_args", []))), sep="\t")
EOF
)

PID=""
cleanup() { if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then kill "$PID"; wait "$PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r NAME MODEL PORT ARGS <<< "${row%$'\r'}"   # strip CR (Python on Windows)
  if [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " =~ " ${NAME} " ]]; then continue; fi
  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "[오류] 포트 ${PORT}에 이미 서버가 떠 있습니다. 끄고 다시 실행하세요." >&2; exit 1
  fi
  echo "== ${NAME}: ${MODEL} 띄우는 중 (로그: logs/vllm_${NAME}.log)"
  # shellcheck disable=SC2086
  vllm serve "$MODEL" --port "$PORT" $ARGS > "logs/vllm_${NAME}.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 180); do   # 최대 30분 (첫 실행은 모델 다운로드)
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[오류] vLLM이 종료됐습니다. 마지막 로그:" >&2; tail -20 "logs/vllm_${NAME}.log" >&2; exit 1
    fi
    sleep 10
  done
  "$PY" -m beta.generate --only "$NAME" ${PILOT[@]+"${PILOT[@]}"}
  cleanup; PID=""
  sleep 5   # GPU 메모리 반환 대기
done
echo "완료."
