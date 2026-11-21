#!/usr/bin/env bash
# llama.cpp 의 llama-server 로 config.yaml 의 로컬 생성기를 하나씩 띄우고, 생성하고, 내립니다.
# vLLM 을 못 쓰는 서버(ppc64le, V100 등)용. GGUF 파일이 없으면 Hugging Face 에서 받습니다.
#
#   bash scripts/generate_llamacpp.sh --pilot 20          # 모델마다 시험 생성
#   bash scripts/generate_llamacpp.sh                     # 전체 생성 (끊겨도 다시 실행하면 이어서 함)
#   bash scripts/generate_llamacpp.sh exaone kanana       # 특정 생성기만
#
# 환경 변수
#   CUDA_VISIBLE_DEVICES  쓸 GPU (nvidia-smi 로 빈 GPU 확인)
#   LLAMA_SERVER          llama-server 경로 (기본: PATH 또는 ~/llama.cpp/build/bin/llama-server)
#   MODELS_DIR            GGUF 저장 위치 (기본: ~/models/gguf)
#   NP                    동시 처리 슬롯 수 (기본 8),  CTX_PER_SLOT  슬롯당 문맥 길이 (기본 2048)
#   HF_ENDPOINT           Hugging Face 주소 (기본 https://huggingface.co)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LLAMA_SERVER=${LLAMA_SERVER:-$(command -v llama-server || echo "$HOME/llama.cpp/build/bin/llama-server")}
MODELS_DIR=${MODELS_DIR:-$HOME/models/gguf}
NP=${NP:-8}
CTX_PER_SLOT=${CTX_PER_SLOT:-2048}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

PILOT=()
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot) PILOT=(--pilot "$2"); shift 2 ;;
    *) ONLY+=("$1"); shift ;;
  esac
done

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "[오류] llama-server 를 찾을 수 없습니다: $LLAMA_SERVER  (LLAMA_SERVER=경로 로 지정하세요)" >&2; exit 1
fi
EXTRA=()
if "$LLAMA_SERVER" --help 2>&1 | grep -q -- "--jinja"; then EXTRA+=(--jinja); fi  # use the model's own chat template
mkdir -p logs "$MODELS_DIR"

# name, model, port, repo, files (tab separated) for every localhost generator that has a gguf entry
mapfile -t ROWS < <("$PY" - <<'EOF'
import yaml
from urllib.parse import urlparse
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
for g in cfg["generation"]["generators"]:
    u = urlparse(g["base_url"])
    if u.hostname in ("localhost", "127.0.0.1", "0.0.0.0") and g.get("gguf"):
        print(g["name"], g["model"], u.port or 8000, g["gguf"]["repo"], " ".join(g["gguf"]["files"]), sep="\t")
EOF
)

PID=""
cleanup() { if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then kill "$PID"; wait "$PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r NAME MODEL PORT REPO FILES <<< "${row%$'\r'}"   # strip CR (Python on Windows)
  if [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " =~ " ${NAME} " ]]; then continue; fi

  DIR="$MODELS_DIR/${REPO//\//__}"
  mkdir -p "$DIR"
  FIRST=""
  for f in $FILES; do
    [[ -z "$FIRST" ]] && FIRST="$DIR/$f"
    if [[ ! -s "$DIR/$f" ]]; then
      echo "== ${NAME}: ${REPO}/${f} 내려받는 중"
      curl -L --fail --retry 3 -C - -o "$DIR/$f.part" "$HF_ENDPOINT/$REPO/resolve/main/$f"
      mv "$DIR/$f.part" "$DIR/$f"
    fi
  done

  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "[오류] 포트 ${PORT}에 이미 서버가 떠 있습니다. 끄고 다시 실행하세요." >&2; exit 1
  fi
  echo "== ${NAME}: llama-server 띄우는 중 (GPU ${CUDA_VISIBLE_DEVICES:-전체}, 포트 ${PORT}, 로그 logs/llama_${NAME}.log)"
  "$LLAMA_SERVER" -m "$FIRST" --alias "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -ngl 999 -np "$NP" -c $((NP * CTX_PER_SLOT)) ${EXTRA[@]+"${EXTRA[@]}"} \
    > "logs/llama_${NAME}.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 120); do   # 최대 10분 (모델 로딩)
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[오류] llama-server 가 종료됐습니다. 마지막 로그:" >&2; tail -20 "logs/llama_${NAME}.log" >&2; exit 1
    fi
    sleep 5
  done
  "$PY" -m beta.generate --only "$NAME" ${PILOT[@]+"${PILOT[@]}"}
  cleanup; PID=""
  sleep 3   # GPU 메모리 반환 대기
done
echo "완료."
