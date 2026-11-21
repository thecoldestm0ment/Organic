# LLM 생성 한국어 음식점 리뷰 탐지 — 3일 베타

사람 리뷰(KR3 원본, 2022년 3월 이전 공개)와 여러 LLM이 생성한 리뷰를 구분할 수 있는지,
그게 겉모습(구두점·띄어쓰기) 때문만은 아닌지, 학습에 안 쓴 LLM에도 통하는지 확인합니다.

## 생성 모델 5개

| 생성기 | 모델 | 계열 | 실행 | 라이선스 |
|---|---|---|---|---|
| qwen | Qwen2.5-7B-Instruct (공식 GGUF Q8_0) | Alibaba | GPU 서버 | Apache-2.0 |
| exaone | EXAONE-3.5-7.8B-Instruct (공식 GGUF Q8_0) | LG AI연구원 | GPU 서버 | EXAONE (비상업) |
| kanana | kanana-1.5-8b-instruct-2505 (DevQuasar 변환 Q8_0) | Kakao | GPU 서버 | Apache-2.0 |
| gpt | API (`config.yaml` 에서 모델 이름 지정) | OpenAI | 팀원 | - |
| gemini | API (`config.yaml` 에서 모델 이름 지정) | Google | 팀원 | - |

오픈 모델은 모두 Hugging Face 승인(gated) 없이 받을 수 있습니다. Kanana는 카카오 공식 GGUF가 없어서 카카오 원본을 변환한 파일을 씁니다 (보고서에 명시).
API 생성기를 빼거나 다른 API로 바꾸려면 **조건을 뽑기 전에** `config.yaml` 을 고치세요.

### 여러 사람이 나눠서 생성할 때 (서버는 오픈 모델, 팀원은 API)

같은 조건 파일을 써야 리뷰 ID가 섞이지 않습니다.

1. 한 사람(서버 담당)이 `python -m beta.prepare_human` 과 `python -m beta.generate --plan-only` 실행
2. `data/interim/` 의 `human.parquet`, `fewshot_pool.parquet`, `conditions.parquet` 세 파일을 팀원에게 전달.
   팀원은 같은 위치(`data/interim/`)에 넣음
3. 각자 자기 생성기만 생성: 서버 `bash scripts/generate_llamacpp.sh`, 팀원 `python -m beta.generate --only gpt,gemini`
4. 팀원의 `generated.jsonl` 을 받아 이어 붙이기:
   `cat generated_api.jsonl >> data/interim/generated.jsonl` (생성기마다 ID가 달라 겹치지 않음)
5. 서버 담당이 `build_dataset` → `export_benchmark`

## GPU 서버(ppc64le + V100)에서 돌리기

이 서버는 IBM POWER(ppc64le)에 V100이라 vLLM과 PyTorch pip 패키지를 쓸 수 없습니다.
그래서 **생성은 llama.cpp(llama-server)** 로 하고, Python 환경은 conda-forge 패키지로 만듭니다 (PyTorch 불필요).

```bash
# 서버 접속
ssh [서버명]

# 코드 받기
git clone https://github.com/thecoldestm0ment/Organic.git ~/Organic && cd ~/Organic

# Python 환경 (생성·데이터 처리·TF-IDF용, torch 없음)
conda create -n beta -c conda-forge python=3.11 git pandas pyarrow numpy scikit-learn pyyaml tqdm pytest openai "httpx<0.28" -y   # 구버전 openai + httpx 0.28 충돌 방지
conda activate beta
python -m pytest -q tests/test_units.py     # torch 없이 도는 테스트만

# 빈 GPU 지정 + llama-server 위치
nvidia-smi
export CUDA_VISIBLE_DEVICES=2
export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server
$LLAMA_SERVER --version

# tmux 안에서 데이터 + 생성
tmux new -s beta
python -m beta.prepare_human
python -m beta.generate --plan-only
bash scripts/generate_llamacpp.sh --pilot 20    # 모델마다 시험 생성 (처음엔 GGUF 다운로드)
bash scripts/generate_llamacpp.sh               # 전체 생성
python -m beta.build_dataset
python -m beta.export_benchmark
```

- `generate_llamacpp.sh` 는 모델마다 GGUF 다운로드(없을 때만) → `llama-server` 실행 → 생성 → 종료를 반복합니다. 로그는 `logs/llama_<이름>.log`.
- GGUF는 `~/models/gguf` 에 저장됩니다 (`MODELS_DIR` 로 변경). Q8_0 기준 모델당 약 8GB.
- 동시 처리 슬롯은 `NP`(기본 8)로 조절합니다. V100 16GB에서 7~8B Q8_0 + 슬롯 8개는 여유 있게 들어갑니다.
- EXAONE 로딩이 `unknown model architecture` 로 실패하면 llama.cpp가 오래된 것이니 업데이트 후 다시 빌드하세요.
- 서버에 떠 있는 모델 이름이 config의 model과 다르면 생성을 거부합니다 (다른 계열로 라벨이 붙는 사고 방지).
- GPU 두 장에 나눠 동시에 돌리려면 조건을 먼저 만든 뒤(`--plan-only`) tmux 창 두 개에서
  `CUDA_VISIBLE_DEVICES=2 bash scripts/generate_llamacpp.sh qwen exaone` 와
  `CUDA_VISIBLE_DEVICES=3 bash scripts/generate_llamacpp.sh kanana` 를 실행하세요 (포트가 생성기마다 달라 겹치지 않음).
- KcELECTRA·임베딩·zero-shot 학습은 PyTorch가 필요해서 이 서버에서는 바로 안 됩니다. Colab이나 다른 x86 GPU에서 `benchmark/` 로 돌리세요.

x86 + compute capability 7.5 이상 GPU(T4, A100, L4 등)라면 vLLM 스크립트 `bash scripts/generate_local.sh` 도 쓸 수 있습니다.

## 로컬 설치 (GPU 없이 데이터 처리·TF-IDF만)

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q               # 오프라인 점검, 1분 이내 (가짜 생성기·소형 랜덤 모델 사용)
```

Colab에서는 이 폴더를 올리고 `pip install -r requirements.txt` 후 `!python -m beta.train_electra` 처럼 실행합니다.
학습 단계는 `data/processed/dataset.parquet` 만 있으면 됩니다.

## 스크립트

| 단계 | 명령 | 결과물 |
|---|---|---|
| 사람 리뷰 | `python -m beta.prepare_human` | KR3 원본(218MB) 다운로드, 길이 구간별 5,000건 + few-shot 풀 2,000건 |
| 생성 조건 | `python -m beta.generate --plan-only` | 리뷰마다 별점·카테고리·목표 길이·few-shot 예시 (`conditions.parquet`) |
| 시험 생성 | `bash scripts/generate_llamacpp.sh --pilot 20` | 생성기×프롬프트마다 20개 출력. 눈으로 확인 |
| 전체 생성 | `bash scripts/generate_llamacpp.sh` | `data/interim/generated.jsonl` (끊겨도 다시 실행하면 이어서 함). API 모델은 `python -m beta.generate --only gpt,gemini` |
| 데이터셋 | `python -m beta.build_dataset` | 정규화, 길이 매칭, 분할, **테스트셋 고정** |
| 팀 공유 | `python -m beta.export_benchmark` | `benchmark/` (아래 "팀 진행 방식") |
| TF-IDF | `python -m beta.tfidf_baseline` | 전 실험 + 상위 특징 `top_features.csv` |
| 임베딩 | `python -m beta.embed_baseline --preset e5-base` | 임베딩 + LR (프리셋: bge-m3, e5-base, e5-large, ko-sroberta, kr-sbert, kcelectra-frozen) |
| KcELECTRA | `python -m beta.train_electra` | main + 생성기 hold-out (GPU) |
| zero-shot | `python -m beta.zeroshot` | Fast-DetectGPT, log-likelihood (GPU, 선택) |
| 팀원 채점 | `python -m beta.score_predictions --model 이름 --dir predictions/이름` | 같은 규칙으로 채점 |
| 보고서 | `python -m beta.report` | `results/REPORT.md`, 진행 여부 판정, 오류 사례 CSV |

## 실험 정의 (모든 모델 공통)

| 실험 | 내용 |
|---|---|
| `main` | 전체 생성기·프롬프트로 학습, 테스트셋 평가 |
| `ablation_surface` | 같은 분할, 구두점·이모지·자모(ㅋㅠ) 제거 → 겉모습 의존도 확인 |
| `ablation_nospace` | 띄어쓰기까지 제거 |
| `logo_<계열>` | 그 생성기 계열을 학습·검증에서 빼고, 그 계열 리뷰 전부 + 사람 테스트셋으로 평가 (**미학습 LLM**) |
| `lopo_<프롬프트>` | 그 프롬프트 유형을 빼고 평가 (미학습 프롬프트) |

임계값은 항상 검증셋에서 FPR 1%/5%가 되도록 정한 뒤 테스트에 그대로 씁니다.
보고서의 "oracle" 값은 테스트셋에서 고른 임계값이라 낙관적입니다.

## 3일 일정

**0일차(오늘 밤)**
- 서버에 올려서 설치, `nvidia-smi` 로 GPU 메모리 확인 후 모델 크기 결정
- `python -m pytest -q`, `python -m beta.prepare_human`

**1일차 — 데이터**
1. `bash scripts/generate_llamacpp.sh --pilot 20` 출력을 읽고 확인: 마크다운, "[가게 이름]", "물론입니다!" 같은 서두, 목표 길이 대비 실제 길이
2. 프롬프트 문구(`beta/prompts.py`)를 고쳤으면 시험 생성만 다시 돌리면 됩니다.
   생성기 목록이나 `prompt_types` 를 바꿨으면 `python -m beta.generate --replan --plan-only` 로 조건을 다시 뽑으세요.
   **전체 생성을 시작한 뒤에는 프롬프트를 바꾸지 마세요** (버전이 섞입니다).
3. `bash scripts/generate_llamacpp.sh` (팀원 API: `python -m beta.generate --only gpt,gemini`) → `python -m beta.build_dataset` → `python -m beta.export_benchmark`
4. `build_dataset` 출력의 길이 구간별 매칭 표에서 `<=20` 구간이 너무 적으면 보고서에 한계로 적기

**2일차 — 모델**
- `python -m beta.tfidf_baseline` 후 `results/tfidf_lr/main/top_features.csv` 확인.
  구두점·공백·ㅋ·특정 어미만 상위에 있으면 지름길 학습입니다.
  정규화(`beta/normalize.py`)를 고친다면 **상위 특징(학습셋 기준)을 보고 결정**하고, 테스트 점수를 보고 고치지 마세요.
- `python -m beta.train_electra`, `python -m beta.embed_baseline --preset ...`, (선택) `python -m beta.zeroshot`

**3일차 — 분석**
- `python -m beta.report` → `results/REPORT.md`
- `results/error_analysis_*.csv` 에서 오탐·미탐 사례 직접 읽기

## 팀 진행 방식 (각자 다른 방법, 같은 시험지)

1. 한 명이 생성까지 끝내고 `python -m beta.export_benchmark` 로 `benchmark/` 를 만들어 **비공개** 드라이브에 공유
   - `reviews.csv`(본문·정답·메타데이터), `splits.csv`(실험별 train/val/test), `llm_generated_all.csv`(생성 리뷰 전체, 참고용), `PROTOCOL.md`(규칙)
2. 각자 원하는 방법(임베딩, 미세조정, 특징 공학 등)으로 `splits.csv` 의 train만 써서 학습, val로만 튜닝
3. val·test 행 점수를 `predictions/<모델이름>/<실험이름>.csv` (`id,score`)로 저장
4. `python -m beta.score_predictions --model <모델이름> --dir predictions/<모델이름> --dataset benchmark/dataset.parquet`
5. 각자의 `results/<모델이름>/` 를 한 곳에 모아 `python -m beta.report` → 전원 점수가 한 보고서에

모델 이름에 본인 이름을 넣으세요(예: `bge-m3_lr_minsu`). 테스트셋 sha256이 같아야 같은 시험지입니다.

## 설계 메모

- **KR3 원본(kr3_raw) 사용**: 전처리본은 이모지·ㅋ·~를 지우고 맞춤법 검사기를 돌려서, 그대로 쓰면 "ㅋㅋ 있으면 사람"이라는 지름길이 생깁니다.
- **잘린 리뷰 제외**: KR3 일부 출처는 미리보기로 크롤링돼 약 200자에서 "..."로 잘려 있습니다(약 9,700건). LLM 리뷰에는 없는 신호라 제외합니다.
- **한국어 필터 (두 클래스 동일)**: 시험 생성에서 Qwen 출력의 약 25%에 중국어·영어가 섞였습니다. 한자·가나가 있거나 영어가 대부분인 글은 생성 단계에서 거절·재생성하고, 데이터셋을 만들 때 사람 리뷰에도 같은 기준을 적용합니다(KR3에서 약 1% 제외).
- **특수 토큰 제거**: 일부 GGUF는 `<|eot_id|>` 같은 대답 끝 토큰을 글자로 출력해서, 생성 단계에서 그 앞까지만 남깁니다.
- **정규화는 두 클래스에 동일하게**: NFC(NFKC는 ㅋ를 다른 문자로 바꿈), 마크다운·줄바꿈·따옴표 통일.
- **분포 맞춤**: 생성 조건(별점·카테고리·목표 길이)을 사람 샘플에서 가져오고, 최종 데이터는 길이 구간마다 사람과 LLM 수를 같게 맞춥니다.
- **few-shot 예시는 사람 데이터와 겹치지 않는 별도 풀**에서 뽑습니다.
- **테스트셋 고정**: `test_freeze.json` 과 다르게 다시 만들면 거부합니다(`--force` 필요).

## 자주 막히는 곳

- "포트 8000에 이미 서버가 떠 있습니다": 이전에 띄운 vLLM을 끄기 (`pkill -f "vllm serve"`, 본인 프로세스인지 확인)
- 429(요청 한도, API 모델): `config.yaml` 의 생성기별 `concurrency` 를 낮추기
- 연속 10회 오류로 생성기 중단: 서버 로그, 키, 모델 이름, 결제 한도 확인
- 일부 최신 모델은 temperature를 지원하지 않음: 해당 생성기에 `supports_temperature: false`
- Gemini 생각(thinking) 토큰이 많이 나올 때: `extra_body: {reasoning_effort: none}` 시도
- 생성기 목록을 바꿨으면 `--replan` (기존 생성 결과는 `.bak` 으로 옮겨짐)

## 베타 벤치마크 (`benchmark_22a67908.zip`)

팀 공통 시험지입니다. 압축을 풀면 `benchmark/` 폴더가 나오고, 규칙은 그 안의 `PROTOCOL.md` 에 있습니다.

- 사람 2,338 + LLM 2,338 (GPT: gpt-4.1-mini, Claude: claude-haiku-4-5, 프롬프트 A·C), 길이 구간별로 수를 맞춤
- 테스트셋 sha256 `22a67908...` (점수를 공유할 때 이 값이 같아야 같은 시험지)
- 생성기 hold-out 실험은 `logo_openai`, `logo_anthropic` 두 개
- TF-IDF 결과 보고서 `REPORT_tfidf.md` 포함

## 라이선스

- 코드: 이 저장소의 코드
- 데이터: 사람 리뷰는 KR3 (https://huggingface.co/datasets/leey4n/KR3, CC BY-NC-SA 4.0)에서 가져와 정규화·필터링했습니다.
  `benchmark_*.zip` 도 같은 **CC BY-NC-SA 4.0** 을 따릅니다: 출처(KR3) 표시, 비상업 용도만, 수정본도 같은 라이선스.
- LLM 리뷰는 탐지 연구용 합성 데이터입니다. 실제 리뷰로 게시하는 데 쓰면 안 됩니다.
- `data/`, `results/` 의 중간 파일은 올리지 않습니다.
