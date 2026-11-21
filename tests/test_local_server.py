"""The non-mock generation path against a fake OpenAI-compatible server
(what vLLM exposes on the GPU server)."""
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from beta import generate, prepare_human
from beta.generate import clean_generation

SERVED = "Qwen/Qwen2.5-7B-Instruct"


class FakeVLLM(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": SERVED, "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["model"] == SERVED and req["messages"][0]["role"] == "user"
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": SERVED,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "<think>짧게 쓰자</think>\n국물이 진하고 맛있어요."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


@pytest.fixture()
def local_cfg(project, server, tmp_path):
    cfg = yaml.safe_load(Path(project["config"]).read_text(encoding="utf-8"))
    for g in cfg["generation"]["generators"]:
        g["base_url"] = f"http://127.0.0.1:{server}/v1"
        g.pop("api_key_env", None)
    cfg["generation"]["max_retries"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    prepare_human.main(["--config", str(path), "--input", str(project["tsv"])])
    return path


def test_think_block_is_stripped():
    assert clean_generation("<think>\n고민\n</think>\n맛있어요.") == ("맛있어요.", None)
    assert clean_generation("<think>끝없는 생각</think>") == ("", "empty")


def test_pilot_against_local_server(local_cfg):
    generate.main(["--config", str(local_cfg), "--only", "qwen", "--pilot", "3"])
    out = local_cfg.parent / "data" / "interim" / "generated_pilot_qwen.jsonl"
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 6  # 3 per prompt type (A, C)
    assert all(r["status"] == "ok" and r["text"] == "국물이 진하고 맛있어요." and r["model"] == SERVED for r in recs)


def test_wrong_model_on_server_is_refused(local_cfg):
    # the server is serving Qwen; generating "exaone" now would mislabel the family
    with pytest.raises(SystemExit, match="떠 있는 모델"):
        generate.main(["--config", str(local_cfg), "--only", "exaone", "--pilot", "1"])


def test_server_down_is_reported(local_cfg):
    cfg = yaml.safe_load(local_cfg.read_text(encoding="utf-8"))
    for g in cfg["generation"]["generators"]:
        g["base_url"] = "http://127.0.0.1:9/v1"  # nothing listens here
    local_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(SystemExit, match="연결할 수 없습니다"):
        generate.main(["--config", str(local_cfg), "--only", "qwen", "--pilot", "1"])
