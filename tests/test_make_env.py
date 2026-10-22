import sys
from pathlib import Path

import pytest

from app.config import Settings
from scripts import make_env


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["make_env.py", *args])
    return make_env.main()


def test_creates_an_env_file_whose_key_the_application_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / ".env"

    assert run(monkeypatch, "--output", str(target), "--client", "me") == 0

    printed = capsys.readouterr().out
    monkeypatch.delenv("API_KEYS", raising=False)
    [config] = Settings(_env_file=target).api_keys
    assert config.client == "me" and config.scopes == {"chat", "admin"}
    assert config.key.get_secret_value() in printed
    assert "LOG_FORMAT" in target.read_text(encoding="utf-8")  # the example settings came along


def test_never_overwrites_an_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / ".env"
    target.write_text("KEEP=me\n", encoding="utf-8")

    assert run(monkeypatch, "--output", str(target)) == 1
    assert target.read_text(encoding="utf-8") == "KEEP=me\n"


def test_each_run_makes_a_different_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = []
    for name in ("one", "two"):
        target = tmp_path / name
        run(monkeypatch, "--output", str(target))
        monkeypatch.delenv("API_KEYS", raising=False)
        keys.append(Settings(_env_file=target).api_keys[0].key.get_secret_value())

    assert keys[0] != keys[1] and len(keys[0]) >= 32


def active_lines(path: Path) -> list[str]:
    # The example file mentions COMPOSE_LLM_URL in a comment; only a real setting counts
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def test_fake_llm_flag_points_compose_at_the_stand_in_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".env"

    run(monkeypatch, "--output", str(target), "--fake-llm")

    assert "COMPOSE_LLM_URL=http://fake-llm:9000/v1" in active_lines(target)
    plain = tmp_path / "plain"
    run(monkeypatch, "--output", str(plain))
    assert not any(line.startswith("COMPOSE_LLM_URL=") for line in active_lines(plain))
