"""Web model upload and selection checks without loading weights."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import config


def test_model_upload_selects_saved_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.csv")
    monkeypatch.setitem(sys.modules, "streamlit_webrtc", None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/app.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    assert app.button(key="save_model").disabled

    app.file_uploader(key="model_uploader").upload("custom.pt", b"test weights").run()
    app.button(key="save_model").click().run()
    assert not app.exception and not app.error
    assert (tmp_path / "custom.pt").read_bytes() == b"test weights"
    assert app.selectbox(key="selected_model").value == "本地模型：custom.pt"
    assert "本地模型：custom.pt" in app.multiselect[0].options

    app.button(key="save_model").click().run()
    assert not app.exception
    assert any("同名模型已存在" in error.value for error in app.error)
    assert (tmp_path / "custom.pt").read_bytes() == b"test weights"
