"""scripts/run_notebooks.py: outputs are saved, and a failing cell stops the run."""
import sys
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_notebooks  # noqa: E402


def write_notebook(path: Path, sources: list[str]) -> None:
    nb = new_notebook(cells=[new_markdown_cell("# test")] + [new_code_cell(s) for s in sources])
    nbformat.write(nb, path)


@pytest.fixture
def notebook_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_notebooks, "NOTEBOOK_DIR", tmp_path)
    return tmp_path


def test_select_orders_by_name_and_accepts_names_without_extension(notebook_dir):
    for name in ("02_b.ipynb", "01_a.ipynb"):
        write_notebook(notebook_dir / name, ["1"])
    assert [p.name for p in run_notebooks.select([])] == ["01_a.ipynb", "02_b.ipynb"]
    assert [p.name for p in run_notebooks.select(["02_b"])] == ["02_b.ipynb"]
    with pytest.raises(SystemExit, match="03_c.ipynb"):
        run_notebooks.select(["03_c"])


def test_run_saves_outputs_and_uses_the_notebook_folder(notebook_dir):
    path = notebook_dir / "ok.ipynb"
    write_notebook(path, ["import os\nprint(os.path.basename(os.getcwd()))", "6 * 7"])
    run_notebooks.run(path)
    cells = nbformat.read(path, as_version=4).cells
    assert cells[1].outputs[0].text.strip() == notebook_dir.name
    assert cells[2].outputs[0].data["text/plain"] == "42"
    assert "execution" not in cells[2].metadata          # no timestamps in the file


def test_failing_cell_exits_with_code_1_and_leaves_the_file(notebook_dir, monkeypatch, capsys):
    path = notebook_dir / "bad.ipynb"
    write_notebook(path, ["x = 1", "raise ValueError('boom')", "x = 2"])
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py"])
    with pytest.raises(SystemExit) as exit_info:
        run_notebooks.main()
    assert exit_info.value.code == 1
    assert "boom" in capsys.readouterr().err
    assert path.read_bytes() == before
