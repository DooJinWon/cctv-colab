import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/CCTV_Privacy_Colab.ipynb"
pytestmark = pytest.mark.skipif(
    not NOTEBOOK.exists(),
    reason="Uploadable notebook is not duplicated inside its embedded source ZIP.",
)


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_schema():
    import nbformat

    nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))


def test_notebook_cells_compile_without_execution():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])


def test_notebook_has_no_saved_outputs():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"]
            assert cell["execution_count"] is None


def test_embedded_source_is_current_and_excludes_runtime_data():
    cell = next(c for c in notebook()["cells"] if c["cell_type"] == "code")
    tree = ast.parse(cell["source"])
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE_ZIP_B64" for t in n.targets)
    )
    payload = base64.b64decode(ast.literal_eval(assignment.value))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            assert not name.endswith((".mp4", ".pt", ".onnx", ".sqlite"))
            assert (
                archive.read(name) == (ROOT / name).read_bytes()
            ), f"Stale embedded source: {name}"


def test_bootstrap_extracts_without_installing_packages(tmp_path, monkeypatch):
    cell = next(c for c in notebook()["cells"] if c["cell_type"] == "code")
    source = cell["source"].replace("INSTALL_DEPENDENCIES = True", "INSTALL_DEPENDENCIES = False")
    source = source.replace(
        "CODE_ROOT = Path('/content/cctv-colab') if IN_COLAB else Path.cwd() / 'cctv-colab-runtime'",
        f'CODE_ROOT = Path({str(tmp_path / "runtime")!r})',
    )
    source = source.replace("IN_COLAB = Path('/content').is_dir()", "IN_COLAB = False")
    namespace = {}
    exec(compile(source, "notebook-setup", "exec"), namespace)
    assert (tmp_path / "runtime/cctv_privacy/pipeline.py").is_file()
    assert (tmp_path / "runtime/README.md").is_file()
