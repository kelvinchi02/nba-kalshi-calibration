"""requirements.txt and pyproject.toml must list the same packages."""
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize(name: str) -> str:
    """Package name without version or extras, with -, _ and . treated alike."""
    bare = re.split(r"[<>=!~\[; ]", name.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", bare).lower()


def test_requirements_and_pyproject_list_the_same_packages():
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = {normalize(line) for line in lines if line.strip() and not line.startswith("#")}
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        project = {normalize(dep) for dep in tomllib.load(f)["project"]["dependencies"]}
    assert requirements == project
    assert {"jupyter", "ipykernel", "nbconvert"} <= project


def test_lock_file_covers_every_dependency():
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    locked = {normalize(n) for n in re.findall(r'^name = "([^"]+)"', lock, flags=re.M)}
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        project = {normalize(dep) for dep in tomllib.load(f)["project"]["dependencies"]}
    assert project <= locked
