"""The sandbox: every check on `--out` before anything is created, then SLIM pointed at it."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


MARKER = ".pii-eval"
PROJECTS_HEADER = "# written by scripts/pii_eval: a sandbox, never a real vault's config"


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def discovered_vault() -> Path | None:
    """The vault SLIM would discover on this machine, from `slim/vaultpath.py` loaded by path
    (stdlib only), so nothing from `slim` is imported before the sandbox is set."""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("_pii_eval_vaultpath",
                                                      REPO / "slim" / "vaultpath.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return Path(mod.discover_vault()).expanduser().resolve()
    except Exception:                                  # noqa: BLE001 - no discovery, no refusal
        return None


def _refusal(p: Path, *, home: Path, vault: Path | None) -> str | None:
    if _inside(p, home / "Library" / "Mobile Documents"):
        return "under iCloud Drive"
    if _inside(p, home / "Library" / "Application Support" / "slim"):
        return "inside the real brain DB folder"
    if vault is not None and _inside(p, vault):
        return "inside the vault SLIM discovers on this machine"
    if ".obsidian" in p.parts:
        return "inside an Obsidian vault"
    for q in (p, *p.parents):
        if (q / ".obsidian").is_dir():
            return "inside an Obsidian vault"
        if (q / ".git").exists():
            return "inside a git working tree"
    return None


def guard(out, *, home: Path | None = None, discovered=discovered_vault) -> Path:
    """`out` resolved once, or exit, WITHOUT creating or writing anything. `out` and the vault,
    data and projects.yaml under it are resolved through symlinks, and each must stay clear of
    iCloud, the brain DB, the discovered vault, any Obsidian vault and any git working tree. An
    existing non-empty `out` must carry pii_eval's marker."""
    home = Path(home or Path.home()).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    vault = discovered()
    targets = {"--out": out}
    for name in ("vault", "data", "projects.yaml"):
        targets[f"--out/{name}"] = (out / name).resolve()
    for name, p in targets.items():
        why = _refusal(p, home=home, vault=vault)
        if why:
            raise SystemExit(f"refusing {name}: it is {why}")
    if out.exists():
        if not out.is_dir():
            raise SystemExit("refusing --out: it is not a directory")
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise SystemExit("refusing --out: not a pii_eval output directory "
                             f"(non-empty, no {MARKER} marker)")
    projects = targets["--out/projects.yaml"]
    if projects.exists() and not projects.read_text(errors="replace").startswith(PROJECTS_HEADER):
        raise SystemExit("refusing --out: its projects.yaml was not written by pii_eval")
    return out


def write_projects(path: Path, classes) -> None:
    lines = [PROJECTS_HEADER, 'owner: "Alex"', "projects:", "  work:",
             '    description: "Day job — meetings and calls."', "  school:",
             '    description: "Coursework — lectures and readings."']
    if classes is not None:
        lines += ["pii:", f"  mask: [{', '.join(classes)}]"]
    path.write_text("\n".join(lines) + "\n")


def _config_outside(config, out: Path) -> list[str]:
    return [name for name in ("VAULT", "DATA_DIR", "PROJECTS_YAML", "DB_PATH", "TRACE_DIR")
            if not _inside(Path(getattr(config, name)).resolve(), out)]


def sandbox(out, *, classes=None, home: Path | None = None, discovered=discovered_vault):
    """Point SLIM at `<out>` and prove it: returns (resolved out, `slim.config`), or exits having
    created and written nothing."""
    out = guard(out, home=home, discovered=discovered)
    already = sys.modules.get("slim.config")
    if already is not None and _config_outside(already, out):
        raise SystemExit("refusing to run: slim was imported before the sandbox was set, and "
                         f"slim.config.{_config_outside(already, out)[0]} resolves outside --out")
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("pii_eval output directory\n")
    vault, data, projects = out / "vault", out / "data", out / "projects.yaml"
    for d in (vault / "Notes" / "work", vault / "Notes" / "school",
              vault / "Attachments" / "_incoming", data):
        d.mkdir(parents=True, exist_ok=True)
    write_projects(projects, classes)
    os.environ["SLIM_VAULT"] = str(vault)
    os.environ["SLIM_DATA_DIR"] = str(data)
    os.environ["SLIM_PROJECTS"] = str(projects)
    import slim.config as config

    bad = _config_outside(config, out)
    if bad:
        raise SystemExit(f"refusing to run: slim.config.{bad[0]} resolves outside --out")
    return out, config
