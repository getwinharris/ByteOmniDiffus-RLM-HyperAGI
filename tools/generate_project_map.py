#!/usr/bin/env python3
"""
Generate project_map.mmd.

This is the DOCS-INDEX repository map generator. It scans the checked-out
repository, not chat memory, and writes a deterministic Mermaid source graph for
agents and maintainers.

Usage:
    python3 tools/generate_project_map.py
    python3 tools/generate_project_map.py --check
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "project_map.mmd"

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".tox",
}

GENERATED_FILES = {"project_map.mmd"}
MANUAL_DOC_NAMES = {
    "AGENTS.md",
    "README.md",
    "Design.md",
    "DESIGN.md",
    "CHANGELOG.md",
    "LICENSE",
    "VERSION",
    "systematicprojectmap.mmd",
}
EXPECTED_ROOT_FILES = ["AGENTS.md", "README.md", "pyproject.toml", "project_map.mmd"]
OPTIONAL_DURABLE_FILES = ["VERSION", "CHANGELOG.md", "Design.md", "DESIGN.md"]
TEXT_SUFFIXES = {".py", ".md", ".mmd", ".toml", ".yaml", ".yml", ".json", ".txt", ".ini", ".cfg"}
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|NOTE)\b[:\s-]*(.*)", re.IGNORECASE)


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    short_name: str
    lineno: int
    kind: str
    calls: tuple[str, ...] = field(default_factory=tuple)
    decorators: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PythonFileInfo:
    path: Path
    module: str
    classes: tuple[str, ...]
    functions: tuple[FunctionInfo, ...]
    imports: tuple[str, ...]
    routes: tuple[str, ...]
    entrypoints: tuple[str, ...]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> list[Path]:
    found: list[Path] = []
    for current_root, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(files):
            found.append(Path(current_root) / name)
    return found


def iter_dirs() -> list[Path]:
    found: list[Path] = [ROOT]
    for current_root, dirs, _files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in dirs:
            found.append(Path(current_root) / name)
    return found


def node_id(label: str) -> str:
    return "N_" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]


def label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "'").replace("\n", "<br/>")


def classify_file(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    relative = rel(path)
    if relative in GENERATED_FILES:
        return "generated"
    if name == "AGENTS.md":
        return "contract"
    if name in MANUAL_DOC_NAMES or suffix in {".md", ".mmd", ".rst"}:
        return "manual-doc"
    if suffix == ".py":
        return "python"
    if suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "config"
    if suffix in {".sh", ".bash", ".ps1"}:
        return "script"
    if "test" in path.parts or path.name.startswith("test_"):
        return "test"
    return "asset-or-other"


def safe_read(path: Path) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in MANUAL_DOC_NAMES:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def parse_python(path: Path) -> PythonFileInfo:
    text = safe_read(path)
    module = rel(path).replace("/", ".").removesuffix(".py")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return PythonFileInfo(path, module, (), (), (), (), ())

    classes: list[str] = []
    functions: list[FunctionInfo] = []
    imports: list[str] = []
    routes: list[str] = []
    entrypoints: list[str] = []
    parent_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            classes.append(node.name)
            parent_stack.append(node.name)
            self.generic_visit(node)
            parent_stack.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imports.append(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                imports.append(("." * node.level) + node.module)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record_function(node, "method" if parent_stack else "function")
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record_function(node, "async-method" if parent_stack else "async-function")
            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            if _is_main_guard(node):
                entrypoints.append("__main__")
            self.generic_visit(node)

        def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
            decorators = tuple(filter(None, (_decorator_name(d) for d in node.decorator_list)))
            route_names = tuple(d for d in decorators if d.startswith(("app.", "router.")))
            routes.extend(route_names)
            calls = tuple(sorted(_calls_inside(node)))
            full_name = ".".join(parent_stack + [node.name]) if parent_stack else node.name
            functions.append(FunctionInfo(full_name, node.name, node.lineno, kind, calls, decorators))

    Visitor().visit(tree)
    return PythonFileInfo(
        path=path,
        module=module,
        classes=tuple(sorted(set(classes))),
        functions=tuple(functions),
        imports=tuple(sorted(set(imports))),
        routes=tuple(sorted(set(routes))),
        entrypoints=tuple(sorted(set(entrypoints))),
    )


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _calls_inside(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                calls.add(name)
    return calls


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    return any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)


def nearest_parent_agent(path: Path, agents: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for agent in agents:
        if agent.parent == path:
            continue
        try:
            path.relative_to(agent.parent)
        except ValueError:
            continue
        candidates.append(agent)
    return sorted(candidates, key=lambda p: len(p.parts), reverse=True)[0] if candidates else None


def scan_todos(files: list[Path]) -> list[tuple[str, int, str, str]]:
    todos: list[tuple[str, int, str, str]] = []
    for path in files:
        text = safe_read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = TODO_RE.search(line)
            if match:
                todos.append((rel(path), lineno, match.group(1).upper(), match.group(2).strip()[:120]))
    return todos


def detect_missing(files: list[Path]) -> list[str]:
    names = {rel(p) for p in files}
    missing: list[str] = []
    for expected in EXPECTED_ROOT_FILES:
        if expected not in names:
            missing.append(expected)
    for optional in OPTIONAL_DURABLE_FILES:
        if optional not in names:
            missing.append(f"optional:{optional}")
    if not any(p.name.startswith("test_") or "tests" in p.parts for p in files):
        missing.append("optional:tests/")
    return missing


def detect_connectedness(py_infos: list[PythonFileInfo]) -> tuple[list[str], dict[str, list[str]]]:
    called: set[str] = set()
    locations: dict[str, list[str]] = {}
    for info in py_infos:
        for fn in info.functions:
            locations.setdefault(fn.short_name, []).append(f"{rel(info.path)}:{fn.lineno}")
            called.update(c.split(".")[-1] for c in fn.calls)
    unconnected = sorted(name for name in locations if name not in called and not name.startswith("_"))
    duplicates = {name: locs for name, locs in locations.items() if len(locs) > 1}
    return unconnected, duplicates


def add_node(lines: list[str], key: str, text: str, indent: str = "  ") -> str:
    nid = node_id(key)
    lines.append(f'{indent}{nid}["{label(text)}"]')
    return nid


def add_edge(lines: list[str], src: str, dst: str, text: str | None = None, indent: str = "  ") -> None:
    if text:
        lines.append(f'{indent}{src} -->|"{label(text)}"| {dst}')
    else:
        lines.append(f"{indent}{src} --> {dst}")


def build_map() -> str:
    files = iter_files()
    dirs = iter_dirs()
    agents = sorted([p for p in files if p.name == "AGENTS.md"], key=rel)
    py_infos = [parse_python(p) for p in files if p.suffix == ".py"]
    todos = scan_todos(files)
    missing = detect_missing(files)
    unconnected, duplicates = detect_connectedness(py_infos)

    lines: list[str] = []
    lines.append("%% project_map.mmd")
    lines.append("%% GENERATED by tools/generate_project_map.py. Do not hand-edit.")
    lines.append("flowchart TD")

    repo = add_node(lines, "repo", "Repository: ByteOmniDiffus-RLM-HyperAGI")
    identity = add_node(lines, "identity", "Project identity: ByteOmniDiffus")
    contract = add_node(lines, "contract", "DOCS-INDEX contract hierarchy")
    add_edge(lines, repo, identity)
    add_edge(lines, repo, contract)

    lines.append("")
    lines.append('  subgraph DirectoryTree["Folders and durable boundaries"]')
    dir_nodes: dict[Path, str] = {}
    for path in sorted(dirs, key=lambda p: (len(p.parts), rel(p) if p != ROOT else "")):
        key = "dir:" + (rel(path) if path != ROOT else ".")
        name = "." if path == ROOT else rel(path)
        dir_nodes[path] = add_node(lines, key, name, "    ")
        if path == ROOT:
            add_edge(lines, repo, dir_nodes[path], indent="    ")
        else:
            add_edge(lines, dir_nodes[path.parent], dir_nodes[path], indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph Contracts["AGENTS.md hierarchy"]')
    agent_nodes: dict[Path, str] = {}
    for agent in agents:
        agent_nodes[agent] = add_node(lines, "agent:" + rel(agent), rel(agent), "    ")
        if agent.parent == ROOT:
            add_edge(lines, contract, agent_nodes[agent], indent="    ")
        else:
            parent = nearest_parent_agent(agent.parent, agents)
            if parent:
                add_edge(lines, agent_nodes[parent], agent_nodes[agent], indent="    ")
            else:
                add_edge(lines, contract, agent_nodes[agent], indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph FilesByType["Files by type"]')
    type_nodes: dict[str, str] = {}
    for file_type in sorted({classify_file(p) for p in files}):
        type_nodes[file_type] = add_node(lines, "type:" + file_type, file_type, "    ")
        add_edge(lines, repo, type_nodes[file_type], indent="    ")
    for path in sorted(files, key=rel):
        ftype = classify_file(path)
        text = f"{rel(path)}\n{ftype}"
        fn = add_node(lines, "file:" + rel(path), text, "    ")
        add_edge(lines, type_nodes[ftype], fn, indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph PythonStructure["Python modules, classes, functions, routes"]')
    fn_nodes: dict[tuple[str, str], str] = {}
    for info in sorted(py_infos, key=lambda item: rel(item.path)):
        module_node = add_node(
            lines,
            "py:" + rel(info.path),
            f"{info.module}\nclasses={len(info.classes)} functions={len(info.functions)} imports={len(info.imports)}",
            "    ",
        )
        add_edge(lines, repo, module_node, indent="    ")
        if info.entrypoints:
            ep = add_node(lines, "entry:" + rel(info.path), "entrypoint: " + ", ".join(info.entrypoints), "    ")
            add_edge(lines, module_node, ep, indent="    ")
        for route in info.routes:
            rn = add_node(lines, "route:" + rel(info.path) + route, "route decorator: " + route, "    ")
            add_edge(lines, module_node, rn, indent="    ")
        for cls in info.classes:
            cn = add_node(lines, "class:" + rel(info.path) + cls, "class " + cls, "    ")
            add_edge(lines, module_node, cn, indent="    ")
        for fn in info.functions:
            fn_node = add_node(lines, "fn:" + rel(info.path) + fn.name, f"{fn.kind} {fn.name}\nline {fn.lineno}", "    ")
            fn_nodes[(rel(info.path), fn.short_name)] = fn_node
            add_edge(lines, module_node, fn_node, indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph Imports["Import graph"]')
    for info in sorted(py_infos, key=lambda item: rel(item.path)):
        src = node_id("py:" + rel(info.path))
        for imported in info.imports[:40]:
            dst = add_node(lines, "import:" + rel(info.path) + imported, imported, "    ")
            add_edge(lines, src, dst, "imports", "    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph CallSignals["Function connection signals"]')
    unc = add_node(lines, "unconnected", f"Potentially unconnected public functions\n{len(unconnected)}", "    ")
    dup = add_node(lines, "duplicates", f"Duplicate function names\n{len(duplicates)}", "    ")
    add_edge(lines, repo, unc, indent="    ")
    add_edge(lines, repo, dup, indent="    ")
    for name in unconnected[:60]:
        n = add_node(lines, "unconnected:" + name, name, "    ")
        add_edge(lines, unc, n, indent="    ")
    for name, locs in sorted(duplicates.items())[:60]:
        n = add_node(lines, "duplicate:" + name, f"{name}\n" + "; ".join(locs[:6]), "    ")
        add_edge(lines, dup, n, indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph DocsAndStatus["Documentation, TODOs, and expected links"]')
    manual = add_node(lines, "manual-docs", "Manual docs and contracts", "    ")
    generated = add_node(lines, "generated-docs", "Generated files", "    ")
    todo_node = add_node(lines, "todos", f"TODO/FIXME markers\n{len(todos)}", "    ")
    missing_node = add_node(lines, "missing", f"Missing or optional durable files\n{len(missing)}", "    ")
    add_edge(lines, repo, manual, indent="    ")
    add_edge(lines, repo, generated, indent="    ")
    add_edge(lines, repo, todo_node, indent="    ")
    add_edge(lines, repo, missing_node, indent="    ")
    for path in sorted(files, key=rel):
        ftype = classify_file(path)
        if ftype in {"manual-doc", "contract"}:
            add_edge(lines, manual, node_id("file:" + rel(path)), indent="    ")
        if ftype == "generated":
            add_edge(lines, generated, node_id("file:" + rel(path)), indent="    ")
    for file_path, lineno, tag, msg in todos[:60]:
        n = add_node(lines, f"todo:{file_path}:{lineno}", f"{tag} {file_path}:{lineno}\n{msg}", "    ")
        add_edge(lines, todo_node, n, indent="    ")
    for item in missing:
        n = add_node(lines, "missing:" + item, item, "    ")
        add_edge(lines, missing_node, n, indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph RuntimeView["Current runtime view from repository files"]')
    byte_runtime = add_node(lines, "runtime:byteomnidiffus", "ByteOmniDiffus runtime boundary", "    ")
    rlm_base = add_node(lines, "runtime:rlm", "Internal RLM framework boundary", "    ")
    map_tool = add_node(lines, "runtime:maptool", "DOCS-INDEX map generator", "    ")
    add_edge(lines, identity, byte_runtime, indent="    ")
    add_edge(lines, byte_runtime, rlm_base, "uses internal base where connected", "    ")
    add_edge(lines, contract, map_tool, "generates", "    ")
    for p in files:
        rp = rel(p)
        if rp.startswith("rlcodar_hyperagi/"):
            add_edge(lines, byte_runtime, node_id("file:" + rp), indent="    ")
        elif rp.startswith("rlm/"):
            add_edge(lines, rlm_base, node_id("file:" + rp), indent="    ")
        elif rp.startswith("tools/"):
            add_edge(lines, map_tool, node_id("file:" + rp), indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph Workflows["Required maintenance workflows"]')
    read_chain = add_node(lines, "workflow:read", "Read applicable AGENTS.md chain", "    ")
    generate = add_node(lines, "workflow:generate", "python3 tools/generate_project_map.py", "    ")
    check = add_node(lines, "workflow:check", "python3 tools/generate_project_map.py --check", "    ")
    issue_pr = add_node(lines, "workflow:issue", "Issue-backed branch and PR with version impact", "    ")
    add_edge(lines, read_chain, generate, indent="    ")
    add_edge(lines, generate, check, indent="    ")
    add_edge(lines, check, issue_pr, indent="    ")
    lines.append("  end")

    lines.append("")
    lines.append(f"  %% totals: files={len(files)} dirs={len(dirs)} python_files={len(py_infos)} agents={len(agents)} todos={len(todos)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the DOCS-INDEX project map")
    parser.add_argument("--check", action="store_true", help="fail if project_map.mmd is stale")
    args = parser.parse_args()

    rendered = build_map()
    if args.check:
        if not OUTPUT.exists():
            print("project_map.mmd is missing", file=sys.stderr)
            return 1
        current = OUTPUT.read_text(encoding="utf-8")
        if current != rendered:
            print("project_map.mmd is stale. Run: python3 tools/generate_project_map.py", file=sys.stderr)
            return 1
        print("project_map.mmd is up to date")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
