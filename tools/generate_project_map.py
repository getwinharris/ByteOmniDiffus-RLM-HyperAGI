#!/usr/bin/env python3
"""
Generate project_map.mmd.

This script is dependency-free and deterministic. It scans the repository tree,
identifies documentation contracts, Python structure, routes, configuration,
generated/manual files, and likely unconnected functions, then writes a Mermaid
graph for coding agents.

Usage:
    python3 tools/generate_project_map.py
    python3 tools/generate_project_map.py --check
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
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
    "README.md",
    "AGENTS.md",
    "Design.md",
    "DESIGN.md",
    "CHANGELOG.md",
    "LICENSE",
    "VERSION",
    "systematicprojectmap.mmd",
}


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    lineno: int
    kind: str
    calls: tuple[str, ...] = field(default_factory=tuple)
    decorators: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PythonFileInfo:
    path: Path
    classes: tuple[str, ...]
    functions: tuple[FunctionInfo, ...]
    imports: tuple[str, ...]
    routes: tuple[str, ...]
    entrypoints: tuple[str, ...]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> Iterable[Path]:
    for current_root, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(files):
            yield Path(current_root) / name


def mermaid_id(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    return "N_" + digest


def mermaid_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "'").replace("\n", "<br/>")


def classify_file(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    if rel(path) in GENERATED_FILES:
        return "generated"
    if name in MANUAL_DOC_NAMES or suffix in {".md", ".mmd", ".rst"}:
        return "manual-doc"
    if suffix == ".py":
        return "python"
    if suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "config"
    if suffix in {".sh", ".bash", ".ps1"}:
        return "script"
    return "asset-or-other"


def parse_python(path: Path) -> PythonFileInfo:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return PythonFileInfo(path, (), (), (), (), ())

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
                imports.append(alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                imports.append(node.module.split(".")[0])

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
            decorators = tuple(_decorator_name(d) for d in node.decorator_list)
            route_names = [d for d in decorators if d.startswith(("app.", "router."))]
            routes.extend(route_names)
            calls = sorted(_calls_inside(node))
            name = ".".join(parent_stack + [node.name]) if parent_stack else node.name
            functions.append(FunctionInfo(name, node.lineno, kind, tuple(calls), decorators))

    Visitor().visit(tree)
    return PythonFileInfo(
        path=path,
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
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    return any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)


def build_map() -> str:
    files = list(iter_files())
    py_infos = [parse_python(p) for p in files if p.suffix == ".py"]
    agents = [p for p in files if p.name == "AGENTS.md"]

    called: set[str] = set()
    defined: set[str] = set()
    duplicates: dict[str, list[str]] = {}

    for info in py_infos:
        for fn in info.functions:
            short = fn.name.split(".")[-1]
            defined.add(short)
            called.update(c.split(".")[-1] for c in fn.calls)
            duplicates.setdefault(short, []).append(f"{rel(info.path)}:{fn.lineno}")

    unconnected = sorted(name for name in defined if name not in called and not name.startswith("_"))
    duplicate_items = {k: v for k, v in duplicates.items() if len(v) > 1}

    lines: list[str] = []
    lines.append("%% project_map.mmd")
    lines.append("%% GENERATED by tools/generate_project_map.py. Do not hand-edit.")
    lines.append("flowchart TD")
    lines.append('  Repo["Repository<br/>ByteOmniDiffus-RLM-HyperAGI"]')
    lines.append('  Contract["DOCS-INDEX contracts<br/>AGENTS.md hierarchy"]')
    lines.append("  Repo --> Contract")

    lines.append("")
    lines.append('  subgraph Contracts["AGENTS.md hierarchy"]')
    for path in sorted(agents, key=rel):
        node = mermaid_id("agent:" + rel(path))
        lines.append(f'    {node}["{mermaid_label(rel(path))}"]')
        if path.parent == ROOT:
            lines.append(f"    Contract --> {node}")
        else:
            parent = _nearest_parent_agent(path.parent, agents)
            if parent:
                lines.append(f"    {mermaid_id('agent:' + rel(parent))} --> {node}")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph Files["Repository files by type"]')
    type_nodes: dict[str, str] = {}
    for file_type in sorted({classify_file(p) for p in files}):
        node = mermaid_id("type:" + file_type)
        type_nodes[file_type] = node
        lines.append(f'    {node}["{file_type}"]')
        lines.append(f"    Repo --> {node}")
    for path in sorted(files, key=rel):
        file_type = classify_file(path)
        node = mermaid_id("file:" + rel(path))
        label = f"{rel(path)}<br/>{file_type}"
        lines.append(f'    {node}["{mermaid_label(label)}"]')
        lines.append(f"    {type_nodes[file_type]} --> {node}")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph PythonStructure["Python structure"]')
    for info in sorted(py_infos, key=lambda x: rel(x.path)):
        file_node = mermaid_id("file:" + rel(info.path))
        py_node = mermaid_id("py:" + rel(info.path))
        summary = [rel(info.path), f"classes: {len(info.classes)}", f"functions: {len(info.functions)}"]
        if info.routes:
            summary.append("routes: " + ", ".join(info.routes[:6]))
        if info.entrypoints:
            summary.append("entrypoints: " + ", ".join(info.entrypoints))
        lines.append(f'    {py_node}["{mermaid_label("<br/>".join(summary))}"]')
        lines.append(f"    {file_node} --> {py_node}")
        for cls in info.classes[:12]:
            cls_node = mermaid_id("class:" + rel(info.path) + ":" + cls)
            lines.append(f'    {cls_node}["class {mermaid_label(cls)}"]')
            lines.append(f"    {py_node} --> {cls_node}")
        for fn in info.functions[:24]:
            fn_node = mermaid_id("fn:" + rel(info.path) + ":" + fn.name)
            label = f"{fn.kind} {fn.name}<br/>line {fn.lineno}"
            lines.append(f'    {fn_node}["{mermaid_label(label)}"]')
            lines.append(f"    {py_node} --> {fn_node}")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph Signals["Indexing signals"]')
    generated = [p for p in files if classify_file(p) == "generated"]
    manual = [p for p in files if classify_file(p) == "manual-doc"]
    lines.append(f'    Generated["Generated files<br/>{len(generated)}"]')
    lines.append(f'    ManualDocs["Manual docs<br/>{len(manual)}"]')
    lines.append(f'    Unconnected["Potentially unconnected public functions<br/>{len(unconnected)}"]')
    lines.append(f'    Duplicates["Duplicate function names<br/>{len(duplicate_items)}"]')
    lines.append("    Repo --> Generated")
    lines.append("    Repo --> ManualDocs")
    lines.append("    Repo --> Unconnected")
    lines.append("    Repo --> Duplicates")
    for name in unconnected[:40]:
        node = mermaid_id("unconnected:" + name)
        lines.append(f'    {node}["{mermaid_label(name)}"]')
        lines.append(f"    Unconnected --> {node}")
    for name, locs in sorted(duplicate_items.items())[:40]:
        node = mermaid_id("duplicate:" + name)
        lines.append(f'    {node}["{mermaid_label(name)}<br/>{mermaid_label("; ".join(locs[:4]))}"]')
        lines.append(f"    Duplicates --> {node}")
    lines.append("  end")

    lines.append("")
    lines.append('  subgraph ExpectedWorkflows["Expected workflows"]')
    lines.append('    ReadChain["Read DOCS-INDEX chain before edits"]')
    lines.append('    GenerateMap["python3 tools/generate_project_map.py"]')
    lines.append('    CheckMap["python3 tools/generate_project_map.py --check"]')
    lines.append('    IssuePR["Issue-backed branch + PR with version impact"]')
    lines.append("    Contract --> ReadChain --> GenerateMap --> CheckMap --> IssuePR")
    lines.append("  end")

    lines.append("")
    lines.append(f"  %% totals: files={len(files)} python_files={len(py_infos)} agents={len(agents)}")
    return "\n".join(lines) + "\n"


def _nearest_parent_agent(path: Path, agents: list[Path]) -> Path | None:
    candidates = []
    for agent in agents:
        if agent.parent == path:
            continue
        try:
            path.relative_to(agent.parent)
        except ValueError:
            continue
        candidates.append(agent)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(p.parts), reverse=True)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
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
