"""
导出项目目录结构为 Markdown 文件。

用法：
    python export_structure.py              # 默认输出 project_structure.md
    python export_structure.py <输出目录>    # 指定输出目录（默认为项目根目录）
    python export_structure.py . --base <名称>  # 自定义基础文件名

特性：
    - 忽略 __pycache__、.git、node_modules、.codebuddy 等常见无关目录
    - 自动编号：若根目录已存在同名文件，自动追加 (1)、(2)...
    - 输出树形目录 + 文件统计
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ---------- 配置 ----------
BASE_FILENAME = "project_structure"          # 基础文件名（不含扩展名）
OUTPUT_EXT = ".md"                           # 输出扩展名
IGNORE_DIRS = {                               # 忽略的目录名
    "__pycache__",
    ".git",
    ".codebuddy",
    "node_modules",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "chroma_db",
}
IGNORE_FILES = {                              # 忽略的文件名
    ".DS_Store",
    "Thumbs.db",
}
IGNORE_EXTENSIONS = {                         # 忽略的扩展名
    ".pyc",
    ".pyo",
    ".sqlite3",
}

# 树形符号
BRANCH = "├── "
LAST_BRANCH = "└── "
PIPE = "│   "
SPACE = "    "


def should_ignore(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in IGNORE_DIRS or name.startswith(".")
    else:
        if name in IGNORE_FILES:
            return True
        ext = os.path.splitext(name)[1].lower()
        return ext in IGNORE_EXTENSIONS


def get_output_path(root: Path, base_name: str) -> Path:
    """返回唯一输出路径；若存在同名文件则自动编号"""
    candidate = root / f"{base_name}{OUTPUT_EXT}"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = root / f"{base_name}({i}){OUTPUT_EXT}"
        if not candidate.exists():
            return candidate
        i += 1


def build_tree(root: Path, prefix: str = "") -> list[str]:
    """递归构建树形目录文本"""
    lines = []
    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        return lines

    # 过滤掉忽略项
    entries = [e for e in entries if not should_ignore(e.name, e.is_dir())]

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = LAST_BRANCH if is_last else BRANCH
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")

        if entry.is_dir():
            extension = SPACE if is_last else PIPE
            lines.extend(build_tree(entry, prefix + extension))

    return lines


def count_files(root_dir: Path) -> dict:
    """统计项目文件信息"""
    stats = {"dirs": 0, "files": 0, "total_lines": 0}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 原地过滤忽略目录
        dirnames[:] = [d for d in dirnames if not should_ignore(d, True)]
        stats["dirs"] += len(dirnames)

        for fname in filenames:
            if should_ignore(fname, False):
                continue
            stats["files"] += 1
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    stats["total_lines"] += sum(1 for _ in f)
            except (PermissionError, OSError):
                pass
    return stats


def export_structure(root: Path, base_name: str = BASE_FILENAME) -> Path:
    """主函数：导出结构并写入 Markdown"""
    output_path = get_output_path(root, base_name)
    stats = count_files(root)
    tree_lines = build_tree(root)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# 项目目录结构

> 导出时间：{now}
> 项目路径：`{root.resolve()}`
> 目录数：{stats["dirs"]} 　 文件数：{stats["files"]} 　 总行数：{stats["total_lines"]}

## 目录树

```
{root.name}/
"""
    for line in tree_lines:
        content += line + "\n"

    content += "```\n"

    # 文件清单
    content += "\n## 文件清单\n\n"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_ignore(d, True)]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        for fname in sorted(filenames):
            if should_ignore(fname, False):
                continue
            fpath = os.path.join(rel_dir, fname) if rel_dir else fname
            content += f"- `{fpath}`\n"

    output_path.write_text(content, encoding="utf-8")
    print(f"[OK] 结构已导出 -> {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    # 解析命令行参数
    args = sys.argv[1:]

    target_dir = Path.cwd()
    base_name = BASE_FILENAME

    i = 0
    while i < len(args):
        if args[i] == "--base" and i + 1 < len(args):
            base_name = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            target_dir = Path(args[i]).resolve()
            if not target_dir.is_dir():
                print(f"[ERROR] 目录不存在: {target_dir}")
                sys.exit(1)
            i += 1
        else:
            i += 1

    export_structure(target_dir, base_name)
