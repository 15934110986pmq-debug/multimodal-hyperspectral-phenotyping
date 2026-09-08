"""多模态高光谱植物表型系统 - 一键安装 / 依赖检测 / 自检 / 启动指引。

用法:
  python bootstrap.py                 # 检测环境 -> 补全依赖 -> 自检 -> 打印启动指引
  python bootstrap.py --run           # 自检通过后直接启动 Streamlit 演示
  python bootstrap.py --with-optional # 一并安装公开数据集的可选依赖
  python bootstrap.py --skip-install  # 只检测与自检, 不自动安装
  python bootstrap.py --quick         # 自检只做模块导入, 不跑端到端流水线

Windows 用户也可直接双击 install.bat, Linux/macOS 执行 ./install.sh。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
OPTIONAL_REQUIREMENTS = ROOT / "requirements-optional.txt"

KEY_MODULES = [
    "numpy", "scipy", "sklearn", "pandas",
    "matplotlib", "tifffile", "streamlit", "plotly",
]

MIN_PYTHON = (3, 9)

CHECK_SNIPPET = r'''
import json, sys
from pathlib import Path
from packaging.requirements import Requirement
from importlib.metadata import version as _v
p = Path(sys.argv[1])
satisfied, needed = [], []
for raw in p.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    try:
        req = Requirement(line)
    except Exception:
        needed.append(line)
        continue
    try:
        cur = _v(req.name)
    except Exception:
        needed.append(line)
        continue
    if req.specifier and not req.specifier.contains(cur, prereleases=True):
        needed.append(line)
    else:
        satisfied.append(line)
print(json.dumps({"satisfied": satisfied, "needed": needed}, ensure_ascii=False))
'''

IMPORT_SNIPPET = r'''
import importlib.util, json
mods = __MODS__
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(json.dumps({"missing": missing}))
'''


def log(text: str = "") -> None:
    print(text)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(use_python: str) -> Path:
    venv_dir = ROOT / ".venv"
    py = _venv_python(venv_dir)
    if py.exists():
        log(f"[环境] 复用已有虚拟环境: {py}")
        return venv_dir

    log("[环境] 未检测到 .venv, 正在创建虚拟环境 ...")
    subprocess.check_call([use_python, "-m", "venv", str(venv_dir)])
    log(f"[环境] 创建完成: {py}")
    return venv_dir


def _check_requirements(python_exe: Path, path: Path) -> tuple[list, list]:
    """用 venv 内的 python 检查 requirements, 返回 (已满足清单, 待安装清单)。"""
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", CHECK_SNIPPET, str(path)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        data = json.loads(proc.stdout.strip())
        return data["satisfied"], data["needed"]
    except Exception:
        return [], [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]


def _install(python_exe: Path, req_file: Path) -> None:
    log(f"[依赖] 正在安装 {req_file.name} ...")
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(req_file)])


def _resolve_dependencies(python_exe: Path, with_optional: bool, skip_install: bool) -> None:
    log("")
    log("==== 依赖检测 ====")
    ok, need = _check_requirements(python_exe, REQUIREMENTS)
    log(f"  - {REQUIREMENTS.name}: 已满足 {len(ok)} 项, 待安装/升级 {len(need)} 项")
    for item in need:
        log(f"     - {item}")

    if need and not skip_install:
        _install(python_exe, REQUIREMENTS)
        _, need = _check_requirements(python_exe, REQUIREMENTS)
        if need:
            log(f"[依赖] 仍有 {len(need)} 项未满足: {need}")
    elif need and skip_install:
        log("[依赖] 检测到缺失依赖, 但已跳过自动安装 (--skip-install)")

    if OPTIONAL_REQUIREMENTS.exists():
        opt_ok, opt_need = _check_requirements(python_exe, OPTIONAL_REQUIREMENTS)
        if opt_need:
            log(f"[可选] {OPTIONAL_REQUIREMENTS.name} 有 {len(opt_need)} 项未安装"
                f" (联网接入公开数据集才需要)")
            if with_optional:
                _install(python_exe, OPTIONAL_REQUIREMENTS)
            else:
                log("[可选] 本次跳过; 需要时用 --with-optional 安装. 不影响本地演示.")
        else:
            log(f"[可选] {OPTIONAL_REQUIREMENTS.name} 已全部安装")


def _selftest_imports(python_exe: Path) -> bool:
    """用 venv 内的 python 自检关键模块能否导入。"""
    log("")
    log("==== 自检 - 模块导入 ====")
    snippet = IMPORT_SNIPPET.replace("__MODS__", repr(KEY_MODULES))
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", snippet],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        missing = set(json.loads(proc.stdout.strip())["missing"])
    except Exception:
        log("  [失败] 无法在虚拟环境中做导入检查")
        return False

    ok_all = True
    for mod in KEY_MODULES:
        if mod in missing:
            log(f"  [缺失] {mod}")
            ok_all = False
        else:
            log(f"  [OK]  {mod}")
    return ok_all


def _selftest_e2e(python_exe: Path) -> bool:
    log("")
    log("==== 自检 - 端到端流水线 ====")
    log("  运行: python main.py --loader synthetic --n-samples 40 --no-plots")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            [str(python_exe), str(ROOT / "main.py"),
             "--loader", "synthetic", "--n-samples", "40", "--no-plots"],
            cwd=str(ROOT), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
    except Exception as exc:
        log(f"  [失败] {exc}")
        return False

    out_lines = proc.stdout.strip().splitlines() or ["(无输出)"]
    log("  最后一行: " + out_lines[-1])
    if proc.returncode != 0:
        log(f"  [失败] 返回码 {proc.returncode}")
        log("  stderr: " + (proc.stderr.strip() or "(无)"))
        return False
    log("  [OK] 流水线运行成功")
    return True


def _print_guidance(python_exe: Path, run_now: bool) -> None:
    log("")
    log("==== 安装完成 - 如何启动 ====")
    log("1) 命令行端到端流水线 (合成数据, 离线可跑):")
    log(f"    {python_exe} main.py --loader synthetic --model random_forest")
    log("2) 交互式演示 (Streamlit):")
    if sys.platform.startswith("win"):
        log("    .venv\\Scripts\\python -m streamlit run app.py")
        log("    或执行 scripts\\run.ps1")
    else:
        log("    .venv/bin/python -m streamlit run app.py")
        log("    或执行 ./scripts/run.sh")
    log("    浏览器打开 http://localhost:8501")
    log("3) 容器化部署: docker compose up --build")

    if run_now:
        log("")
        log("==== 正在启动 Streamlit 演示 ====")
        subprocess.call([str(python_exe), "-m", "streamlit", "run", str(ROOT / "app.py")],
                        cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="一键安装 / 依赖检测 / 自检 / 启动指引")
    parser.add_argument("--run", action="store_true", help="自检通过后直接启动演示")
    parser.add_argument("--with-optional", action="store_true",
                        help="一并安装公开数据集可选依赖")
    parser.add_argument("--skip-install", action="store_true",
                        help="只检测与自检, 不自动安装依赖")
    parser.add_argument("--quick", action="store_true",
                        help="自检只做模块导入, 不跑端到端流水线")
    args = parser.parse_args()

    log("多模态高光谱植物表型系统 - 一键安装引导")
    log(f"项目目录: {ROOT}")

    if sys.version_info < MIN_PYTHON:
        required = ".".join(map(str, MIN_PYTHON))
        log(f"[错误] Python 版本过低: {sys.version.split()[0]}, 需要 >= {required}")
        return 1
    log(f"[环境] Python {sys.version.split()[0]} (引导脚本)")

    venv_dir = _ensure_venv(sys.executable)
    venv_py = _venv_python(venv_dir)

    _resolve_dependencies(venv_py, args.with_optional, args.skip_install)

    imported = _selftest_imports(venv_py)
    e2e_ok = True if args.quick else _selftest_e2e(venv_py)
    passed = imported and e2e_ok
    log(f"")
    log(f"==== 自检结果: {'通过' if passed else '未通过'} ====")
    if not passed:
        return 1

    _print_guidance(venv_py, args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
