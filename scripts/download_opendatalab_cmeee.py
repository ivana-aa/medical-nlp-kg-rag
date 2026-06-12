"""Download CMeEE from OpenDataLab/OpenXLab and optionally train NER.

OpenDataLab currently recommends the ``openxlab`` CLI for dataset downloads.
The CLI requires the user to log in with their own OpenXLab AK/SK before
downloading files. This script does not store credentials and will stop with a
clear message if login is missing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "OpenDataLab/CMeEE"
DEFAULT_SOURCE_PATH = "/raw/CMeEE.tar.gz"
DEFAULT_TARGET_DIR = ROOT / "data" / "raw" / "CMeEE"


def find_openxlab() -> str | None:
    """Find the openxlab executable even when Python Scripts is not on PATH."""
    executable = shutil.which("openxlab") or shutil.which("openxlab.exe")
    if executable:
        return executable

    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    candidates = [scripts_dir / "openxlab.exe", scripts_dir / "openxlab"]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def run_command(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run a command and mirror its output to the console."""
    print("\n$ " + " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        safe_output = result.stdout.replace("\ufffd", "?")
        try:
            print(safe_output)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(safe_output.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")
    return result


def download_cmeee(openxlab: str, repo: str, source_path: str, target_dir: Path) -> bool:
    """Download CMeEE archive with openxlab. Return False when login is missing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            openxlab,
            "dataset",
            "download",
            "--dataset-repo",
            repo,
            "--source-path",
            source_path,
            "--target-path",
            str(target_dir),
        ]
    )
    if result.returncode == 0:
        return True

    output = result.stdout or ""
    if "Please login openxlab" in output or "openxlab login" in output:
        print(
            "\nOpenDataLab/OpenXLab 需要先登录并配置 AK/SK。请在终端运行：\n"
            "  openxlab login\n"
            "如果系统提示找不到 openxlab，可运行：\n"
            f"  \"{openxlab}\" login\n"
            "登录完成后重新运行本脚本。"
        )
        return False

    raise RuntimeError(f"OpenXLab download failed with exit code {result.returncode}")


def find_archive(target_dir: Path) -> Path | None:
    """Find the downloaded CMeEE archive."""
    for pattern in ["CMeEE.tar.gz", "*.tar.gz", "*.tgz", "*.zip"]:
        matches = sorted(target_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    """Extract a tar archive while preventing path traversal."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if root not in [target, *target.parents]:
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(extract_dir)


def safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive while preventing path traversal."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            target = (extract_dir / member.filename).resolve()
            if root not in [target, *target.parents]:
                raise RuntimeError(f"Unsafe archive member path: {member.filename}")
        zip_file.extractall(extract_dir)


def extract_nested_zips(root_dir: Path) -> None:
    """Extract zip files found inside the downloaded package."""
    for zip_path in sorted(root_dir.rglob("*.zip")):
        extract_dir = zip_path.parent / zip_path.stem
        marker = extract_dir / ".extracted"
        if marker.exists():
            continue
        print(f"Extracting nested zip {zip_path} -> {extract_dir}")
        safe_extract_zip(zip_path, extract_dir)
        marker.write_text("ok", encoding="utf-8")


def extract_archive(target_dir: Path) -> None:
    """Extract the downloaded archive if one is present."""
    archive_path = find_archive(target_dir)
    if archive_path is None:
        print(f"No archive found under {target_dir}; skip extraction.")
        return

    if archive_path.suffix == ".zip":
        extract_dir = target_dir / "extracted" / archive_path.stem
        print(f"Extracting {archive_path} -> {extract_dir}")
        safe_extract_zip(archive_path, extract_dir)
    else:
        extract_dir = target_dir / "extracted"
        print(f"Extracting {archive_path} -> {extract_dir}")
        safe_extract_tar(archive_path, extract_dir)
    extract_nested_zips(target_dir / "extracted")


def run_prepare(cmeee_dir: Path) -> None:
    """Convert raw CMeEE JSON files into BIO format."""
    result = run_command([sys.executable, "scripts/prepare_ner_data.py", "--cmeee-dir", str(cmeee_dir)])
    if result.returncode != 0:
        raise RuntimeError("prepare_ner_data.py failed")


def run_sampling(train_limit: int, dev_limit: int, test_limit: int) -> None:
    """Build the default CPU-friendly training subset from converted BIO."""
    result = run_command(
        [
            sys.executable,
            "scripts/sample_ner_data.py",
            "--source-dir",
            str(ROOT / "data" / "processed" / "ner_demo"),
            "--target-dir",
            str(ROOT / "data" / "processed" / "ner_train"),
            "--train",
            str(train_limit),
            "--dev",
            str(dev_limit),
            "--test",
            str(test_limit),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError("sample_ner_data.py failed")


def run_training() -> None:
    """Train and evaluate the NER model with the current processed data."""
    train = run_command([sys.executable, "scripts/train_ner.py"])
    if train.returncode != 0:
        raise RuntimeError("train_ner.py failed")


def run_evaluation() -> None:
    """Evaluate the active NER checkpoint."""
    eval_result = run_command([sys.executable, "scripts/evaluate_ner.py"])
    if eval_result.returncode != 0:
        raise RuntimeError("evaluate_ner.py failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CMeEE from OpenDataLab/OpenXLab.")
    parser.add_argument("--dataset-repo", default=DEFAULT_REPO, help="OpenXLab dataset repo name.")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH, help="Remote file path in dataset repo.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Local raw data directory.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing local archive/files.")
    parser.add_argument("--skip-extract", action="store_true", help="Do not extract downloaded archive.")
    parser.add_argument("--prepare", action="store_true", help="Convert CMeEE JSON files to BIO.")
    parser.add_argument("--sample", action="store_true", help="Create CPU-friendly BIO subset for training.")
    parser.add_argument("--sample-train", type=int, default=500, help="Training examples for sampled subset.")
    parser.add_argument("--sample-dev", type=int, default=150, help="Dev examples for sampled subset.")
    parser.add_argument("--sample-test", type=int, default=150, help="Test examples for sampled subset.")
    parser.add_argument("--train", action="store_true", help="Run NER training after conversion.")
    parser.add_argument("--evaluate", action="store_true", help="Run NER evaluation after training.")
    parser.add_argument("--all", action="store_true", help="Download, extract, prepare, train and evaluate.")
    args = parser.parse_args()

    target_dir = args.target_dir.resolve()
    if args.all:
        args.prepare = True
        args.sample = True
        args.train = True
        args.evaluate = True

    if not args.skip_download:
        openxlab = find_openxlab()
        if openxlab is None:
            print(
                "未找到 openxlab CLI。请先安装：\n"
                "  py -m pip install openxlab -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
            raise SystemExit(2)
        if not download_cmeee(openxlab, args.dataset_repo, args.source_path, target_dir):
            raise SystemExit(2)

    if not args.skip_extract:
        extract_archive(target_dir)

    if args.prepare:
        run_prepare(target_dir)
    if args.sample:
        run_sampling(args.sample_train, args.sample_dev, args.sample_test)
    if args.train:
        run_training()
    if args.evaluate:
        run_evaluation()

    print("\nDone.")


if __name__ == "__main__":
    main()
