import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description="Fetch MolTrans dataset file (Human/full.csv).")
    ap.add_argument("--src_csv", default="", help="Optional local full.csv path.")
    ap.add_argument("--repo_url", default="https://github.com/kexinhuang12345/MolTrans.git")
    ap.add_argument("--repo_dir", default="external/MolTrans")
    ap.add_argument("--dataset_file_rel", default="dataset/human/full.csv")
    ap.add_argument("--out_csv", default="data/raw/human/moltrans/full.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.src_csv:
        src = Path(args.src_csv)
        if not src.is_file():
            raise FileNotFoundError(f"src_csv not found: {src}")
        if out_csv.exists() and not args.force:
            print(f"[SKIP] exists: {out_csv}")
            return
        shutil.copy2(src, out_csv)
        print(f"[OK] copied local csv -> {out_csv}")
        return

    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", args.repo_url, str(repo_dir)])
    else:
        print(f"[INFO] repo exists: {repo_dir}")

    src = repo_dir / args.dataset_file_rel
    if not src.is_file():
        raise FileNotFoundError(f"dataset file not found in repo: {src}")

    if out_csv.exists() and not args.force:
        print(f"[SKIP] exists: {out_csv}")
        return
    shutil.copy2(src, out_csv)
    print(f"[OK] fetched -> {out_csv}")


if __name__ == "__main__":
    main()

