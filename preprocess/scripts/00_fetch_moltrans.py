import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


EXPECTED_SHA256 = {
    "human": "ccda331c2697dbd12c96e5c4e855e98f162b9766086d945913bd0b90154aede3",
    "biosnap": "b0155eb273f0ceea2b883c2eb66108365e679c1d66c96cef413133257f991585",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(path: Path, dataset: str):
    observed = sha256_file(path)
    expected = EXPECTED_SHA256[dataset]
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {dataset}: expected {expected}, observed {observed}. "
            "The file is not the byte-identical table used by PMG-DTI."
        )
    print(f"[OK] SHA-256 verified: {observed}")


def run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(
        description="Fetch the Human or BioSNAP interaction table distributed by DrugBAN."
    )
    ap.add_argument("--dataset", choices=["human", "biosnap"], default="human")
    ap.add_argument("--src_csv", default="", help="Optional local full.csv path.")
    ap.add_argument("--repo_url", default="https://github.com/peizhenbai/DrugBAN.git")
    ap.add_argument("--repo_dir", default="external/DrugBAN")
    ap.add_argument(
        "--dataset_file_rel",
        default="",
        help="Optional path inside the source repository; defaults to datasets/<dataset>/full.csv.",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional output path; a dataset-specific repository path is used by default.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_file_rel = args.dataset_file_rel or f"datasets/{args.dataset}/full.csv"
    default_out = (
        "data/raw/human/moltrans/full.csv"
        if args.dataset == "human"
        else "data/raw/biosnap/full.csv"
    )
    out_csv = Path(args.out_csv or default_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.src_csv:
        src = Path(args.src_csv)
        if not src.is_file():
            raise FileNotFoundError(f"src_csv not found: {src}")
        if out_csv.exists() and not args.force:
            print(f"[SKIP] exists: {out_csv}")
            verify_dataset(out_csv, args.dataset)
            return
        shutil.copy2(src, out_csv)
        print(f"[OK] copied local csv -> {out_csv}")
        verify_dataset(out_csv, args.dataset)
        return

    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", args.repo_url, str(repo_dir)])
    else:
        print(f"[INFO] repo exists: {repo_dir}")

    src = repo_dir / dataset_file_rel
    if not src.is_file():
        raise FileNotFoundError(f"dataset file not found in repo: {src}")

    if out_csv.exists() and not args.force:
        print(f"[SKIP] exists: {out_csv}")
        verify_dataset(out_csv, args.dataset)
        return
    shutil.copy2(src, out_csv)
    print(f"[OK] fetched -> {out_csv}")
    verify_dataset(out_csv, args.dataset)


if __name__ == "__main__":
    main()

