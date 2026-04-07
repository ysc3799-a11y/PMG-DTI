import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def mirror_processed_path(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    if "interim" not in parts:
        return None
    i = parts.index("interim")
    parts[i] = "processed"
    return Path(*parts)


def resolve_repr_layers(num_layers: int, pool: str):
    num_layers = int(num_layers)
    if pool == "last":
        return [num_layers]
    if pool == "last4mean":
        start = max(1, num_layers - 3)
        return list(range(start, num_layers + 1))
    raise ValueError(f"Unsupported pool mode: {pool}")


def extract_batch_embeddings(model, toks, repr_layers, labels, batch, max_len, dtype, pool):
    out = model(toks, repr_layers=repr_layers, return_contacts=False)
    outputs = {}
    if pool == "last":
        reps = out["representations"][repr_layers[0]]  # [B, T, D]
        for bi, pid in enumerate(labels):
            seq_len = min(len(batch[bi][1]), int(max_len))
            emb = reps[bi, 1 : 1 + seq_len].detach().cpu()
            emb = emb.to(torch.float16 if dtype == "fp16" else torch.float32)
            outputs[pid] = emb
        return outputs

    if pool == "last4mean":
        for bi, pid in enumerate(labels):
            seq_len = min(len(batch[bi][1]), int(max_len))
            stacked = []
            for li in repr_layers:
                rep = out["representations"][li][bi, 1 : 1 + seq_len]
                stacked.append(rep)
            emb = torch.stack(stacked, dim=0).mean(dim=0).detach().cpu()
            emb = emb.to(torch.float16 if dtype == "fp16" else torch.float32)
            outputs[pid] = emb
        return outputs

    raise ValueError(f"Unsupported pool mode: {pool}")


def main():
    ap = argparse.ArgumentParser(description="Extract ESM2 residue-level token embeddings.")
    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--seq_col", default="Protein")
    ap.add_argument("--model_name", default="esm2_t6_8M_UR50D")
    ap.add_argument("--pool", default="last", choices=["last", "last4mean"], help="Representation pooling across ESM layers.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--out_pt", default="data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt")
    ap.add_argument("--no_mirror_processed", action="store_true", help="Disable auto mirror to data/processed/*")
    args = ap.parse_args()

    try:
        import esm
    except Exception as e:
        raise RuntimeError(
            "Package 'fair-esm' is required. Install with: pip install fair-esm"
        ) from e

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)
    df = pd.read_csv(in_csv)
    if args.pid_col not in df.columns or args.seq_col not in df.columns:
        raise ValueError(f"missing columns: {args.pid_col}, {args.seq_col}")

    prot = df[[args.pid_col, args.seq_col]].drop_duplicates(args.pid_col)
    pairs = [(str(r[args.pid_col]), str(r[args.seq_col])[: int(args.max_len)]) for _, r in prot.iterrows()]
    if not pairs:
        raise RuntimeError("No protein sequence found.")

    loader_fn = getattr(esm.pretrained, args.model_name, None)
    if loader_fn is None:
        raise ValueError(f"Unsupported ESM model_name: {args.model_name}")

    model, alphabet = loader_fn()
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layers = resolve_repr_layers(model.num_layers, args.pool)

    out_tokens = {}
    with torch.no_grad():
        for batch in chunks(pairs, int(args.batch_size)):
            labels = [x[0] for x in batch]
            _, _, toks = batch_converter(batch)
            toks = toks.to(device)
            batch_embs = extract_batch_embeddings(
                model=model,
                toks=toks,
                repr_layers=repr_layers,
                labels=labels,
                batch=batch,
                max_len=args.max_len,
                dtype=args.dtype,
                pool=args.pool,
            )
            out_tokens.update(batch_embs)

    any_emb = next(iter(out_tokens.values()))
    payload = {
        "dim": int(any_emb.shape[1]),
        "tokens": out_tokens,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "pool_mode": args.pool,
        "repr_layers": repr_layers,
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    print(f"[OK] saved: {out_pt}")

    if not args.no_mirror_processed:
        m_pt = mirror_processed_path(out_pt)
        if m_pt is not None and m_pt != out_pt:
            m_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, m_pt)
            print(f"[OK] mirrored: {m_pt}")

    print(
        f"[INFO] proteins={len(out_tokens)} dim={payload['dim']} pool={payload['pool_mode']} layers={payload['repr_layers']}"
    )


if __name__ == "__main__":
    main()
