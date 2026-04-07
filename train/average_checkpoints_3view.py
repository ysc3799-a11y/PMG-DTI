import argparse
import json
import os
from typing import Dict, List

import torch


def parse_args():
    p = argparse.ArgumentParser(description='Average multiple checkpoint state_dicts saved by 3view training script.')
    p.add_argument('--ckpts', nargs='*', default=None, help='checkpoint paths to average')
    p.add_argument('--manifest', default=None, help='topk_checkpoints.json from a run directory')
    p.add_argument('--top_n', type=int, default=3, help='use first top_n entries from manifest')
    p.add_argument('--out', required=True, help='output averaged checkpoint path')
    return p.parse_args()


def load_ckpt_paths(args) -> List[str]:
    if args.ckpts:
        return [os.path.abspath(x) for x in args.ckpts]
    if args.manifest:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        paths = [os.path.abspath(x['path']) for x in entries[: max(1, int(args.top_n))]]
        return paths
    raise SystemExit('Provide either --ckpts or --manifest')


def average_state_dicts(paths: List[str]) -> Dict[str, torch.Tensor]:
    acc = None
    n = 0
    for path in paths:
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if acc is None:
            acc = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    acc[k] = v.detach().clone().float()
                else:
                    raise TypeError(f'Non-tensor entry in state_dict for key {k}')
        else:
            if acc.keys() != state.keys():
                raise ValueError('Checkpoint state_dict keys do not match')
            for k, v in state.items():
                acc[k].add_(v.detach().float())
        n += 1
    for k in acc:
        acc[k].div_(float(n))
    return acc


def main():
    args = parse_args()
    paths = load_ckpt_paths(args)
    if len(paths) == 0:
        raise SystemExit('No checkpoints found to average')
    avg_state = average_state_dicts(paths)

    meta_ckpt = torch.load(paths[0], map_location='cpu')
    thresholds = []
    vals = []
    epochs = []
    for p in paths:
        ck = torch.load(p, map_location='cpu')
        thresholds.append(float(ck.get('best_threshold', 0.5)))
        vals.append(float(ck.get('best_val', float('nan'))))
        epochs.append(int(ck.get('epoch', -1)))

    payload = dict(meta_ckpt)
    payload['model'] = avg_state
    payload['averaged_from'] = paths
    payload['averaged_num_ckpts'] = len(paths)
    payload['best_threshold'] = float(sorted(thresholds)[len(thresholds) // 2])
    payload['best_val_mean'] = float(sum(vals) / max(len(vals), 1))
    payload['averaged_epochs'] = epochs

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(payload, out)
    print(f'[OK] averaged {len(paths)} checkpoints -> {out}')
    print(json.dumps({'ckpts': paths, 'best_threshold': payload['best_threshold'], 'best_val_mean': payload['best_val_mean']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
