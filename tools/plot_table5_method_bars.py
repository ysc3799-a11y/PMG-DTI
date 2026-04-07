import argparse
from pathlib import Path
from typing import Iterable, List, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


DATASET_ORDER = ["Human", "BioSNAP"]
METHOD_ORDER = ["TransformerCPI", "MolTrans", "DeepConv-DTI", "DrugBAN", "DrugLAMP", "TFU-DTI"]
METRIC_LABELS = {"auc": "AUC", "aupr": "AUPR"}
BASE_COLOR = "#c9d2dd"
HIGHLIGHT_COLOR = "#1f5aa6"
GRID_COLOR = "#d8dde4"
EDGE_COLOR = "#202020"
TEXT_COLOR = "#111111"
BG_COLOR = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="tools/table5_method_metrics.csv",
        help="CSV with columns: dataset, method, auc, aupr",
    )
    parser.add_argument(
        "--out_dir",
        default="figures/table5_bars",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auc", "aupr"],
        choices=["auc", "aupr"],
        help="Metrics to plot",
    )
    return parser.parse_args()


def _ensure_ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["dataset", "method"]).reset_index(drop=True)


def _ylims(values: Iterable[float]) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    pad = max((vmax - vmin) * 0.20, 0.005)
    lower = max(0.0, vmin - pad)
    upper = min(1.0, vmax + pad * 0.35)
    if upper <= lower:
        upper = min(1.0, lower + 0.02)
    return [lower, upper]


def _map_y(value: float, y_min: float, y_max: float, top: float, height: float) -> float:
    ratio = (float(value) - y_min) / max(y_max - y_min, 1e-8)
    return top + height * (1.0 - ratio)


def _svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    weight: str = "normal",
    anchor: str = "middle",
    fill: str = TEXT_COLOR,
    rotate: float = 0.0,
) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'font-size="{size}"',
        f'font-family="Times New Roman, Times, serif"',
        f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
        f'fill="{fill}"',
    ]
    if abs(float(rotate)) > 1e-8:
        attrs.append(f'transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"')
    return f"<text {' '.join(attrs)}>{escape(str(text))}</text>"


def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr} />'
    )


def _svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = EDGE_COLOR, width: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}" />'
    )


def _ticks(y_min: float, y_max: float, n_ticks: int = 5) -> Sequence[float]:
    return np.linspace(y_min, y_max, n_ticks)


def build_metric_svg(df: pd.DataFrame, metric: str) -> str:
    label = METRIC_LABELS[metric]
    width = 1400
    height = 620
    title_h = 70
    margin_l = 80
    margin_r = 40
    margin_b = 150
    panel_gap = 46
    panel_w = (width - margin_l - margin_r - panel_gap) / 2.0
    panel_h = height - title_h - margin_b - 40
    panel_top = title_h + 35

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG_COLOR}" />',
        _svg_text(width / 2.0, 38, f"Table 5 Method Comparison by {label}", size=24, weight="bold"),
    ]

    for panel_idx, dataset in enumerate(DATASET_ORDER):
        sub = df[df["dataset"] == dataset].copy()
        x0 = margin_l + panel_idx * (panel_w + panel_gap)
        y0 = panel_top
        inner_l = 60
        inner_r = 18
        inner_t = 16
        inner_b = 96
        chart_x = x0 + inner_l
        chart_y = y0 + inner_t
        chart_w = panel_w - inner_l - inner_r
        chart_h = panel_h - inner_t - inner_b
        y_min, y_max = _ylims(sub[metric].to_numpy(dtype=float))

        parts.append(_svg_text(x0 + panel_w / 2.0, y0 - 6, dataset, size=22, weight="bold"))
        parts.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_text(x0 + 18, chart_y + chart_h / 2.0, label, size=16, weight="bold", rotate=-90))

        tick_values = list(_ticks(y_min, y_max))
        for tick in tick_values:
            ty = _map_y(float(tick), y_min, y_max, chart_y, chart_h)
            parts.append(_svg_line(chart_x, ty, chart_x + chart_w, ty, GRID_COLOR, 1.0, dash="4 4"))
            parts.append(_svg_text(chart_x - 10, ty + 5, f"{float(tick):.3f}", size=11, anchor="end"))

        n = len(sub)
        slot_w = chart_w / max(n, 1)
        bar_w = slot_w * 0.62

        for idx, (_, row) in enumerate(sub.iterrows()):
            value = float(row[metric])
            bx = chart_x + idx * slot_w + (slot_w - bar_w) / 2.0
            by = _map_y(value, y_min, y_max, chart_y, chart_h)
            bh = chart_y + chart_h - by
            color = HIGHLIGHT_COLOR if str(row["method"]) == "TFU-DTI" else BASE_COLOR
            parts.append(_svg_rect(bx, by, bar_w, bh, color))
            parts.append(_svg_text(bx + bar_w / 2.0, max(chart_y - 4, by - 8), f"{value:.4f}", size=11, rotate=-90))
            label_x = bx + bar_w / 2.0
            label_y = chart_y + chart_h + 52
            parts.append(_svg_text(label_x, label_y, str(row["method"]), size=12, rotate=-28))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"dataset", "method", "auc", "aupr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = _ensure_ordered(df)

    for metric in args.metrics:
        svg = build_metric_svg(df, metric)
        out_path = out_dir / f"table5_{metric}_human_biosnap_bar.svg"
        out_path.write_text(svg, encoding="utf-8")
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
