import argparse
from pathlib import Path
from typing import Iterable, List, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


DATASET_ORDER = ["Human", "BioSNAP"]
METHOD_ORDER = ["TransformerCPI", "MolTrans", "DeepConv-DTI", "DrugBAN", "DrugLAMP", "TFU-DTI"]
METRIC_ORDER = ["auc", "aupr"]
METRIC_LABELS = {"auc": "AUC", "aupr": "AUPR"}
METRIC_COLORS = {"auc": "#bbcfe8", "aupr": "#659bca"}
TFU_METRIC_COLORS = {"auc": "#92bbd7", "aupr": "#376eae"}
GRID_COLOR = "#d8dde4"
EDGE_COLOR = "#202020"
TEXT_COLOR = "#111111"
BG_COLOR = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tools/table5_method_metrics.csv")
    parser.add_argument("--out", default="figures/table5_bars/table5_auc_aupr_grouped_human_biosnap_bar.svg")
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
    pad = max((vmax - vmin) * 0.20, 0.01)
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


def _svg_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = EDGE_COLOR,
    width: float = 1.0,
    fill_opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity:.2f}" stroke="{stroke}" stroke-width="{width:.1f}" />'
    )


def _ticks(y_min: float, y_max: float, n_ticks: int = 5) -> Sequence[float]:
    return np.linspace(y_min, y_max, n_ticks)


def build_svg(df: pd.DataFrame) -> str:
    width = 1800
    height = 860
    title_h = 92
    legend_h = 46
    margin_l = 98
    margin_r = 48
    margin_b = 210
    panel_gap = 60
    panel_w = (width - margin_l - margin_r - panel_gap) / 2.0
    panel_h = height - title_h - legend_h - margin_b - 36
    panel_top = title_h + legend_h + 26

    all_values = df[METRIC_ORDER].to_numpy(dtype=float).reshape(-1)
    global_y_min, global_y_max = _ylims(all_values)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG_COLOR}" />',
        _svg_text(width / 2.0, 46, "Table 5 Method Comparison by AUC and AUPR", size=30, weight="bold"),
    ]

    legend_y = 98
    legend_x = width / 2.0 - 130
    for idx, metric in enumerate(METRIC_ORDER):
        lx = legend_x + idx * 150
        parts.append(_svg_rect(lx, legend_y - 14, 26, 16, METRIC_COLORS[metric], stroke=EDGE_COLOR, width=0.9))
        parts.append(_svg_text(lx + 40, legend_y, METRIC_LABELS[metric], size=18, anchor="start"))

    for panel_idx, dataset in enumerate(DATASET_ORDER):
        sub = df[df["dataset"] == dataset].copy()
        x0 = margin_l + panel_idx * (panel_w + panel_gap)
        y0 = panel_top
        inner_l = 72
        inner_r = 24
        inner_t = 20
        inner_b = 132
        chart_x = x0 + inner_l
        chart_y = y0 + inner_t
        chart_w = panel_w - inner_l - inner_r
        chart_h = panel_h - inner_t - inner_b
        y_min, y_max = global_y_min, global_y_max

        parts.append(_svg_text(x0 + panel_w / 2.0, y0 - 8, dataset, size=28, weight="bold"))
        parts.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_text(x0 + 22, chart_y + chart_h / 2.0, "Score", size=20, weight="bold", rotate=-90))

        for tick in _ticks(y_min, y_max):
            ty = _map_y(float(tick), y_min, y_max, chart_y, chart_h)
            parts.append(_svg_line(chart_x, ty, chart_x + chart_w, ty, GRID_COLOR, 1.0, dash="4 4"))
            parts.append(_svg_text(chart_x - 12, ty + 6, f"{float(tick):.3f}", size=14, anchor="end"))

        n = len(sub)
        slot_w = chart_w / max(n, 1)
        group_w = slot_w * 0.76
        bar_gap = group_w * 0.10
        bar_w = (group_w - bar_gap) / 2.0

        for idx, (_, row) in enumerate(sub.iterrows()):
            is_tfu = str(row["method"]) == "TFU-DTI"
            group_x = chart_x + idx * slot_w + (slot_w - group_w) / 2.0
            for metric_idx, metric in enumerate(METRIC_ORDER):
                value = float(row[metric])
                bx = group_x + metric_idx * (bar_w + bar_gap)
                by = _map_y(value, y_min, y_max, chart_y, chart_h)
                bh = chart_y + chart_h - by
                fill = TFU_METRIC_COLORS[metric] if is_tfu else METRIC_COLORS[metric]
                stroke = EDGE_COLOR
                stroke_w = 1.2 if is_tfu else 1.0
                parts.append(_svg_rect(bx, by, bar_w, bh, fill, stroke=stroke, width=stroke_w))
                parts.append(
                    _svg_text(
                        bx + bar_w / 2.0,
                        max(chart_y - 4, by - 8),
                        f"{value:.4f}",
                        size=14 if is_tfu else 12,
                        weight="bold" if is_tfu else "normal",
                        rotate=-90,
                    )
                )

            label_x = group_x + group_w / 2.0
            label_y = chart_y + chart_h + 72
            label_weight = "bold" if is_tfu else "normal"
            label_fill = TEXT_COLOR
            label_size = 16 if is_tfu else 15
            parts.append(_svg_text(label_x, label_y, str(row["method"]), size=label_size, rotate=-28, weight=label_weight, fill=label_fill))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"dataset", "method", "auc", "aupr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = _ensure_ordered(df)

    svg = build_svg(df)
    out_path.write_text(svg, encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
