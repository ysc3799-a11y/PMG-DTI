import argparse
from pathlib import Path
from typing import Iterable, List, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


DATASET_ORDER = ["Human", "BioSNAP"]
K_ORDER = [1, 2, 3, 5]
METRIC_ORDER = ["auc", "aupr"]
METRIC_LABELS = {"auc": "AUC", "aupr": "AUPR"}
LINE_COLORS = {"Human": "#376eae", "BioSNAP": "#d9772b"}
MARKER_FILLS = {"Human": "#92bbd7", "BioSNAP": "#f2b277"}
GRID_COLOR = "#d8dde4"
EDGE_COLOR = "#202020"
TEXT_COLOR = "#111111"
BG_COLOR = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tools/table9_topk_metrics.csv")
    parser.add_argument("--out", default="figures/topk_analysis/table9_topk_auc_aupr_lines.svg")
    return parser.parse_args()


def _ensure_ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["k"] = pd.Categorical(df["k"], categories=K_ORDER, ordered=True)
    return df.sort_values(["dataset", "k"]).reset_index(drop=True)


def _ylims(values: Iterable[float]) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    pad = max((vmax - vmin) * 0.20, 0.01)
    lower = max(0.0, vmin - pad)
    upper = min(1.0, vmax + pad * 0.25)
    if upper <= lower:
        upper = min(1.0, lower + 0.02)
    return [lower, upper]


def _map_x(value: float, x_values: Sequence[float], left: float, width: float) -> float:
    xmin = float(min(x_values))
    xmax = float(max(x_values))
    ratio = (float(value) - xmin) / max(xmax - xmin, 1e-8)
    return left + width * ratio


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


def _svg_circle(cx: float, cy: float, r: float, fill: str, stroke: str = EDGE_COLOR, width: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}" />'
    )


def _svg_polyline(points: Sequence[tuple], stroke: str, width: float = 2.6) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width:.1f}" stroke-linejoin="round" stroke-linecap="round" />'


def _ticks(y_min: float, y_max: float, n_ticks: int = 5) -> Sequence[float]:
    return np.linspace(y_min, y_max, n_ticks)


def _label_offset(metric: str, dataset: str, idx: int) -> tuple[float, float, str]:
    dx = 0.0
    dy = -14.0 if dataset == "Human" else -10.0
    anchor = "middle"
    if idx == 0:
        dx = 16.0
        anchor = "start"
    elif idx == len(K_ORDER) - 1:
        dx = -16.0
        anchor = "end"
    if metric == "aupr" and dataset == "BioSNAP" and idx == 2:
        dy = 24.0
    elif metric == "auc" and dataset == "BioSNAP" and idx == 2:
        dy = -16.0
    elif dataset == "Human" and idx == 2:
        dy = -18.0
    return dx, dy, anchor


def build_svg(df: pd.DataFrame) -> str:
    width = 1780
    height = 820
    title_h = 92
    legend_h = 44
    margin_l = 98
    margin_r = 48
    margin_b = 148
    panel_gap = 62
    panel_w = (width - margin_l - margin_r - panel_gap) / 2.0
    panel_h = height - title_h - legend_h - margin_b - 28
    panel_top = title_h + legend_h + 24

    all_values = df[METRIC_ORDER].to_numpy(dtype=float).reshape(-1)
    global_y_min, global_y_max = _ylims(all_values)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG_COLOR}" />',
        _svg_text(width / 2.0, 46, "Top-K Pocket Analysis by AUC and AUPR", size=30, weight="bold"),
    ]

    legend_y = 98
    legend_x = width / 2.0 - 138
    for idx, dataset in enumerate(DATASET_ORDER):
        lx = legend_x + idx * 170
        line_x1 = lx
        line_x2 = lx + 40
        parts.append(_svg_line(line_x1, legend_y - 4, line_x2, legend_y - 4, LINE_COLORS[dataset], 3.2))
        parts.append(_svg_circle((line_x1 + line_x2) / 2.0, legend_y - 4, 5.4, MARKER_FILLS[dataset], stroke=LINE_COLORS[dataset], width=1.4))
        parts.append(_svg_text(lx + 54, legend_y, dataset, size=18, anchor="start"))

    for panel_idx, metric in enumerate(METRIC_ORDER):
        x0 = margin_l + panel_idx * (panel_w + panel_gap)
        y0 = panel_top
        inner_l = 88
        inner_r = 20
        inner_t = 20
        inner_b = 82
        chart_x = x0 + inner_l
        chart_y = y0 + inner_t
        chart_w = panel_w - inner_l - inner_r
        chart_h = panel_h - inner_t - inner_b
        y_min, y_max = global_y_min, global_y_max

        parts.append(_svg_text(x0 + panel_w / 2.0, y0 - 8, METRIC_LABELS[metric], size=28, weight="bold"))
        parts.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, EDGE_COLOR, 1.2))
        parts.append(_svg_text(x0 + 22, chart_y + chart_h / 2.0, "Score", size=24, weight="bold", rotate=-90))

        for tick in _ticks(y_min, y_max):
            ty = _map_y(float(tick), y_min, y_max, chart_y, chart_h)
            parts.append(_svg_line(chart_x, ty, chart_x + chart_w, ty, GRID_COLOR, 1.0, dash="4 4"))
            parts.append(_svg_text(chart_x - 12, ty + 6, f"{float(tick):.3f}", size=18, anchor="end"))

        for k in K_ORDER:
            kx = _map_x(float(k), K_ORDER, chart_x, chart_w)
            parts.append(_svg_text(kx, chart_y + chart_h + 46, f"K={k}", size=20))

        parts.append(_svg_text(chart_x + chart_w / 2.0, chart_y + chart_h + 90, "Top-K", size=22, weight="bold"))

        for dataset in DATASET_ORDER:
            sub = df[df["dataset"] == dataset].copy()
            points = []
            for _, row in sub.iterrows():
                k = float(row["k"])
                value = float(row[metric])
                px = _map_x(k, K_ORDER, chart_x, chart_w)
                py = _map_y(value, y_min, y_max, chart_y, chart_h)
                points.append((px, py))
            parts.append(_svg_polyline(points, LINE_COLORS[dataset], width=3.6))
            for idx, (px, py) in enumerate(points):
                value = float(sub.iloc[idx][metric])
                parts.append(_svg_circle(px, py, 6.2, MARKER_FILLS[dataset], stroke=LINE_COLORS[dataset], width=1.6))
                dx, dy, anchor = _label_offset(metric, dataset, idx)
                parts.append(
                    _svg_text(
                        px + dx,
                        py + dy,
                        f"{value:.4f}",
                        size=17,
                        weight="bold" if int(sub.iloc[idx]["k"]) == 3 else "normal",
                        anchor=anchor,
                    )
                )

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
    required = {"dataset", "k", "auc", "aupr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = _ensure_ordered(df)

    svg = build_svg(df)
    out_path.write_text(svg, encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
