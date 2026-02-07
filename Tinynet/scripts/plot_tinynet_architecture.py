# Author: Yuyang Chen
# School: Zhejiang University
# Last Modified: 2026.2.7
# Description: Standalone script to draw a research-style TinyNet architecture graph.

"""Draw TinyNet architecture diagram and save it to the plots folder."""

from __future__ import annotations

import argparse
import os


# Purpose: Escape XML special characters for safe SVG text.
# Inputs: Raw string value.
# Outputs: XML-safe string.
# Assumptions: Input is a regular text value.
def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# Purpose: Build SVG rectangle element with centered text lines.
# Inputs: Geometry, label text lines, and style options.
# Outputs: SVG snippet as string.
# Assumptions: Coordinates are in canvas space.
def _svg_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    lines: list[str],
    fill: str,
    stroke: str = "#263238",
    stroke_w: int = 2,
    rx: int = 10,
    font_size: int = 16,
) -> str:
    parts = [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' ry='{rx}' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='{stroke_w}'/>"
    ]
    line_h = int(font_size * 1.3)
    total_h = line_h * len(lines)
    ty = y + (h - total_h) // 2 + font_size
    for line in lines:
        parts.append(
            f"<text x='{x + w // 2}' y='{ty}' text-anchor='middle' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='{font_size}' fill='#111827'>"
            f"{_xml_escape(line)}</text>"
        )
        ty += line_h
    return "\n".join(parts)


# Purpose: Build SVG arrow for data-flow connection.
# Inputs: Start and end coordinates.
# Outputs: SVG line snippet with arrow marker.
# Assumptions: Marker id `arrow-head` exists in defs.
def _svg_arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#455A64") -> str:
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='2.5' "
        "marker-end='url(#arrow-head)'/>"
    )


# Purpose: Draw TinyNet architecture and save as SVG file.
# Inputs: Output SVG path.
# Outputs: None.
# Side effects: Writes SVG file to disk.
def draw_tinynet_architecture_svg(out_svg_path: str) -> None:
    width, height = 2450, 1240
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs>",
        "<linearGradient id='bg-grad' x1='0%' y1='0%' x2='100%' y2='100%'>",
        "<stop offset='0%' stop-color='#F8FAFC'/>",
        "<stop offset='100%' stop-color='#EEF2F7'/>",
        "</linearGradient>",
        "<marker id='arrow-head' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'>",
        "<polygon points='0 0, 10 3.5, 0 7' fill='#455A64'/>",
        "</marker>",
        "</defs>",
        "<rect x='0' y='0' width='100%' height='100%' fill='url(#bg-grad)'/>",
        "<text x='1225' y='58' text-anchor='middle' font-family='Helvetica,Arial,sans-serif' "
        "font-size='34' font-weight='700' fill='#0F172A'>TinyNet Multi-Task Architecture (INT8 Deployment View)</text>",
        "<text x='1225' y='94' text-anchor='middle' font-family='Helvetica,Arial,sans-serif' "
        "font-size='18' fill='#334155'>Horizontal data flow with shared stem and dual task branches</text>",
    ]

    lines.append(
        _svg_rect(
            60,
            210,
            300,
            190,
            [
                "Input (C=16, L=320)",
                "c0: PPG",
                "c1-c4: ACC(mod,x,y,z)",
                "c5-c15: zero padding",
            ],
            fill="#DDEBFF",
        )
    )
    lines.append(
        _svg_rect(
            78,
            430,
            264,
            90,
            ["AF temporal stream", "uses PPG-only input"],
            fill="#FDECC8",
            stroke="#B45309",
            font_size=15,
        )
    )
    lines.append(
        _svg_rect(
            430,
            220,
            330,
            150,
            ["Shared Stem", "Conv1d(7) + BN + ReLU", "32 channels"],
            fill="#D9F5E5",
        )
    )
    lines.append(
        _svg_rect(
            420,
            430,
            350,
            160,
            ["Stem Internal"],
            fill="#ECFDF3",
            stroke="#16A34A",
            font_size=15,
        )
    )
    lines.append(
        _svg_rect(
            890,
            120,
            640,
            200,
            ["HR Branch (Regression)", "ResBlock1 -> ResBlock2 -> ResBlock3", "Head: 1x1 Conv -> GAP -> Dropout -> FC(106)"],
            fill="#FDE2E2",
            stroke="#B91C1C",
        )
    )
    lines.append(
        _svg_rect(
            890,
            360,
            640,
            180,
            ["HR Internal"],
            fill="#FFF1F2",
            stroke="#E11D48",
            font_size=15,
        )
    )
    lines.append(
        _svg_rect(
            890,
            620,
            780,
            270,
            [
                "AF Branch (Classification)",
                "Spatial path: 3x ResBlock + BN",
                "Temporal path: AvgPool -> AvgPool -> 2x ResBlock -> Data/Gate Conv -> Sigmoid gate -> GlobalPool",
                "Fusion: Add -> SE -> 1x1 Conv + BN + ReLU -> GAP -> Dropout -> FC(1)",
            ],
            fill="#FFECCF",
            stroke="#B45309",
        )
    )
    lines.append(
        _svg_rect(
            890,
            930,
            780,
            180,
            ["AF Internal"],
            fill="#FFF7ED",
            stroke="#C2410C",
            font_size=15,
        )
    )
    lines.append(
        _svg_rect(
            1830,
            180,
            390,
            120,
            ["HR Output", "HR logits (106 bins)", "BPM post-decoding"],
            fill="#E0F2FE",
            stroke="#0369A1",
        )
    )
    lines.append(
        _svg_rect(
            1830,
            705,
            390,
            120,
            ["AF Output", "AF probability (sigmoid)", "threshold=0.5 for report"],
            fill="#EDE9FE",
            stroke="#6D28D9",
        )
    )

    lines.append(_svg_arrow(360, 305, 430, 295))
    lines.append(_svg_arrow(760, 295, 890, 220))
    lines.append(_svg_arrow(760, 315, 890, 740))
    lines.append(_svg_arrow(1530, 220, 1830, 240))
    lines.append(_svg_arrow(1670, 780, 1830, 780))
    lines.append(_svg_arrow(760, 470, 890, 740))
    lines.append(
        "<text x='775' y='280' font-family='Helvetica,Arial,sans-serif' font-size='14' fill='#334155'>split</text>"
    )
    lines.append(
        "<text x='1880' y='705' font-family='Helvetica,Arial,sans-serif' font-size='14' fill='#334155'>parallel task outputs</text>"
    )

    lines.append(_svg_rect(442, 495, 86, 54, ["Conv7"], "#FFFFFF", stroke="#16A34A", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(554, 495, 70, 54, ["BN"], "#FFFFFF", stroke="#16A34A", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(650, 495, 96, 54, ["ReLU"], "#FFFFFF", stroke="#16A34A", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_arrow(528, 522, 554, 522, color="#16A34A"))
    lines.append(_svg_arrow(624, 522, 650, 522, color="#16A34A"))

    lines.append(_svg_rect(930, 430, 110, 54, ["ResBlk1"], "#FFFFFF", stroke="#E11D48", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1060, 430, 110, 54, ["ResBlk2"], "#FFFFFF", stroke="#E11D48", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1190, 430, 110, 54, ["ResBlk3"], "#FFFFFF", stroke="#E11D48", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1320, 430, 90, 54, ["1x1Conv"], "#FFFFFF", stroke="#E11D48", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1430, 430, 70, 54, ["GAP"], "#FFFFFF", stroke="#E11D48", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_arrow(1040, 457, 1060, 457, color="#E11D48"))
    lines.append(_svg_arrow(1170, 457, 1190, 457, color="#E11D48"))
    lines.append(_svg_arrow(1300, 457, 1320, 457, color="#E11D48"))
    lines.append(_svg_arrow(1410, 457, 1430, 457, color="#E11D48"))

    lines.append(_svg_rect(940, 995, 180, 58, ["Spatial stream"], "#FFFFFF", stroke="#C2410C", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1150, 995, 180, 58, ["Temporal stream"], "#FFFFFF", stroke="#C2410C", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1360, 995, 130, 58, ["Add + SE"], "#FFFFFF", stroke="#C2410C", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_rect(1520, 995, 130, 58, ["Head"], "#FFFFFF", stroke="#C2410C", stroke_w=1, rx=6, font_size=13))
    lines.append(_svg_arrow(1120, 1024, 1360, 1024, color="#C2410C"))
    lines.append(_svg_arrow(1330, 1024, 1360, 1024, color="#C2410C"))
    lines.append(_svg_arrow(1490, 1024, 1520, 1024, color="#C2410C"))

    lines.append("</svg>")

    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Purpose: Parse args and run architecture plotting.
# Inputs: CLI arguments.
# Outputs: None.
# Side effects: Generates SVG file in the plots directory.
def main() -> None:
    parser = argparse.ArgumentParser(description="Draw TinyNet architecture graph.")
    parser.add_argument("--out", type=str, default=None, help="Output SVG path. Default: <project>/plots/TinyNet_Model_Structure.svg")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_path = args.out or os.path.join(project_root, "plots", "TinyNet_Model_Structure.svg")
    draw_tinynet_architecture_svg(output_path)
    print(f"Saved architecture graph: {output_path}")


if __name__ == "__main__":
    main()
