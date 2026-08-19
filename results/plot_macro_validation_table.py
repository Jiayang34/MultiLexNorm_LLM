#!/usr/bin/env python3
"""Generate the poster table for macro-average validation results."""

import argparse
import csv
import statistics
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install Pillow"
    ) from exc


RESULTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESULTS_DIR.parent
QWEN_MACRO_PATH = RESULTS_DIR / "qwen35_pipeline_macro_summary.csv"
QWEN_FINAL_PATH = (
    RESULTS_DIR / "qwen35_final_new_detector_threshold_comparison.csv"
)
DEEPSEEK_PATH = RESULTS_DIR / "deepseek_val_base_newdetector_threshold.csv"
DEEPSEEK_FOUR_WAY_PATH = (
    PROJECT_ROOT
    / "llm_pipeline"
    / "data"
    / "val_threshold_runs"
    / "deepseek-v4-pro"
    / "deepseek_val_base_baseT_new_newT.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "macro_validation_results_table.png"
DEFAULT_DEEPSEEK_FOUR_WAY_OUTPUT = (
    RESULTS_DIR / "deepseek_val_base_baseT_new_newT_table.png"
)

REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def mean(rows, column):
    return statistics.fmean(float(row[column]) for row in rows)


def load_qwen_results():
    macro_rows = read_csv(QWEN_MACRO_PATH)
    overall = next(
        row for row in macro_rows if row["stage"] == "overall_final"
    )

    baseline_err = float(overall["old_macro_err"])
    baseline_f1 = float(overall["old_macro_f1"])
    new_detector_err = float(overall["new_macro_err"])
    new_detector_f1 = float(overall["new_macro_f1"])

    final_rows = [
        row
        for row in read_csv(QWEN_FINAL_PATH)
        if row["system"] == "new_detector_with_threshold"
    ]
    optimized_err = baseline_err + mean(
        final_rows, "overall_final_delta_err"
    )
    optimized_f1 = baseline_f1 + mean(
        final_rows, "overall_final_delta_f1"
    )

    return {
        "baseline": (baseline_err, baseline_f1),
        "new_detector": (new_detector_err, new_detector_f1),
        "optimized": (optimized_err, optimized_f1),
    }


def load_deepseek_results():
    rows = read_csv(DEEPSEEK_PATH)
    return {
        "baseline": (mean(rows, "base_err"), mean(rows, "base_f1")),
        "new_detector": (
            mean(rows, "new_detector_err"),
            mean(rows, "new_detector_f1"),
        ),
        "optimized": (
            mean(rows, "threshold_search_err"),
            mean(rows, "threshold_search_f1"),
        ),
    }


def load_deepseek_four_way_results(path):
    rows = read_csv(path)
    return {
        "base_default": (
            mean(rows, "base_default_err"),
            mean(rows, "base_default_f1"),
        ),
        "base_optimized": (
            mean(rows, "base_search_err"),
            mean(rows, "base_search_f1"),
        ),
        "new_default": (
            mean(rows, "new_default_err"),
            mean(rows, "new_default_f1"),
        ),
        "new_optimized": (
            mean(rows, "new_search_err"),
            mean(rows, "new_search_f1"),
        ),
    }


def percent(value):
    return f"{100 * value:.2f}"


def improvement(final, baseline):
    return f"{100 * (final - baseline):+.2f}"


def build_table_rows(qwen, deepseek):
    return [
        [
            "Original detector + default thresholds",
            percent(qwen["baseline"][0]),
            percent(qwen["baseline"][1]),
            percent(deepseek["baseline"][0]),
            percent(deepseek["baseline"][1]),
        ],
        [
            "Length-aware detector + default thresholds",
            percent(qwen["new_detector"][0]),
            percent(qwen["new_detector"][1]),
            percent(deepseek["new_detector"][0]),
            percent(deepseek["new_detector"][1]),
        ],
        [
            "Length-aware detector + optimized thresholds",
            percent(qwen["optimized"][0]),
            percent(qwen["optimized"][1]),
            percent(deepseek["optimized"][0]),
            percent(deepseek["optimized"][1]),
        ],
        [
            "Final improvement over baseline",
            improvement(qwen["optimized"][0], qwen["baseline"][0]),
            improvement(qwen["optimized"][1], qwen["baseline"][1]),
            improvement(
                deepseek["optimized"][0], deepseek["baseline"][0]
            ),
            improvement(
                deepseek["optimized"][1], deepseek["baseline"][1]
            ),
        ],
    ]


def build_deepseek_four_way_rows(results):
    return [
        [
            "Original detector + default thresholds",
            percent(results["base_default"][0]),
            percent(results["base_default"][1]),
        ],
        [
            "Original detector + optimized thresholds",
            percent(results["base_optimized"][0]),
            percent(results["base_optimized"][1]),
        ],
        [
            "Length-aware detector + default thresholds",
            percent(results["new_default"][0]),
            percent(results["new_default"][1]),
        ],
        [
            "Length-aware detector + optimized thresholds",
            percent(results["new_optimized"][0]),
            percent(results["new_optimized"][1]),
        ],
        [
            "Final improvement over baseline",
            improvement(
                results["new_optimized"][0],
                results["base_default"][0],
            ),
            improvement(
                results["new_optimized"][1],
                results["base_default"][1],
            ),
        ],
    ]


def centered_text(draw, bounds, text, font, fill, multiline=False):
    left, top, right, bottom = bounds
    if multiline:
        box = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=6, align="center"
        )
    else:
        box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    position = (
        (left + right - width) / 2,
        (top + bottom - height) / 2 - box[1],
    )
    if multiline:
        draw.multiline_text(
            position,
            text,
            font=font,
            fill=fill,
            spacing=6,
            align="center",
        )
    else:
        draw.text(position, text, font=font, fill=fill)


def render_table(rows, output, dpi):
    width, height = 3000, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(BOLD_FONT, 70)
    header_font = ImageFont.truetype(BOLD_FONT, 43)
    body_font = ImageFont.truetype(REGULAR_FONT, 45)
    body_bold = ImageFont.truetype(BOLD_FONT, 45)
    blue = (12, 105, 180)
    dark = (30, 43, 56)
    grid = (196, 207, 217)
    very_light_blue = (244, 249, 253)
    highlight = (224, 241, 255)
    gain = (229, 246, 235)
    green = (0, 103, 63)

    title = "Macro-average Validation Results across 12 Languages (%)"
    centered_text(
        draw, (0, 35, width, 135), title, title_font, dark
    )

    left = 90
    top = 190
    column_widths = [1400, 375, 375, 375, 375]
    row_heights = [170, 175, 175, 175, 175]
    xs = [left]
    ys = [top]
    for column_width in column_widths:
        xs.append(xs[-1] + column_width)
    for row_height in row_heights:
        ys.append(ys[-1] + row_height)

    headers = [
        "Configuration",
        "Qwen\nERR ↑",
        "Qwen\nF1 ↑",
        "DeepSeek\nERR ↑",
        "DeepSeek\nF1 ↑",
    ]

    draw.rectangle((left, ys[0], xs[-1], ys[1]), fill=blue)
    row_backgrounds = [
        (255, 255, 255),
        very_light_blue,
        highlight,
        gain,
    ]
    for row_index in range(4):
        draw.rectangle(
            (left, ys[row_index + 1], xs[-1], ys[row_index + 2]),
            fill=row_backgrounds[row_index],
        )

    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill=grid, width=3)
    for y in ys:
        draw.line((left, y, xs[-1], y), fill=grid, width=3)

    for column_index, header in enumerate(headers):
        bounds = (
            xs[column_index],
            ys[0],
            xs[column_index + 1],
            ys[1],
        )
        if column_index == 0:
            box = draw.textbbox((0, 0), header, font=header_font)
            text_height = box[3] - box[1]
            draw.text(
                (
                    xs[column_index] + 32,
                    (ys[0] + ys[1] - text_height) / 2 - box[1],
                ),
                header,
                font=header_font,
                fill="white",
            )
        else:
            centered_text(
                draw,
                bounds,
                header,
                header_font,
                "white",
                multiline=True,
            )

    for row_index, row in enumerate(rows):
        font = body_bold if row_index == 3 else body_font
        for column_index, text in enumerate(row):
            bounds = (
                xs[column_index],
                ys[row_index + 1],
                xs[column_index + 1],
                ys[row_index + 2],
            )
            color = green if row_index == 3 and column_index > 0 else dark
            if column_index == 0:
                box = draw.textbbox((0, 0), text, font=font)
                text_height = box[3] - box[1]
                draw.text(
                    (
                        xs[column_index] + 32,
                        (
                            ys[row_index + 1]
                            + ys[row_index + 2]
                            - text_height
                        )
                        / 2
                        - box[1],
                    ),
                    text,
                    font=font,
                    fill=color,
                )
            else:
                centered_text(draw, bounds, text, font, color)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(dpi, dpi), optimize=True)


def render_deepseek_four_way_table(rows, output, dpi):
    width, height = 2300, 1180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(BOLD_FONT, 64)
    header_font = ImageFont.truetype(BOLD_FONT, 39)
    body_font = ImageFont.truetype(REGULAR_FONT, 39)
    body_bold = ImageFont.truetype(BOLD_FONT, 39)
    blue = (12, 105, 180)
    dark = (30, 43, 56)
    grid = (196, 207, 217)
    very_light_blue = (244, 249, 253)
    highlight = (224, 241, 255)
    gain = (229, 246, 235)
    green = (0, 103, 63)

    title = "DeepSeek Validation Results across 12 Languages (%)"
    centered_text(
        draw, (0, 35, width, 135), title, title_font, dark
    )

    left = 80
    top = 185
    column_widths = [1420, 390, 390]
    row_heights = [160, 155, 155, 155, 155, 155]
    xs = [left]
    ys = [top]
    for column_width in column_widths:
        xs.append(xs[-1] + column_width)
    for row_height in row_heights:
        ys.append(ys[-1] + row_height)

    headers = [
        "Configuration",
        "DeepSeek\nERR ↑",
        "DeepSeek\nF1 ↑",
    ]

    draw.rectangle((left, ys[0], xs[-1], ys[1]), fill=blue)
    row_backgrounds = [
        (255, 255, 255),
        very_light_blue,
        (255, 255, 255),
        highlight,
        gain,
    ]
    for row_index in range(5):
        draw.rectangle(
            (left, ys[row_index + 1], xs[-1], ys[row_index + 2]),
            fill=row_backgrounds[row_index],
        )

    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill=grid, width=3)
    for y in ys:
        draw.line((left, y, xs[-1], y), fill=grid, width=3)

    for column_index, header in enumerate(headers):
        bounds = (
            xs[column_index],
            ys[0],
            xs[column_index + 1],
            ys[1],
        )
        if column_index == 0:
            box = draw.textbbox((0, 0), header, font=header_font)
            text_height = box[3] - box[1]
            draw.text(
                (
                    xs[column_index] + 30,
                    (ys[0] + ys[1] - text_height) / 2 - box[1],
                ),
                header,
                font=header_font,
                fill="white",
            )
        else:
            centered_text(
                draw,
                bounds,
                header,
                header_font,
                "white",
                multiline="\n" in header,
            )

    for row_index, row in enumerate(rows):
        font = body_bold if row_index == 4 else body_font
        for column_index, text in enumerate(row):
            bounds = (
                xs[column_index],
                ys[row_index + 1],
                xs[column_index + 1],
                ys[row_index + 2],
            )
            color = green if row_index == 4 and column_index > 0 else dark
            if column_index == 0:
                box = draw.textbbox((0, 0), text, font=font)
                text_height = box[3] - box[1]
                draw.text(
                    (
                        xs[column_index] + 30,
                        (
                            ys[row_index + 1]
                            + ys[row_index + 2]
                            - text_height
                        )
                        / 2
                        - box[1],
                    ),
                    text,
                    font=font,
                    fill=color,
                )
            else:
                centered_text(draw, bounds, text, font, color)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(dpi, dpi), optimize=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the macro-average validation results table."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--deepseek-four-way",
        action="store_true",
        help="Render DeepSeek 1/2/3/4 validation comparison table.",
    )
    parser.add_argument(
        "--deepseek-four-way-path",
        type=Path,
        default=DEEPSEEK_FOUR_WAY_PATH,
        help="Input CSV for DeepSeek 1/2/3/4 validation comparison.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output = args.output
    if args.deepseek_four_way:
        if output == DEFAULT_OUTPUT:
            output = DEFAULT_DEEPSEEK_FOUR_WAY_OUTPUT
        rows = build_deepseek_four_way_rows(
            load_deepseek_four_way_results(args.deepseek_four_way_path)
        )
        render_deepseek_four_way_table(rows, output, args.dpi)
        print(f"Wrote {output}")
        return

    rows = build_table_rows(load_qwen_results(), load_deepseek_results())
    render_table(rows, output, args.dpi)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
