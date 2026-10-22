"""Plota as curvas do log CSV sem exigir dependência gráfica adicional."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_csv")
    parser.add_argument("output_png")
    args = parser.parse_args()
    with Path(args.log_csv).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = ("L_ppo", "L_semantic", "L_inverse_dynamics", "L_hero_motion", "success_rate")
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    colors = ((30,90,180),(180,50,50),(30,140,80),(160,90,180),(220,140,20))
    left, top, width, height = 80, 40, 1070, 590
    draw.rectangle((left, top, left+width, top+height), outline=(40,40,40), width=2)
    for tick in range(6):
        y = top + height - tick * height // 5
        draw.line((left, y, left+width, y), fill=(225,225,225))
        draw.text((15, y-7), f"{tick/5:.1f}", fill=(50,50,50))
    for column, color in zip(columns, colors, strict=True):
        values = [float(row[column]) for row in rows]
        low, high = min(values, default=0), max(values, default=1)
        spread = max(1e-9, high-low)
        points = [(left + index * width // max(1, len(values)-1),
                   top + height - int((value-low)/spread*height))
                  for index, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text((left + 12 + columns.index(column)*210, top+8), column, fill=color)
    image.save(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
