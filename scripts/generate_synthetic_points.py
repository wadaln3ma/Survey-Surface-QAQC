import csv, math, random
from pathlib import Path
import typer

app = typer.Typer(add_completion=False)

@app.command()
def main(out: Path = typer.Option(..., help='CSV output path')):
    random.seed(42)
    cx, cy = 46.675, 24.715  # Riyadh-ish
    pts = []
    for _ in range(1500):
        dx = (random.random() - 0.5) * 0.02
        dy = (random.random() - 0.5) * 0.02
        x = cx + dx
        y = cy + dy
        r = math.sqrt((dx*111_000)**2 + (dy*111_000)**2)  # meters approx
        z = 600 + 30*math.cos(r/220) + random.random()*0.8
        pts.append((x, y, z))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['x','y','z']); w.writerows(pts)
    print(f'Wrote {len(pts)} points to {out}')

if __name__ == '__main__':
    app()
