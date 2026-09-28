"""Record docs/images/demo.gif from the demo page.

Needs the service running with the agreements loaded and a real LLM configured
(three questions, about 0.3 US cents with gpt-4.1-mini). Uses the installed
Microsoft Edge, so no browser download is needed.

    uv run python -m scripts.record_demo --url http://localhost:8000/
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "demo.gif"
VIEWPORT = {"width": 900, "height": 640}

# (agreement BOE id, question): a normal answer, a trick question and a referral.
SCENES = [
    ("BOE-A-2025-7766", "¿Cuántos días de vacaciones tengo?"),
    ("BOE-A-2025-7766", "¿Tengo derecho a un coche de empresa?"),
    ("BOE-A-2022-479", "¿Cuántas horas al año se trabajan en el metal?"),
]


class Recorder:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.frames: list[tuple[Image.Image, int]] = []

    def shot(self, ms: int) -> None:
        image = Image.open(io.BytesIO(self.page.screenshot()))
        self.frames.append((image.convert("RGB"), ms))

    def save(self, path: Path) -> None:
        # One shared palette keeps colours stable across frames and the file small.
        palette = self.frames[0][0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        images = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f, _ in self.frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=[ms for _, ms in self.frames],
            loop=0,
            optimize=True,
        )


def record(base_url: str, out: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        rec = Recorder(page)
        for agreement, question in SCENES:
            page.goto(base_url)
            page.select_option("#agreement", agreement)
            page.fill("#question", question)
            rec.shot(1800)
            page.click("#send")
            page.wait_for_selector("#result:not(.hidden)", timeout=120_000)
            rec.shot(4800)
        browser.close()
    rec.save(out)
    print(f"{out} · {len(rec.frames)} frames · {out.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--url", default="http://localhost:8000/")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    record(args.url, args.out)


if __name__ == "__main__":
    main()
