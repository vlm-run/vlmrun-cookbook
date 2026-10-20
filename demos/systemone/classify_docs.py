# /// script
# requires-python = ">=3.11"
# dependencies = ["vlmrun[typesafe,ws]>=0.9.2", "diskcache>=5.6", "av>=14"]
# ///
"""Classify OmniDocBench pages by document type: one calibrated System One read per page.

    uv run classify_docs.py --n-per-category 3 --mp4    # 30 pages and a reel
    uv run classify_docs.py --all --ws -c 8             # all 1651 pages over one WS session

Ground truth is OmniDocBench's own page_attribute.data_source (10 types). On all 1651 pages: 57.7% top1 /
71.1% top2 / 80.5% top3 at p50 133 ms, against always-book 16.7% and uniform 10%. Raw confidence is
overconfident (ECE 0.262); one temperature, T=2.9, takes it to 0.041 without changing the ranking.

Downloads and reads are cached on disk, so re-running a command (say, to re-render the reel) is free.
"""

import hashlib
import io
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from fractions import Fraction
from functools import cache, partial
from itertools import accumulate, count
from pathlib import Path
from typing import Annotated

import av
import typer
from cachetools import cached
from diskcache import Cache
from PIL import Image, ImageDraw, ImageFont
from rich import print
from rich.markup import escape
from rich.progress import track
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from vlmrun.client import VLMRun
from vlmrun.client.systemone import usage_of
from vlmrun.common.mime import data_url

DATASET = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main"
GEIST = "https://raw.githubusercontent.com/vercel/geist-font/v1.7.2/fonts/GeistMono/ttf/GeistMono-Regular.ttf"
MODEL = os.getenv("TYPESAFE_DEFAULT_MODEL", "google/diffusiongemma-26b-a4b-it")
DISK = Cache(os.getenv("SYSTEMONE_CACHE", Path.home() / ".cache" / "vlmrun-systemone"))
HOLD_MS = 1600  # how long a --slow-intro page stays on screen

# Each type names what it is NOT: the descriptions were rewritten twice from full-dataset confusion matrices.
CATEGORIES = {
    "book": "a page from a printed BOOK or novel -- continuous narrative prose in paragraphs, on clean white modern "
    "paper, ordinary black text, no illustrations or color. NOT historical_document (this is not aged, yellowed or "
    "archaic-looking -- if the paper or type looks old, pick that instead), NOT note (this is machine-printed, never "
    "handwritten), NOT colorful_textbook or PPT2PDF (no diagrams, callout boxes, or large areas of empty space), NOT "
    "academic_literature (no citations, references or abstract) and NOT research_report (no charts, tables, logo, or "
    "formal report headings).",
    "PPT2PDF": "a presentation slide exported to PDF -- a big title, a few short bullet points, and a lot of empty "
    "space. NOT colorful_textbook: a slide has far less text and far more blank space than a textbook page, which is "
    "dense with running text around its graphics.",
    "academic_literature": "a page from an academic paper or journal article -- look for inline citations like "
    "'(Smith, 2020)' or '[12]', an abstract, equations, or a references list; often two narrow text columns. NOT book "
    "(a book never cites sources) and NOT research_report (no corporate logo, title page, or executive summary).",
    "exam_paper": "a page from a school exam or test -- numbered questions (1, 2, 3...) each followed by BLANK space "
    "or lettered options (A, B, C, D) waiting to be answered. NOT note: notes are someone's filled-in answers or "
    "summaries, not blank questions.",
    "colorful_textbook": "a page from an illustrated K-12 textbook -- saturated, cartoonish colors, diagrams, boxed "
    "callouts or sidebars, large friendly fonts, printed for children. NOT note (this is professionally printed, never "
    "handwritten) and NOT PPT2PDF (this has much more running text than a slide).",
    "newspaper": "a page from a newspaper -- FOUR OR MORE narrow print columns of small text packed tightly together, "
    "a masthead or dateline. NOT magazine: a newspaper's columns are narrower and its print smaller, with no large "
    "glossy photo dominating the page.",
    "magazine": "a page from a glossy consumer magazine -- ONE OR TWO large photos or illustrations that dominate the "
    "page, plus a large stylized headline or pull quote, in fewer and wider columns than a newspaper. NOT "
    "research_report (its imagery is a photo for a general reader, not a chart or data table) and NOT newspaper "
    "(bigger photos, fewer, wider columns).",
    "research_report": "a page from a corporate or financial research report. Only pick this if you can see a "
    "company/institution logo, a cover or title page naming the report, an 'Executive Summary', or a financial "
    "chart/graph with currency or percentage figures. A page with only a 'Results' or 'Methodology' heading and no "
    "logo or financial chart is academic_literature, not this -- academic papers use those same section headings. "
    "Plain paragraphs of formal-sounding text alone are book or academic_literature, not this -- do not use this as "
    "your default guess.",
    "note": "a page of handwritten or informally typed personal notes -- visibly irregular handwriting, underlines, "
    "arrows, or a casual layout no printing press or word processor produced.",
    "historical_document": "a scanned historical or archival document -- visibly aged or yellowed paper, ink "
    "bleed-through, or an old typeface or handwriting style, in an archival context rather than a personal notebook.",
}
STATE = "Look at this single page image and decide which of the ten document types it is."
QUESTIONS = {
    "document_type": {"type": "choice", "instructions": "What kind of document page is this?", "options": CATEGORIES}
}
PROMPT = hashlib.sha256(json.dumps([STATE, QUESTIONS]).encode()).hexdigest()[:12]  # a reworded prompt re-reads


@cached(DISK, key=lambda url: f"fetch/{url}")
@retry(stop=stop_after_attempt(4), wait=wait_exponential(max=8), reraise=True)
def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def page_url(path: str) -> str:
    return f"{DATASET}/images/{urllib.parse.quote(path)}"  # filenames hold spaces, em-dashes and CJK


@cached(DISK, key=lambda: "omnidocbench/pages")
def pages() -> list[tuple[str, str]]:
    """(image_path, data_source) for each of the dataset's 1651 pages."""
    info = (r["page_info"] for r in json.loads(fetch(f"{DATASET}/OmniDocBench.json")))
    return [(p["image_path"], p["page_attribute"]["data_source"]) for p in info]


def sample(n: int | None, seed: int) -> list[tuple[str, str]]:
    """A stratified sample: each type's pages shuffled with one seeded RNG, first n kept (all of them for n=None)."""
    rng, by_type = random.Random(seed), defaultdict(list)
    for path, truth in pages():
        by_type[truth].append(path)
    picked = []
    for truth, pool in sorted(by_type.items()):
        rng.shuffle(pool)
        picked += [(path, truth) for path in pool[:n]]
    return picked


def encode(raw: bytes, max_side: int = 1024) -> str:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS, reducing_gap=None)
    img.save(buf := io.BytesIO(), "JPEG", quality=90)
    return data_url(buf.getvalue(), "image/jpeg")


@contextmanager
def system_one(questions: dict, **options):
    """Yield read(image) -> (answers, ms, cost) over one System One stream (POST or WS), opened on first use.

    `options` go straight to `client.gateway.systemone.stream`. Reads retry with backoff; a dropped WS session is
    reopened by whichever read notices first."""
    client = VLMRun()  # raises unless VLMRUN_API_KEY is set
    connect, ws = partial(client.gateway.systemone.stream, questions, **options), options.get("transport") == "ws"
    stream, lock = None, threading.Lock()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(max=8), reraise=True)
    def read(image: str) -> tuple[dict, float, float]:
        nonlocal stream
        with lock:
            stream = stream or connect().__enter__()
        s, start = stream, time.perf_counter()
        try:
            r = s.send(image)
        except Exception:
            with lock:
                if ws and s is stream:
                    s.close(cancel=True)
                    stream = None
            raise
        ms = (time.perf_counter() - start) * 1000
        return {k: a.model_dump() for k, a in r.answers.items()}, ms, usage_of(r).get("cost", 0.0)

    try:
        yield read
    finally:
        if stream:
            stream.close()


def soften(p: dict[str, float], t: float) -> dict[str, float]:
    """Temperature scaling: divide the implied logits by T, resoftmax. Flattens peaks, never reorders them."""
    w = {k: max(v, 1e-12) ** (1 / t) for k, v in p.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def ece(rows: list[dict], t: float, bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| per confidence bucket, weighted by bucket size."""
    gap = defaultdict(float)
    for r in rows:
        p = soften(r["probabilities"], t)
        pred = max(p, key=p.get)
        gap[min(int(p[pred] * bins), bins - 1)] += p[pred] - (pred == r["truth"])
    return sum(map(abs, gap.values())) / len(rows)


def nll(rows: list[dict], t: float) -> float:
    return -sum(math.log(max(soften(r["probabilities"], t)[r["truth"]], 1e-12)) for r in rows)


def topk(r: dict, k: int) -> bool:
    return r["truth"] in sorted(r["probabilities"], key=r["probabilities"].get, reverse=True)[:k]


# ---- the reel: 1080x1080, the page on the left, everything the model said on the right
BG, CARD, LINE, BAR = (14, 16, 22), (22, 24, 30), (36, 40, 50), (40, 44, 56)
INK, DIM, OK, BLUE = (240, 240, 245), (150, 152, 160), (140, 230, 120), (80, 110, 150)


@cache
def geist() -> bytes | None:
    with suppress(Exception):
        return fetch(GEIST)


@cache
def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Pillow's bundled sans, or Geist Mono for data (downloaded once; the sans when offline)."""
    return ImageFont.truetype(io.BytesIO(ttf), size) if mono and (ttf := geist()) else ImageFont.load_default(size)


def fit(d: ImageDraw.ImageDraw, text: str, width: int, sizes: tuple[int, ...], mono: bool = False):
    return next((f for s in sizes if d.textlength(text, font=(f := font(s, mono))) <= width), font(sizes[-1], mono))


def frame(page: Image.Image, r: dict, i: int, n: int, spent: float, t: float, model: str, intro: int) -> Image.Image:
    im, held = Image.new("RGB", (1080, 1080), BG), i <= intro
    d, probs = ImageDraw.Draw(im), soften(r["probabilities"], t)
    d.text((40, 24), "OmniDocBench · classified from pixels", font=font(40), fill=INK)
    d.text((40, 76), f"{model.split('/')[-1]} · document_type", font=font(20, True), fill=DIM)
    speed, pace = (f"{r['latency_ms'] / HOLD_MS:.2g}×", "slowed for legibility") if held else ("1×", "real-time")
    d.text((1040, 20), speed, font=font(34, True), fill=INK, anchor="ra")
    d.text((1040, 68), pace, font=font(16), fill=DIM, anchor="ra")

    d.rectangle(box := (40, 140, 720, 1015), outline=LINE)
    k = min((box[2] - box[0]) / page.width, (box[3] - box[1]) / page.height)
    page = page.resize((max(1, round(page.width * k)), max(1, round(page.height * k))), Image.LANCZOS)
    im.paste(page, (box[0] + (box[2] - box[0] - page.width) // 2, box[1] + (box[3] - box[1] - page.height) // 2))

    d.rectangle((740, 140, 1060, 1015), fill=CARD, outline=LINE)
    label, verdict = r["pred"].replace("_", " "), "correct" if r["pred"] == r["truth"] else "wrong"
    d.text((760, 152), label, font=fit(d, label, 280, (36, 30, 25, 21)), fill=INK)
    verdict += f"  ·  truth: {r['truth'].replace('_', ' ')}"
    d.text((760, 198), verdict, font=fit(d, verdict, 280, (15, 13, 11), mono=True), fill=DIM)
    d.text((760, 234), f"document type  ·  calibrated, T={t:g}", font=font(17), fill=DIM)
    for y, c in zip(count(262, 26), CATEGORIES):  # fixed order, same row every frame
        on, p = c == r["pred"], probs.get(c, 0.0)
        d.text((760, y), c.replace("_", " "), font=font(14), fill=INK if on else DIM)
        d.rectangle((932, y + 4, 1006, y + 12), fill=BAR)
        d.rectangle((932, y + 4, 932 + max(int(74 * p), 2), y + 12), fill=OK if on else BLUE)
        d.text((1008, y), f"{p:.0%}", font=font(13, True), fill=DIM)
    d.line((760, 530, 1040, 530), fill=LINE)
    meters = {"page": f"{i}/{n}", "latency": f"{r['latency_ms']:.0f} ms", "total cost": f"${spent:.4f}"}
    for y, (key, value) in zip(count(552, 74), meters.items()):
        d.text((760, y), " ".join(key.upper()), font=font(13, True), fill=(130, 132, 140))
        d.text((760, y + 20), value, font=font(40, True), fill=OK)
    d.text((760, 760), f"${r['cost_usd']:.5f} this page", font=font(14, True), fill=(140, 142, 150))
    note = (
        f"first {intro} pages held {HOLD_MS / 1000:.1f}s each so you can actually read them"
        if held
        else "this frame is on screen for exactly its own read latency, no artificial hold"
    )
    d.text((40, 1030), note, font=font(18), fill=DIM)
    return im


@contextmanager
def reel(path: Path, tune: str = "stillimage"):
    """Yield add(image, hold_ms): an H.264 mp4 that holds every frame for its own duration (variable frame rate)."""
    out = av.open(str(path), "w", options={"movflags": "faststart"})
    v = out.add_stream("libx264", rate=1000, options={"crf": "18", "preset": "slow", "tune": tune})
    v.width, v.height, v.pix_fmt, v.codec_context.time_base = 1080, 1080, "yuv420p", Fraction(1, 1000)
    t, last = 0.0, None

    def add(img: Image.Image, hold_ms: float) -> None:
        nonlocal t, last
        poster = 0 if last else 1000  # the first frame doubles as a 1 s poster
        last = av.VideoFrame.from_image(img)
        last.pts, last.time_base = round(t), v.codec_context.time_base
        out.mux(v.encode(last))
        t += max(hold_ms, 33) + poster

    try:
        yield add
    finally:
        if last:
            last.pts = round(t)  # repeat the last frame so its own hold plays out
            out.mux(v.encode(last))
        out.mux(v.encode())
        out.close()


def main(
    n_per_category: int = 20,
    all_pages: Annotated[bool, typer.Option("--all", help="Every one of the 1651 pages.")] = False,
    ws: Annotated[bool, typer.Option("--ws", help="One WS session for the run instead of a POST per page.")] = False,
    concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Reads in flight (WS caps at 8).")] = 1,
    seed: int = 0,
    model: str = MODEL,
    temperature: Annotated[float, typer.Option(help="Calibration T for the reel; fit for the default model.")] = 2.9,
    mp4: Annotated[bool, typer.Option("--mp4", help="Record a reel, each page held for its own read latency.")] = False,
    slow_intro: Annotated[int, typer.Option(help=f"Hold the first N pages {HOLD_MS} ms; 0 is pure 1×.")] = 5,
    out: Path = Path("classify_docs.json"),
) -> None:
    """Classify OmniDocBench pages by document type, from pixels."""
    picked, totals = sample(None if all_pages else n_per_category, seed), Counter(t for _, t in pages())
    transport = "ws" if ws else "http"
    print(f"classifying {len(picked)} of {totals.total()} pages over {transport} with {model}")
    stream = dict(state=STATE, model=model, detail="high", concurrency=concurrency, transport=transport)
    with system_one(QUESTIONS, **stream) as read:

        @cached(DISK, key=lambda path, truth: f"classify/{model}/{transport}/{PROMPT}/{path}")
        def classify(path: str, truth: str) -> dict:
            answers, ms, cost = read(encode(fetch(page_url(path))))
            a = answers["document_type"]
            return dict(
                path=path, truth=truth, pred=a["choice"], probabilities=a["probabilities"], latency_ms=ms, cost_usd=cost
            )

        def attempt(page: tuple[str, str]) -> dict | None:
            try:
                return classify(*page)
            except Exception as e:  # one page out past its retries should not cost the rest of the run
                print(f"[red]skipped[/] {escape(page[0])}: {escape(repr(e))}")

        with ThreadPoolExecutor(concurrency) as pool:
            rows = [r for r in track(pool.map(attempt, picked), "reading", len(picked)) if r]
    if not rows:
        raise SystemExit("every page failed")

    lat, spent = sorted(r["latency_ms"] for r in rows), sum(r["cost_usd"] for r in rows)
    fit_t, (top_type, top_n) = min((i / 20 for i in range(10, 121)), key=partial(nll, rows)), totals.most_common(1)[0]
    stats = {f"top{k}": sum(topk(r, k) for r in rows) / len(rows) for k in (1, 2, 3)}
    stats |= {"ece_raw": ece(rows, 1), "ece": ece(rows, temperature), "ece_fit": ece(rows, fit_t), "t_fit": fit_t}
    print(
        f"accuracy  top1 {stats['top1']:.1%}  top2 {stats['top2']:.1%}  top3 {stats['top3']:.1%}  "
        f"(chance: uniform {1 / len(totals):.1%}, always-{top_type} {top_n / totals.total():.1%})"
    )
    print(
        f"latency   p50 {statistics.median(lat):.0f} ms  mean {statistics.mean(lat):.0f} ms  "
        f"p90 {lat[min(len(lat) - 1, int(0.9 * len(lat)))]:.0f} ms"
    )
    print(f"cost      ${spent:.4f} total, ${spent / len(rows):.5f} per page")
    print(
        f"calibration  ECE raw {stats['ece_raw']:.3f} -> {stats['ece']:.3f} at T={temperature:g} "
        f"(this run's own fit: T={fit_t:.2f} -> {stats['ece_fit']:.3f})"
    )

    table = Table("type", "top1", "top3", "n", "confused for", title="per type")
    for c in sorted(totals):
        if sub := [r for r in rows if r["truth"] == c]:
            hits = (f"{sum(topk(r, k) for r in sub) / len(sub):.0%}" for k in (1, 3))
            wrong = Counter(r["pred"] for r in sub if r["pred"] != c).most_common(3)
            table.add_row(c, *hits, str(len(sub)), ", ".join(f"{k}×{v}" for k, v in wrong))
    print(table)
    summary = {"model": model, "prompt": PROMPT, "n": len(rows), "temperature": temperature, "cost_usd": spent} | stats
    out.write_text(json.dumps(summary | {"rows": rows}, indent=1))
    print(f"-> {out}")

    if mp4:
        with reel(path := out.with_suffix(".mp4")) as add:
            running = accumulate(r["cost_usd"] for r in rows)
            for i, (r, so_far) in enumerate(zip(track(rows, "rendering"), running, strict=True), 1):
                page = Image.open(io.BytesIO(fetch(page_url(r["path"])))).convert("RGB")
                hold = HOLD_MS if i <= slow_intro else r["latency_ms"]
                add(frame(page, r, i, len(rows), so_far, temperature, model, slow_intro), hold)
        print(f"-> {path}")


if __name__ == "__main__":
    typer.run(main)
