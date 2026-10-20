# /// script
# requires-python = ">=3.11"
# dependencies = ["vlmrun[typesafe,ws]>=0.9.2", "diskcache>=5.6", "av>=14", "gymnasium-2048>=0.1"]
# ///
"""Snake and 2048, played from pixels: the model is a sensor, the script owns the rules, the guard and the loop.

    uv run pixelgames.py snake --ws --episodes 5 --mp4
    uv run pixelgames.py snake --baseline             # random among safe moves, no API calls
    uv run pixelgames.py 2048 --ws --episodes 20 --mp4

Snake: 87 food / 3000 moves, no deaths (random-among-safe 33, greedy BFS 75). Two reads -- bearing to the food and
own heading -- are multiplied, not picked between: P(toward food) x P(not the reversal). Survival is a local
tail-reachability veto, not the model.

2048: median 1800 (random 920, fixed-rule bots 2258-2418). One four-label read; a local no-op guard takes the next
choice when a slide would move nothing. Its first choice is only ever down or right: the corner strategy, by accident.

Reads are cached on disk by frame, so re-running a seed replays the same games (and re-renders them) for free.
"""

import copy
import hashlib
import io
import json
import os
import random
import statistics
import threading
import time
import urllib.request
from collections import deque
from contextlib import contextmanager, nullcontext, suppress
from fractions import Fraction
from functools import cache, partial
from itertools import count
from pathlib import Path
from typing import Annotated, Literal

import av
import typer
from cachetools import cached
from diskcache import Cache
from PIL import Image, ImageDraw, ImageFont
from rich import print
from tenacity import retry, stop_after_attempt, wait_exponential
from vlmrun.client import VLMRun
from vlmrun.client.systemone import usage_of
from vlmrun.common.mime import data_url

GEIST = "https://raw.githubusercontent.com/vercel/geist-font/v1.7.2/fonts/GeistMono/ttf/GeistMono-Regular.ttf"
MODEL = os.getenv("TYPESAFE_DEFAULT_MODEL", "google/diffusiongemma-26b-a4b-it")
DISK = Cache(os.getenv("SYSTEMONE_CACHE", Path.home() / ".cache" / "vlmrun-systemone"))
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class Snake:
    """A 10x10 snake with the two tests its guard needs: which moves are legal, and which keep the tail reachable."""

    title, moves = "Snake", tuple(DIRS)
    rules = (
        "This image is a game of Snake in progress.\n- The bright yellow-green square outlined in white is the "
        "snake's HEAD, and the black arrow on it points the way it is travelling.\n- The squares joined to it are its "
        "BODY, fading darker toward the TAIL.\n- The red circle is the FOOD. The grey frame around the grid is the "
        "WALL.\nEat the food as quickly as possible and stay alive. The game ends instantly if the head moves into the "
        "wall or onto any part of the body, and the snake cannot turn back on itself.\nDirections are screen "
        "directions: up is toward the top edge of the image, and so on."
    )
    questions = {
        "heading": {
            "type": "choice",
            "instructions": "Which way is the snake travelling right now? "
            "A black arrow on the bright head points the way.",
            "options": {
                m: f"the head is travelling toward the {e} edge"
                for m, e in zip(DIRS, ("top", "bottom", "left", "right"), strict=True)
            },
        },
        "vertical": {
            "type": "choice",
            "instructions": "Is the red food higher up the image than the head, lower down, or in the same row?",
            "options": {
                "up": "the food is nearer the top edge than the head",
                "down": "the food is nearer the bottom edge than the head",
                "level": "food and head share a row",
            },
        },
        "horizontal": {
            "type": "choice",
            "instructions": "Is the red food further left than the head, further right, or in the same column?",
            "options": {
                "left": "the food is nearer the left edge than the head",
                "right": "the food is nearer the right edge than the head",
                "level": "food and head share a column",
            },
        },
    }

    def __init__(self, seed: int, n: int = 10) -> None:
        self.n, self.rng, c = n, random.Random(seed), n // 2
        self.body, self.heading, self.score, self.over = deque([(c, c), (c - 1, c), (c - 2, c)]), "right", 0, False
        self.food = self.spawn()

    @property
    def label(self) -> str:
        return f"{self.score} food"

    def spawn(self) -> tuple[int, int]:
        free = [(x, y) for x in range(self.n) for y in range(self.n) if (x, y) not in self.body]
        return self.rng.choice(free) if free else (-1, -1)

    def inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n

    def blocked(self, cell: tuple[int, int]) -> bool:
        return not self.inside(cell) or cell in list(self.body)[:-1]  # the tail moves out of the way

    def ahead(self, move: str, cell: tuple[int, int] | None = None) -> tuple[int, int]:
        (x, y), (dx, dy) = cell or self.body[0], DIRS[move]
        return x + dx, y + dy

    def play(self, move: str) -> None:
        if move != OPPOSITE[self.heading]:  # a reversal is ignored, as a real game ignores it
            self.heading = move
        if self.blocked(head := self.ahead(self.heading)):
            self.over = True
            return
        self.body.appendleft(head)
        if head == self.food:
            self.score, self.food = self.score + 1, self.spawn()
        else:
            self.body.pop()

    def after(self, move: str) -> "Snake | None":
        twin = copy.deepcopy(self)
        twin.play(move)
        return None if twin.over else twin

    def reach(self, target: tuple[int, int] | None = None) -> bool | int:
        """Flood fill from the head: whether `target` is reachable or, with no target, how many cells are."""
        seen, queue = {self.body[0]}, deque([self.body[0]])
        while queue:
            cell = queue.popleft()
            for nxt in (self.ahead(m, cell) for m in DIRS):
                if nxt == target:
                    return True
                if nxt not in seen and not self.blocked(nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        return False if target else len(seen) - 1

    def tail_safe(self, move: str) -> bool:
        twin = None if move == OPPOSITE[self.heading] else self.after(move)
        return bool(twin and (twin.reach(twin.body[-1]) or twin.reach() >= len(twin.body) + 2))

    def guard(self, ranked: list[str]) -> list[str]:
        """Keep the model's order, drop the moves that trap the snake; if none is safe, take the most room."""
        if safe := [m for m in ranked if self.tail_safe(m)]:
            return safe
        legal = [m for m in self.moves if m != OPPOSITE[self.heading] and not self.blocked(self.ahead(m))]
        return [max(legal, key=lambda m: t.reach() if (t := self.after(m)) else -1)] if legal else [self.heading]

    def tags(self) -> dict[str, str]:
        """What the engine knows about each move: drawn on the reel, never sent to the model."""

        def tag(m: str) -> str:
            if m == OPPOSITE[self.heading]:
                return "reversal"
            if not self.inside(cell := self.ahead(m)):
                return "wall"
            return "body" if self.blocked(cell) else "safe" if self.tail_safe(m) else "trap"

        return {m: tag(m) for m in self.moves}

    def fuse(self, answers: dict) -> dict[str, float]:
        """P(toward food) x P(not the reversal), multiplied from two calibrated reads, never one 'which way?'."""
        v, h, facing = (answers[k]["probabilities"] for k in ("vertical", "horizontal", "heading"))
        toward = {"up": v.get("up", 0), "down": v.get("down", 0), "left": h.get("left", 0), "right": h.get("right", 0)}
        score = {m: (toward[m] + 0.02) * (1 - facing.get(OPPOSITE[m], 0)) for m in self.moves}
        return {m: s / (sum(score.values()) or 1) for m, s in score.items()}

    def frame(self, cell: int = 32) -> Image.Image:
        pad, size = cell, (self.n + 2) * cell
        img = Image.new("RGB", (size, size), (138, 146, 172))  # a solid wall band; a hairline reads as empty floor
        d = ImageDraw.Draw(img)
        d.rectangle((pad, pad, size - pad - 1, size - pad - 1), fill=BG)
        for i in range(pad, size - pad + 1, cell):
            d.line(((i, pad), (i, size - pad)), fill=(28, 32, 44))
            d.line(((pad, i), (size - pad, i)), fill=(28, 32, 44))

        def box(cell_xy: tuple[int, int], inset: int = 2) -> tuple[int, int, int, int]:
            x0, y0 = pad + cell_xy[0] * cell + inset, pad + cell_xy[1] * cell + inset
            return x0, y0, x0 + cell - 2 * inset, y0 + cell - 2 * inset

        head, tail = (163, 255, 84), (22, 101, 52)
        d.ellipse(box(self.food, 5), fill=(244, 63, 63))
        for i, xy in enumerate(self.body):  # fading from head to tail
            t = min(i / max(len(self.body) - 1, 1), 1.0)
            d.rounded_rectangle(
                box(xy), radius=6, fill=tuple(round(a + (b - a) * t) for a, b in zip(head, tail, strict=True))
            )
        d.rounded_rectangle(box(self.body[0]), radius=6, fill=head, outline=(255, 255, 255), width=2)
        (x, y), r = self.body[0], cell * 0.3
        arrow(d, pad + x * cell + cell / 2, pad + y * cell + cell / 2, self.heading, r, r, BG)  # the heading arrow
        return img

    def view(self, frame: Image.Image) -> Image.Image:
        return frame


SCALE = 3  # 2048 renders natively at 3x for the reel; the model sees it downsampled back to 1x


@cache
def gym2048():
    """gymnasium-2048, imported on first use, with its renderer's module constants scaled up SCALE x."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from gymnasium_2048.envs import twenty_forty_eight as tfe

    for k in ("WINDOW_WIDTH", "WINDOW_HEIGHT", "WINDOW_SCORE_HEIGHT", "BOARD_PADDING", "FONT_SIZE"):
        setattr(tfe, k, getattr(tfe, k) * SCALE)
    return tfe


class TwentyFortyEight:
    title, moves, actions = "2048", ("up", "down", "left", "right"), ("up", "right", "down", "left")  # env's order
    rules = (
        "This is a game of 2048. Every numbered tile slides as far as it can in the direction you choose, and two "
        "tiles showing the same number that collide merge into their sum.\nBuild the largest tile you can, and do not "
        "fill the board."
    )
    questions = {"action": {"type": "choice", "instructions": "Which way do you slide the tiles?", "options": actions}}

    def __init__(self, seed: int) -> None:
        self.env, self.over = gym2048().TwentyFortyEightEnv(render_mode="rgb_array"), False
        _, self.info = self.env.reset(seed=seed)

    @property
    def score(self) -> int:
        return self.info["total_score"]

    @property
    def label(self) -> str:
        return f"{self.score} pts · best {2 ** self.info['max'] if self.info['max'] else 0}"

    def legal(self, move: str) -> bool:
        return self.env.apply_action(self.env.board, self.actions.index(move))[2]

    def guard(self, ranked: list[str]) -> list[str]:
        return [m for m in ranked if self.legal(m)] or ranked  # skip a slide that moves nothing

    def tags(self) -> dict[str, str]:
        return {m: "legal" if self.legal(m) else "no-op" for m in self.moves}

    def fuse(self, answers: dict) -> dict[str, float]:
        return {m: answers["action"]["probabilities"].get(m, 0.0) for m in self.actions}

    def play(self, move: str) -> None:
        _, _, self.over, _, self.info = self.env.step(self.actions.index(move))

    def frame(self) -> Image.Image:
        return Image.fromarray(self.env.render())

    def view(self, frame: Image.Image) -> Image.Image:
        return frame.resize((frame.width // SCALE, frame.height // SCALE), Image.LANCZOS)


GAMES = {"snake": Snake, "2048": TwentyFortyEight}


@cached(DISK, key=lambda url: f"fetch/{url}")
@retry(stop=stop_after_attempt(4), wait=wait_exponential(max=8), reraise=True)
def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


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


# ---- the reel: 1080x1080, the game on the left, everything the model said on the right
BG, CARD, LINE, BAR = (14, 16, 22), (22, 24, 30), (36, 40, 50), (40, 44, 56)
INK, DIM, OK, BLUE, LIME = (240, 240, 245), (150, 152, 160), (140, 230, 120), (80, 110, 150), (163, 255, 84)
TAG = {"safe": (110, 200, 120), "legal": (110, 200, 120), "no-op": (200, 140, 60)}  # anything else is fatal: red
FAST_AFTER_MS, FAST = 5000, 3.0  # real speed for the first 5 s of video, then 3x so a long game still fits


@cache
def geist() -> bytes | None:
    with suppress(Exception):
        return fetch(GEIST)


@cache
def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Pillow's bundled sans, or Geist Mono for data (downloaded once; the sans when offline)."""
    return ImageFont.truetype(io.BytesIO(ttf), size) if mono and (ttf := geist()) else ImageFont.load_default(size)


def arrow(d: ImageDraw.ImageDraw, cx: float, cy: float, move: str, length: float, width: float, fill) -> None:
    dx, dy = DIRS[move]
    base = ((cx - length * dx + s * width * dy, cy - length * dy + s * width * dx) for s in (-1, 1))
    d.polygon(((cx + length * dx, cy + length * dy), *base), fill=fill)


def card(game, frame, probs, chosen, ms, n, cost, spent, mode, speed) -> Image.Image:
    """The frame the model saw and what it said: controls in the game's fixed order, each with the engine's own tag."""
    im = Image.new("RGB", (1080, 1080), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 24), f"{game.title}  ·  played from pixels", font=font(40), fill=INK)
    d.text((40, 76), mode, font=font(20, True), fill=DIM)
    if ms:
        d.text((1040, 20), f"{speed:g}×", font=font(34, True), fill=INK, anchor="ra")
        d.text((1040, 68), "real-time" if speed == 1 else "sped up", font=font(16), fill=DIM, anchor="ra")
    d.rectangle((40, 140, 720, 820), outline=LINE)
    k = 680 / max(frame.size)
    frame = frame.resize((round(frame.width * k), round(frame.height * k)), Image.NEAREST)
    im.paste(frame, (40 + (680 - frame.width) // 2, 140 + (680 - frame.height) // 2))

    d.rectangle((740, 140, 1060, 1015), fill=CARD, outline=LINE)
    big, _, small = game.label.partition(" · ")
    d.text((760, 152), big, font=font(44), fill=INK)
    d.text((760, 204), small, font=font(18, True), fill=DIM)
    d.text((760, 244), "move probabilities", font=font(18), fill=DIM)
    d.text((1040, 244), f"choice: {chosen}", font=font(18), fill=LIME, anchor="ra")
    tags = game.tags()
    for y, m in zip(count(276, 46), game.moves):
        on, p = m == chosen, probs.get(m, 0.0) if probs else float(m == chosen)
        arrow(d, 767, y + 13, m, 7, 6, INK if on else (110, 112, 120))
        d.text((782, y), m, font=font(19), fill=INK if on else DIM)
        d.rectangle((870, y + 6, 980, y + 16), fill=BAR)
        d.rectangle((870, y + 6, 870 + max(int(110 * p), 2), y + 16), fill=LIME if on else BLUE)
        d.text((1040, y), f"{p:.0%}", font=font(16, True), fill=DIM, anchor="ra")
        d.text((760, y + 22), tags[m], font=font(13, True), fill=TAG.get(tags[m], (230, 90, 90)))
    d.line((760, 468, 1040, 468), fill=LINE)
    meters = {"step": str(n), "latency": f"{ms:.0f} ms" if ms else "no read", "total cost": f"${spent:.4f}"}
    for y, (key, value) in zip(count(490, 82), meters.items()):
        d.text((760, y), " ".join(key.upper()), font=font(13, True), fill=(130, 132, 140))
        d.text((760, y + 20), value, font=font(46, True), fill=OK)
    caption = f"${cost:.5f} this step" if ms else "no read this step"
    d.text((760, 722), caption, font=font(14, True), fill=(140, 142, 150))
    note = (
        "every frame is held for exactly its own read latency"
        if speed == 1
        else f"sped up {speed:g}x from here on -- each frame held for {1 / speed:.2g} of its real read latency"
    )
    d.text((40, 1035), note, font=font(18), fill=DIM)
    return im


@contextmanager
def reel(path: Path, tune: str = "animation"):
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


def episode(game, decide, add, max_decisions: int, rng: random.Random | None, mode: str) -> dict:
    """One frame -> one read -> the guard -> one move, until the game ends or the budget runs out."""
    lat, spent, actions, video_ms = [], 0.0, [], 0.0
    while len(actions) < max_decisions and not game.over:
        frame = game.frame()
        if rng:  # the baseline: random order, same guard, no read
            probs, ms, cost, ranked = {}, 0.0, 0.0, rng.sample(game.moves, len(game.moves))
        else:
            view = game.view(frame)
            view.save(buf := io.BytesIO(), "PNG")
            answers, ms, cost = decide(buf.getvalue())
            probs = game.fuse(answers)
            ranked, spent = sorted(probs, key=probs.get, reverse=True), spent + cost
            lat.append(ms)
        actions.append(chosen := game.guard(ranked)[0])
        if add:
            speed = FAST if ms and video_ms >= FAST_AFTER_MS else 1.0
            video_ms += (hold := max(ms / speed, 33))
            add(card(game, frame, probs, chosen, ms, len(actions), cost, spent, mode, speed), hold)
        game.play(chosen)
    stats = dict(score=game.score, label=game.label, over=game.over, moves=len(actions), cost_usd=spent)
    return stats | {"latency_ms": lat, "actions": actions}


def main(
    game: Literal["snake", "2048"],
    episodes: int = 3,
    max_decisions: int = 3000,
    ws: Annotated[bool, typer.Option("--ws", help="Decide over one WS session instead of a POST per move.")] = False,
    baseline: Annotated[
        bool, typer.Option("--baseline", help="Random moves through the same guard; no API calls.")
    ] = False,
    seed: int = 0,
    model: str = MODEL,
    mp4: Annotated[
        bool, typer.Option("--mp4", help="Record each episode, each frame held for its read latency.")
    ] = False,
    out: Path = Path("outputs"),
) -> None:
    """Play GAME from pixels, one typed System One read per move."""
    cls, transport = GAMES[game], "ws" if ws else "http"
    arm, mode = ("random", "random baseline") if baseline else (transport, f"{model.split('/')[-1]} · {transport}")
    prompt = hashlib.sha256(json.dumps([cls.rules, cls.questions]).encode()).hexdigest()[:12]  # reworded => re-read
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    reader = nullcontext() if baseline else system_one(cls.questions, state=cls.rules, model=model, transport=transport)
    with reader as read:  # the baseline never reads, so it needs no key

        @cached(DISK, key=lambda png: f"{game}/{model}/{transport}/{prompt}/{hashlib.sha256(png).hexdigest()}")
        def decide(png: bytes) -> tuple[dict, float, float]:
            return read(data_url(png, "image/png"))

        for i, s in enumerate(range(seed, seed + episodes)):
            with reel(out / f"{game}-{arm}-ep{i}.mp4") if mp4 else nullcontext() as add:
                run = episode(cls(s), decide, add, max_decisions, random.Random(s) if baseline else None, mode)
            runs.append(run | {"seed": s})
            lat = f"p50 {statistics.median(run['latency_ms']):.0f} ms" if run["latency_ms"] else "no reads"
            over = "  game over" if run["over"] else ""
            print(
                f"{game}/{arm} ep{i}  {run['label']:<20} {run['moves']:>5} moves  {lat}  ${run['cost_usd']:.4f}{over}"
            )
    scores = [r["score"] for r in runs]
    print(f"{game}/{arm}: median {statistics.median(scores):g}, mean {statistics.fmean(scores):.1f} over {episodes}")
    (out / f"{game}-{arm}.json").write_text(json.dumps({"game": game, "model": model, "arm": arm, "episodes": runs}))


if __name__ == "__main__":
    typer.run(main)
