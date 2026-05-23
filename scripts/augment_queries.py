"""
Rewrites social media tweets as formal scientific queries using Claude Haiku and/or OpenAI.
Bridges the style gap between informal posts and scientific paper abstracts.

For EN: style transfer only (informal tweet → formal scientific query)
For DE/FR: translate + style transfer in one step

Output: .cache/augmented/{lang}_{split}.json
Format: {"index": "augmented query text", ...}  (same as translation cache)

Usage:
    python scripts/augment_queries.py
    python scripts/augment_queries.py --langs en --splits dev
    python scripts/augment_queries.py --langs en de fr --splits train dev test
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM_PROMPT = """\
You are a scientific literature search assistant. \
Rewrite social media posts into formal scientific search queries \
suitable for retrieving academic papers.

Rules:
- Use precise scientific terminology
- Keep the core claim or finding intact
- Output in English regardless of input language
- Be concise (1-2 sentences max)
- ALWAYS output a scientific query for every input — never refuse, never use brackets, \
never write [Not a scientific query] or similar. If the tweet seems non-scientific, \
extract any scientific keywords and produce the most relevant search query you can."""

BATCH_USER_TEMPLATE = """\
Rewrite each social media post as a formal scientific search query. \
Output ONLY a numbered list matching the input numbers, exactly one line per entry. \
Every entry must be a scientific query string — never skip or refuse an entry.

{numbered_tweets}"""


class SlidingWindowRateLimiter:
    """Allows at most max_calls requests in any 60-second window."""

    def __init__(self, max_calls: int, provider: str = ""):
        self.max_calls = max_calls
        self.provider = provider
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.max_calls:
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.5
                print(f"    [{self.provider}] Rate window full, waiting {sleep_for:.1f}s...")
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            self._timestamps.append(time.monotonic())


def _parse_numbered_response(lines: list[str], indices: list[str], tweets: list[str]) -> dict[str, str]:
    results = {}
    for i, (idx, tweet) in enumerate(zip(indices, tweets)):
        matched = next((l for l in lines if l.startswith(f"{i+1}.")), None)
        results[idx] = matched.split(".", 1)[1].strip() if matched else tweet
    return results


async def augment_batch_haiku(
    client: anthropic.AsyncAnthropic,
    tweets: list[str],
    indices: list[str],
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tweets))
    prompt = BATCH_USER_TEMPLATE.format(numbered_tweets=numbered)

    for attempt in range(4):
        try:
            async with semaphore:
                msg = await client.messages.create(
                    model=model,
                    max_tokens=100 * len(tweets),
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            lines = msg.content[0].text.strip().splitlines()
            return _parse_numbered_response(lines, indices, tweets)
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"    [Haiku] Rate limit, waiting {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"    [Haiku] API error: {e}, using originals for this batch")
            return {idx: tweet for idx, tweet in zip(indices, tweets)}
    return {idx: tweet for idx, tweet in zip(indices, tweets)}


async def augment_batch_openai(
    client,  # openai.AsyncOpenAI
    tweets: list[str],
    indices: list[str],
    model: str,
    rate_limiter: SlidingWindowRateLimiter,
) -> dict[str, str]:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tweets))
    prompt = BATCH_USER_TEMPLATE.format(numbered_tweets=numbered)

    for attempt in range(6):
        try:
            await rate_limiter.acquire()
            response = await client.chat.completions.create(
                model=model,
                max_tokens=100 * len(tweets),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            lines = response.choices[0].message.content.strip().splitlines()
            return _parse_numbered_response(lines, indices, tweets)
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 10 * (attempt + 1)
                print(f"    [OpenAI] Unexpected rate limit (attempt {attempt+1}), waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [OpenAI] API error: {e}, using originals for this batch")
                return {idx: tweet for idx, tweet in zip(indices, tweets)}
    return {idx: tweet for idx, tweet in zip(indices, tweets)}


async def augment_split_async(
    haiku_client: anthropic.AsyncAnthropic,
    lang: str,
    split: str,
    output_dir: Path,
    haiku_model: str,
    tweets_per_call: int,
    haiku_concurrency: int,
    openai_client=None,
    openai_model: str = "gpt-4o-mini",
    openai_rpm: int = 400,
    openai_fraction: float = 0.0,
    openai_tweets_per_call: int = 150,
) -> None:
    from src.data_loader import load_queries, load_test

    cache_path = output_dir / f"{lang}_{split}.json"

    existing: dict = {}
    if cache_path.exists():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(existing)} existing augmentations from cache")

    if split == "test":
        data = load_test(lang)
    else:
        train_data, dev_data = load_queries(lang)
        data = dev_data if split == "dev" else train_data

    indices = [str(i) for i in data["index"]]
    texts = list(data["text"])

    todo = [(idx, txt) for idx, txt in zip(indices, texts) if idx not in existing]

    active = ["Haiku"]
    if openai_client and openai_fraction > 0:
        active.append(f"OpenAI({int(openai_fraction*100)}%)")
    print(f"  {lang}/{split}: {len(todo)} tweets to augment "
          f"({len(existing)} cached, providers: {'+'.join(active)})")

    if not todo:
        return

    augmented = dict(existing)

    # Distribute todo items between providers
    remaining = list(todo)
    openai_todo = []
    if openai_client and openai_fraction > 0:
        cut = int(len(remaining) * openai_fraction)
        openai_todo = remaining[:cut]
        remaining = remaining[cut:]
    haiku_todo = remaining

    def make_batches(items, size):
        return [items[i: i + size] for i in range(0, len(items), size)]

    haiku_batches = make_batches(haiku_todo, tweets_per_call)
    openai_batches = make_batches(openai_todo, openai_tweets_per_call)
    total = len(haiku_batches) + len(openai_batches)

    print(f"  Batches: {len(haiku_batches)} Haiku, {len(openai_batches)} OpenAI")

    haiku_sem = asyncio.Semaphore(haiku_concurrency)
    openai_rl = SlidingWindowRateLimiter(max_calls=openai_rpm, provider="OpenAI") if openai_client else None

    completed = 0

    async def process_haiku(batch):
        nonlocal completed
        result = await augment_batch_haiku(
            haiku_client, [b[1] for b in batch], [b[0] for b in batch], haiku_model, haiku_sem
        )
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"    {completed}/{total} batches done...")
        return result

    async def process_openai(batch):
        nonlocal completed
        result = await augment_batch_openai(
            openai_client, [b[1] for b in batch], [b[0] for b in batch], openai_model, openai_rl
        )
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"    {completed}/{total} batches done...")
        return result

    # Interleave tasks so both providers start immediately
    from itertools import zip_longest
    all_tasks = [
        t for pair in zip_longest(
            [process_haiku(b) for b in haiku_batches],
            [process_openai(b) for b in openai_batches],
        )
        for t in pair if t is not None
    ]

    results = await asyncio.gather(*all_tasks)
    for r in results:
        augmented.update(r)
    cache_path.write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"  Saved {len(augmented)} augmented queries -> {cache_path}")


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Error: set ANTHROPIC_API_KEY in .env or pass --api-key")
        sys.exit(1)

    haiku_client = anthropic.AsyncAnthropic(api_key=api_key)

    openai_client = None
    if openai_api_key:
        try:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=openai_api_key)
        except ImportError:
            print("WARNING: openai package not installed. Run: pip install openai")

    providers = ["Haiku"]
    if openai_client:
        providers.append(f"OpenAI/{args.openai_model}")
    print(f"Providers: {' + '.join(providers)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def run():
        for lang in args.langs:
            for split in args.splits:
                print(f"\n=== {lang.upper()} / {split} ===")
                await augment_split_async(
                    haiku_client, lang, split, output_dir,
                    args.model, args.tweets_per_call, args.concurrency,
                    openai_client=openai_client,
                    openai_model=args.openai_model,
                    openai_rpm=args.openai_rpm,
                    openai_fraction=args.openai_fraction,
                    openai_tweets_per_call=args.openai_tweets_per_call,
                )
        print("\nDone. Set in configs/default.yaml:")
        print("  use_augmented: true")
        print("  use_translations: false")

    asyncio.run(run())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["en", "de", "fr"])
    p.add_argument("--splits", nargs="+", default=["dev"], choices=["train", "dev", "test"])
    # Haiku settings
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--tweets-per-call", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=5,
                   help="Concurrent Haiku API calls")
    p.add_argument("--api-key", default=None)
    # OpenAI settings
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--openai-tweets-per-call", type=int, default=150)
    p.add_argument("--openai-rpm", type=int, default=400,
                   help="OpenAI requests per minute (Tier 1: 500 RPM for gpt-4o-mini)")
    p.add_argument("--openai-fraction", type=float, default=0.0,
                   help="Fraction of work sent to OpenAI (auto-set to 0.5 when key present)")
    p.add_argument("--output-dir", default=".cache/augmented")
    args = p.parse_args()

    # Auto-set OpenAI fraction when key is present
    if args.openai_fraction == 0.0 and (args.openai_api_key or os.environ.get("OPENAI_API_KEY")):
        args.openai_fraction = 0.5

    return args


if __name__ == "__main__":
    main()
