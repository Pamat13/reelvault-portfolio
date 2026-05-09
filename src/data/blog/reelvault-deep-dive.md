---
author: Jorge Pérez Amat
pubDatetime: 2026-05-09T09:00:00Z
title: "I built a local LLM pipeline to make my 2,695 saved Instagram reels searchable"
slug: reelvault-deep-dive
featured: true
draft: true
tags:
  - llm
  - python
  - personal-project
  - local-first
description: "Whisper + Gemma 4 + BGE-M3 on a laptop GPU. Why local-first matters, and the three blockers I had to debug through."
---

## TL;DR

- Built a 6-stage local LLM pipeline (Whisper + Gemma 4 + BGE-M3) to classify 2,695 Instagram reels — no cloud APIs
- Hit 3 hard bugs: Windows DLL hell, Q4 GPU OOM, Instagram rate-limiting — solved each with a targeted fix
- 2,491 reels indexed, ~92% auto-classified; the vault now has full keyword + semantic search

## Table of contents

## The saved-reels problem

Instagram lets you save reels. It gives you zero way to search them.

After years of tapping the bookmark icon, I had 2,695 saved reels. Tutorials, recipes, climbing technique breakdowns, random comedy clips I half-remembered. Finding anything meant scrolling, scrolling, and scrolling some more. The algorithm that served me the reel has no interest in helping me retrieve it.

The obvious answer is "just don't save so many reels" — but that's not the point. The point is that I had 2,695 units of information I had deliberately curated, and they were effectively locked in a dark room.

The slightly less obvious answer is "use a third-party app." I tried a few. They required OAuth access to my account, stored data on their servers, and charged subscriptions for the feature that actually mattered. None of them felt right.

So I built ReelVault: a fully local pipeline that downloads every reel, runs three different models over it, and produces a searchable vault of Markdown files backed by SQLite with FTS5 and vector embeddings. No cloud, no subscriptions, no data leaving my laptop.

## Why local-first

I want to be specific about what "local-first" means here, because it's not just a philosophical stance — it was a practical constraint that shaped every technical decision.

**Privacy.** My saved reels include personal interests I don't want indexed by a third-party service. Health topics, financial content, political commentary. Running everything locally means none of that leaves my machine.

**Cost.** 2,695 reels is a lot of tokens. Running OCR + summarization + tagging through GPT-4o Vision would cost somewhere between $80 and $200 at current pricing depending on how many frames per reel and how verbose the prompts. With a laptop GPU and open-weight models, the cost is electricity — roughly $0.

**Control.** Local models don't get deprecated, don't change behavior between API versions, and don't rate-limit me. I can re-run the pipeline any time with the same reproducible results.

**Learning.** I wanted to understand how these models actually work in practice. There's no better way to learn than to hit the real failure modes.

The tradeoff is setup complexity and hardware dependency. The pipeline requires an Nvidia GPU with at least 8GB VRAM and a willingness to debug Windows-specific DLL issues. That's a real barrier — but it's my barrier to clear once, not an ongoing subscription.

## Architecture

The pipeline has six stages that run sequentially per reel, with a SQLite jobs database tracking state so any crash or interruption is resumable:

1. **Discover** — read the list of saved reel URLs from Instagram (via yt-dlp with session cookies)
2. **Download** — fetch the MP4 and thumbnail; store in `vault/<id>/`
3. **Transcribe** — faster-whisper medium extracts audio and produces a transcript with timestamps
4. **Vision** — Gemma 4 E4B-IT Q4 receives 4 evenly-spaced frames and produces OCR text, a summary, category tags, and a confidence score
5. **Embed** — BGE-M3 generates dense vectors; MiniLM-L12-v2 generates sparse vectors for hybrid search
6. **Index** — results are written to a Markdown file and inserted into `index.db` (FTS5 table + vector table)

The vault stores each reel as a self-contained directory: the MP4, a thumbnail, and a `meta.md` file with all extracted metadata in YAML frontmatter. The SQLite database is the search layer — full-text search over transcripts and summaries, plus cosine similarity over BGE-M3 embeddings for semantic queries.

A FastAPI + Svelte 5 review UI (Phase 2) lets me manually triage the ~8% of reels where the model confidence was below threshold.

## Hard part 1: torchcodec → ffmpeg frame extraction

Getting 4 frames out of an MP4 sounds trivial. It wasn't.

**Symptom:**

```text
OSError: [WinError 126] The specified module could not be found
```

This appeared when importing `torchcodec` on Windows. The library installed cleanly via pip; the error only appeared at import time.

**Wrong assumption:** I initially tried to fall back to `torchvision.io.read_video`, which is documented as the standard PyTorch video reader. What I didn't know: `torchvision.io.read_video` was removed in torchvision 0.26. The call silently fell through to a broken path.

**Root cause:** `torchcodec` requires a full FFmpeg shared library build — the real `.dll` files, not just the CLI binaries. The Windows installer for FFmpeg (and most conda builds) doesn't ship the shared libraries in a form that Python's `ctypes` can find at runtime. The `[WinError 126]` is Windows telling you it found the DLL name but couldn't resolve all its dependencies.

**Fix:** I threw out the Python media libraries entirely and rewrote frame extraction as ffmpeg subprocess calls piping raw PNG bytes back to Python:

```python
def _extract_frames(mp4_path: Path, n_frames: int = 4) -> list:
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    frames = []
    for i in range(n_frames):
        t = duration * (i + 1) / (n_frames + 1)
        cmd = ["ffmpeg", "-ss", str(t), "-i", str(mp4_path),
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        png = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
    return frames
```

The evenly-spaced timestamps (`duration * (i + 1) / (n_frames + 1)`) avoid the first and last frames, which are often black fades. Each frame comes back as a PIL Image ready to pass to the vision model.

**Lesson:** On Windows, native media libraries are a minefield. Python bindings to FFmpeg frequently assume a Unix-style shared library layout that doesn't exist on Windows. The subprocess approach is verbose and not elegant, but it works everywhere FFmpeg is installed as a CLI — which is a much broader set of machines.

## Hard part 2: Gemma 4 Q4 OOM with device_map="auto"

With frame extraction working, the next blocker was loading Gemma 4 E4B-IT at Q4 quantization on 8GB VRAM.

**Symptom:**

```text
torch.OutOfMemoryError: CUDA out of memory.
```

This was followed by layers falling back to CPU — and then bitsandbytes crashing because Q4 kernels have no CPU implementation.

**Wrong assumption:** `device_map="auto"` is supposed to be the smart option. Transformers inspects available memory, calculates a device map, and splits the model across CPU and GPU as needed. I assumed this would handle a tight VRAM situation gracefully.

**Root cause:** The assumption fails for quantized models. `device_map="auto"` with `load_in_4bit=True` will offload some layers to CPU RAM if VRAM is insufficient. But bitsandbytes Q4 quantization only compiles CUDA kernels — there is no CPU fallback path. The moment any layer lands on CPU, the forward pass crashes. The `device_map="auto"` logic doesn't account for this constraint.

A second issue: I was missing `bnb_4bit_compute_dtype=torch.float16`. Without it, bitsandbytes defaults to float32 compute even though the weights are 4-bit, doubling the activation memory needed during inference.

**Fix:**

```python
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # key: was missing
    ),
    device_map="cuda:0",      # force ALL layers on GPU
    torch_dtype=torch.float16,
)
```

Forcing `device_map="cuda:0"` puts every layer on the GPU. If it doesn't fit, you get an OOM at load time — which is honest. The alternative (silent CPU offload that crashes at inference time) is much harder to debug. With the correct `bnb_4bit_compute_dtype`, Gemma 4 E4B-IT Q4 fits comfortably in 8GB.

**Lesson:** When using bitsandbytes quantized models on a single consumer GPU, `device_map="cuda:0"` with explicit `torch_dtype` is safer than `device_map="auto"`. Auto is smart about splitting; it's not smart about which splits are actually valid for a given quantization backend.

## Hard part 3: Instagram rate-limiting and the cookies file

The download stage looked straightforward: pass a list of URLs to yt-dlp and let it run. It worked — for about 420 reels.

**Symptom (round 1):** After ~420 reels, every download returned:

```text
ERROR: [Instagram] <id>: Requested content is not available,
rate-limit reached or login required.
```

**Wrong assumption (round 1):** I configured `cookiesfrombrowser=("chrome",)` in the yt-dlp options, which is documented as the way to automatically grab session cookies from your running browser. I assumed this would maintain a valid session.

**Root cause (round 1):** Chrome locks its SQLite cookie database with an exclusive file lock while the browser is running. yt-dlp attempts to copy the database, but can't acquire the lock:

```text
Could not copy Chrome cookie database
```

So yt-dlp was running without any session cookie at all — anonymous, as far as Instagram was concerned. Anonymous users hit the rate limit much faster.

**Symptom (round 2):** After fixing the cookie issue by closing Chrome before running the pipeline, the same error recurred after a few hundred more reels. Instagram's rate-limiting is aggressive even for authenticated sessions doing bulk downloads.

**Fix:** Export cookies once using the "Get cookies.txt LOCALLY" browser extension (exports in Netscape format), save to a file, and point yt-dlp at it:

```python
ydl_opts = {
    "cookiefile": str(Path(__file__).parents[3] / "instagram_cookies.txt"),
    # ...
}
```

For the rate-limiting itself: I added randomized delays between downloads (15–45 seconds) and a longer backoff (10–30 minutes) whenever yt-dlp returned a rate-limit error. The pipeline re-queues failed downloads and retries on the next run. Over ~5 days with several multi-hour pauses, all downloadable reels completed.

**Lesson:** Cookie automation via browser process is fragile — file locks, browser updates, and session expiry all break it. A one-time manually exported cookie file is less convenient but far more reliable for bulk operations. And bulk scraping any platform's content requires respecting their rate limits; aggressive retries just make the backoff longer.

## Results

After ~5 days of pipeline runtime on an RTX 5070 Laptop (8GB VRAM), resuming across multiple sessions:

- **2,491 reels indexed** out of 2,695 attempted
- **~92% auto-classified** with high confidence — no human review needed
- **~8% manually triaged** via the FastAPI + Svelte review UI, where Gemma's confidence score fell below threshold (blurry content, unusual language, music-only reels)
- **~7.4% discarded** — deleted reels (404 from Instagram), private account content that became inaccessible, and reels with silent audio where faster-whisper returned empty transcripts

The vault now supports three query modes:
- Full-text search over transcripts and summaries (FTS5)
- Semantic search via BGE-M3 cosine similarity
- Hybrid (RRF fusion of both)

A reel I bookmarked 18 months ago about a specific climbing hold technique: found in 0.3 seconds with a semantic query. That's the whole point.

## What I learned

**Local LLM pipelines on consumer hardware are viable — but the surface area for platform-specific bugs is large.** Windows DLL issues, CUDA kernel limitations, and browser lock files are all problems you don't hit in a cloud notebook. Budget debugging time accordingly.

**SQLite is remarkably capable as an ML infrastructure layer.** FTS5 full-text search, vector similarity via sqlite-vec, and job tracking in the same database file that fits on a USB drive. SQLite's capabilities at personal-project scale exceeded my expectations.

**Confidence scores are worth the extra inference cost.** Having Gemma return a 0–1 confidence alongside each classification made the human review step tractable — I only had to look at ~200 reels instead of spot-checking all 2,491. A binary pass/fail from the model would have been much less useful.

**The boring infrastructure matters more than the models.** Resumability, per-reel state tracking, and graceful retry logic are what made a 5-day pipeline actually finish. The models are the interesting part on paper; the job queue is the part that made it work in practice.

Full case study — architecture diagram, tech stack, and links — at [proyecto-info-instagram-portfolio.vercel.app/projects/reelvault](https://proyecto-info-instagram-portfolio.vercel.app/projects/reelvault).
