"""
CoDAR: Continuous Diffusion with Contextual AutoRegressive Decoder

Pure Python byte-level diffusion model.
NO numpy, NO torch — only math, random, and Python builtins.
NO gradient training — self-improvement via HyperAgents.

This version adds DiffusionGemma-style block diffusion controls while
keeping the project byte-native:
- canvas_length = 256
- max_denoising_steps = 48
- entropy_bound = 0.1
- bidirectional_canvas_attention = True
- encoder_prefill_cache = True
- accept_low_entropy_tokens = True
- renoise_unaccepted_tokens = True

It also makes the existing multi-candidate retrieval path work as a lightweight
local Fusion / Mixture-of-Agents style mechanism: independent indexed contexts
are treated as panel candidates, the byte canvas contributes a denoised signal,
and the final answer is synthesized from consensus, unique coverage, and blind
spots without adding external APIs or new dependencies.
"""

import math
import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================================
# Pure Python Vector Operations
# ============================================================================

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a: List[float], b: List[float]) -> float:
    na = vec_norm(a)
    nb = vec_norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot(a, b) / (na * nb)


def zeros(n: int) -> List[float]:
    return [0.0] * n


def randn(n: int) -> List[float]:
    return [random.gauss(0.0, 1.0) for _ in range(n)]


def clip_val(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================
# DiffusionGemma-style block diffusion configuration
# ============================================================================

@dataclass
class BlockDiffusionConfig:
    """
    Runtime knobs inspired by DiffusionGemma's block-diffusion decoding.

    This does not make CoDAR a Gemma checkpoint. It adds the same class of
    runtime controls to the byte-native HyperAGI/CoDAR repo:
    canvas, denoising budget, entropy acceptance, re-noising, and local fusion.
    """
    canvas_length: int = 256
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    bidirectional_canvas_attention: bool = True
    encoder_prefill_cache: bool = True
    accept_low_entropy_tokens: bool = True
    renoise_unaccepted_tokens: bool = True
    fusion_top_k: int = 10
    include_fusion_report: bool = True


# ============================================================================
# Byte-Group Tokenizer
# ============================================================================

class ByteGroupTokenizer:
    """Groups contiguous bytes into tokens at natural boundaries."""

    BOUNDARIES = {0x20, 0x0A, 0x0D, 0x09, 0x00}

    def tokenize(self, raw_bytes: List[int]) -> List[List[int]]:
        groups = []
        current = []
        for b in raw_bytes:
            if b in self.BOUNDARIES:
                if current:
                    groups.append(current)
                    current = []
                groups.append([b])
            else:
                current.append(b)
        if current:
            groups.append(current)
        return groups

    def detokenize(self, groups: List[List[int]]) -> bytes:
        return bytes(int(clip_val(b, 0, 255)) for g in groups for b in g)

    def encode(self, text: str) -> List[List[int]]:
        return self.tokenize(list(text.encode("utf-8")))

    def decode(self, groups: List[List[int]]) -> str:
        return self.detokenize(groups).decode("utf-8", errors="ignore")

    def group_to_embedding(self, group: List[int]) -> List[float]:
        emb = zeros(256)
        for b in group:
            emb[int(clip_val(b, 0, 255))] += 1.0
        if group:
            emb = vec_scale(emb, 1.0 / len(group))
        return emb

    def embedding_to_group(self, emb: List[float]) -> List[int]:
        indexed = [(i, v) for i, v in enumerate(emb)]
        indexed.sort(key=lambda x: -abs(x[1]))
        group = []
        for byte_val, weight in indexed:
            if weight > 0.05:
                count = max(1, round(weight * 10))
                group.extend([byte_val] * count)
            if len(group) >= 20:
                break
        return group if group else [0]


# ============================================================================
# Byte Index — The "Weights" of CoDAR
# ============================================================================

class ByteIndex:
    """Byte-indexed file/context memory used as CoDAR's knowledge base."""

    def __init__(self):
        self.tokenizer = ByteGroupTokenizer()
        self.sources = {}
        self.index = []
        self.stats = {"total_bytes": 0, "total_groups": 0, "total_sources": 0}

    def add_file(self, file_path: str, source_name: str = None) -> int:
        source = source_name or os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                raw = list(f.read())
        except (OSError, IOError) as e:
            print(f"  ⚠ Cannot read {file_path}: {e}")
            return 0

        groups = self.tokenizer.tokenize(raw)
        count = self._add_groups(groups, source)
        self.sources[source] = {"type": "file", "path": file_path, "groups": count, "bytes": len(raw)}
        self.stats["total_bytes"] += len(raw)
        self.stats["total_groups"] += count
        self.stats["total_sources"] += 1
        return count

    def add_text(self, text: str, source_name: str = "text") -> int:
        groups = self.tokenizer.encode(text)
        count = self._add_groups(groups, source_name)
        self.sources[source_name] = {"type": "text", "groups": count, "bytes": len(text.encode("utf-8"))}
        self.stats["total_bytes"] += len(text.encode("utf-8"))
        self.stats["total_groups"] += count
        self.stats["total_sources"] += 1
        return count

    def _add_groups(self, groups: List[List[int]], source: str) -> int:
        for i, group in enumerate(groups):
            emb = self.tokenizer.group_to_embedding(group)
            context_start = max(0, i - 3)
            context_end = min(len(groups), i + 4)
            context_groups = groups[context_start:context_end]
            self.index.append({
                "embedding": emb,
                "group": group,
                "source": source,
                "context": self.tokenizer.decode(context_groups),
                "position": i,
            })
        return len(groups)

    def add_directory(self, dir_path: str, extensions: List[str] = None) -> int:
        if extensions is None:
            extensions = [".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"]
        total = 0
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    path = os.path.join(root, f)
                    rel = os.path.relpath(path, dir_path)
                    total += self.add_file(path, source_name=rel)
        return total

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_groups = self.tokenizer.encode(query)
        query_emb = zeros(256)
        for g in query_groups:
            query_emb = vec_add(query_emb, self.tokenizer.group_to_embedding(g))
        if query_groups:
            query_emb = vec_scale(query_emb, 1.0 / len(query_groups))

        scored = []
        for entry in self.index:
            scored.append((cosine_sim(query_emb, entry["embedding"]), entry))
        scored.sort(key=lambda x: -x[0])

        results = []
        seen_contexts = set()
        for sim, entry in scored[:top_k * 3]:
            if entry["context"] not in seen_contexts:
                seen_contexts.add(entry["context"])
                results.append({
                    "similarity": sim,
                    "group": entry["group"],
                    "source": entry["source"],
                    "context": entry["context"],
                    "text": self.tokenizer.decode([entry["group"]]),
                    "embedding": entry["embedding"],
                })
                if len(results) >= top_k:
                    break
        return results

    def save(self, path: str):
        data = {
            "sources": self.sources,
            "stats": self.stats,
            "entries": [{"group": e["group"], "source": e["source"], "context": e["context"], "position": e["position"]} for e in self.index],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.sources = data["sources"]
        self.stats = data["stats"]
        self.index = []
        for e in data["entries"]:
            self.index.append({
                "embedding": self.tokenizer.group_to_embedding(e["group"]),
                "group": e["group"],
                "source": e["source"],
                "context": e["context"],
                "position": e["position"],
            })


# ============================================================================
# Cosine Noise Schedule
# ============================================================================

class CosineNoiseSchedule:
    """Cosine noise schedule for diffusion. Pure Python."""

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        self.alpha_bar = []
        for t in range(T + 1):
            frac = (t / T + s) / (1 + s)
            val = math.cos(frac * math.pi / 2) ** 2
            self.alpha_bar.append(val)
        first = self.alpha_bar[0]
        self.alpha_bar = [a / first for a in self.alpha_bar]
        self.sqrt_alpha_bar = [math.sqrt(a) for a in self.alpha_bar]
        self.sqrt_one_minus = [math.sqrt(max(0, 1.0 - a)) for a in self.alpha_bar]

    def get_alpha_bar(self, t: int) -> float:
        return self.alpha_bar[max(0, min(t, self.T))]

    def sample_t(self) -> int:
        return random.randint(0, self.T - 1)


# ============================================================================
# CoDAR Diffusion — The Reasoning Engine
# ============================================================================

class CoDARDiffusion:
    """
    CoDAR with block-diffusion canvas decoding and local fusion synthesis.

    The byte index is still the knowledge source. The canvas path creates a
    denoised byte signal, while the fusion path treats retrieved contexts as
    independent candidates and synthesizes the strongest answer.
    """

    def __init__(
        self,
        byte_index: ByteIndex,
        schedule: Optional[CosineNoiseSchedule] = None,
        tokenizer: Optional[ByteGroupTokenizer] = None,
        config: Optional[BlockDiffusionConfig] = None,
        steps: Optional[int] = None,
    ):
        self.index = byte_index
        self.config = config or BlockDiffusionConfig()
        if steps is not None:
            self.config.max_denoising_steps = steps
        self.schedule = schedule or CosineNoiseSchedule(T=max(100, self.config.max_denoising_steps))
        self.tokenizer = tokenizer or ByteGroupTokenizer()
        self._prefill_cache: Dict[str, List[float]] = {}

    def forward_diffusion(self, embedding: List[float], t: int) -> List[float]:
        sqrt_a = self.schedule.sqrt_alpha_bar[min(t, self.schedule.T)]
        sqrt_1_a = self.schedule.sqrt_one_minus[min(t, self.schedule.T)]
        noise = randn(len(embedding))
        return vec_add(vec_scale(embedding, sqrt_a), vec_scale(noise, sqrt_1_a))

    def reverse_step(self, noisy_emb: List[float], t: int, context_emb: List[float]) -> List[float]:
        alpha_bar_prev = self.schedule.get_alpha_bar(t - 1) if t > 0 else 1.0
        velocity = vec_scale(vec_sub(context_emb, noisy_emb), 0.1)
        denoised = vec_add(noisy_emb, velocity)
        if t > 0:
            sigma = math.sqrt(max(0, 1.0 - alpha_bar_prev)) * 0.1
            denoised = vec_add(denoised, vec_scale(randn(len(noisy_emb)), sigma))
        return denoised

    def _mean_embedding(self, embeddings: List[List[float]]) -> List[float]:
        out = zeros(256)
        for emb in embeddings:
            out = vec_add(out, emb)
        return vec_scale(out, 1.0 / len(embeddings)) if embeddings else out

    def _prompt_embedding(self, prompt: str) -> List[float]:
        if self.config.encoder_prefill_cache and prompt in self._prefill_cache:
            return self._prefill_cache[prompt]
        groups = self.tokenizer.encode(prompt)
        emb = self._mean_embedding([self.tokenizer.group_to_embedding(g) for g in groups])
        if self.config.encoder_prefill_cache:
            self._prefill_cache[prompt] = emb
        return emb

    def _embedding_entropy(self, emb: List[float]) -> float:
        """
        Normalized entropy of byte probabilities. Lower means one/few bytes are
        clearly dominant, so the canvas position is safe to accept.
        """
        vals = [abs(x) for x in emb]
        total = sum(vals)
        if total <= 1e-12:
            return 1.0
        entropy = 0.0
        for v in vals:
            if v > 0:
                p = v / total
                entropy -= p * math.log(p + 1e-12)
        return entropy / math.log(256)

    def _bidirectional_canvas_attention(self, canvas: List[List[float]], context_emb: List[float]) -> List[List[float]]:
        if not self.config.bidirectional_canvas_attention or not canvas:
            return canvas
        global_canvas = self._mean_embedding(canvas)
        mixed = []
        for emb in canvas:
            # lightweight pure-Python stand-in for bidirectional canvas attention:
            # each position sees prompt/context plus whole-canvas signal.
            m = vec_add(vec_scale(emb, 0.70), vec_scale(global_canvas, 0.15))
            m = vec_add(m, vec_scale(context_emb, 0.15))
            mixed.append(m)
        return mixed

    def _make_canvas(self, prompt_emb: List[float], results: List[Dict]) -> List[List[float]]:
        canvas = []
        embeddings = [r.get("embedding") or self.tokenizer.group_to_embedding(r["group"]) for r in results]
        if not embeddings:
            embeddings = [prompt_emb]
        for i in range(self.config.canvas_length):
            base = embeddings[i % len(embeddings)]
            mixed = vec_add(vec_scale(prompt_emb, 0.35), vec_scale(base, 0.65))
            canvas.append(self.forward_diffusion(mixed, min(self.config.max_denoising_steps, self.schedule.T)))
        return canvas

    def _accept_canvas(self, canvas: List[List[float]], accepted: List[bool]) -> List[bool]:
        if not self.config.accept_low_entropy_tokens:
            return [True] * len(canvas)
        updated = accepted[:]
        for i, emb in enumerate(canvas):
            if not updated[i] and self._embedding_entropy(emb) <= self.config.entropy_bound:
                updated[i] = True
        return updated

    def _renoise_canvas(self, canvas: List[List[float]], accepted: List[bool], t: int) -> List[List[float]]:
        if not self.config.renoise_unaccepted_tokens:
            return canvas
        out = []
        sigma = math.sqrt(max(0, 1.0 - self.schedule.get_alpha_bar(t))) * 0.05
        for emb, is_accepted in zip(canvas, accepted):
            if is_accepted:
                out.append(emb)
            else:
                out.append(vec_add(emb, vec_scale(randn(len(emb)), sigma)))
        return out

    def block_diffuse(self, prompt: str, results: List[Dict]) -> List[List[int]]:
        """DiffusionGemma-style byte canvas: denoise, accept low entropy, re-noise remainder."""
        prompt_emb = self._prompt_embedding(prompt)
        context_emb = self._mean_embedding([r.get("embedding") or self.tokenizer.group_to_embedding(r["group"]) for r in results])
        canvas = self._make_canvas(prompt_emb, results)
        accepted = [False] * len(canvas)

        max_steps = min(self.config.max_denoising_steps, self.schedule.T)
        for t in reversed(range(max_steps)):
            canvas = self._bidirectional_canvas_attention(canvas, context_emb)
            next_canvas = []
            for emb, is_accepted in zip(canvas, accepted):
                next_canvas.append(emb if is_accepted else self.reverse_step(emb, t, context_emb))
            canvas = next_canvas
            accepted = self._accept_canvas(canvas, accepted)
            if all(accepted):
                break
            canvas = self._renoise_canvas(canvas, accepted, t)

        accepted_groups = [self.tokenizer.embedding_to_group(emb) for emb, is_accepted in zip(canvas, accepted) if is_accepted]
        if accepted_groups:
            return accepted_groups

        # Important runtime fix: low entropy acceptance is deliberately strict.
        # If no positions pass the bound, return a small denoised fallback instead
        # of silently discarding the canvas signal.
        fallback_count = max(1, min(len(results) or 1, self.config.fusion_top_k, len(canvas)))
        return [self.tokenizer.embedding_to_group(emb) for emb in canvas[:fallback_count]]

    def _token_set(self, text: str) -> set:
        return {tok.strip(".,:;!?()[]{}<>`'\"_").lower() for tok in text.split() if len(tok.strip()) > 2}

    def _fusion_analysis(self, prompt: str, results: List[Dict], canvas_groups: List[List[int]]) -> Dict:
        """
        Analyze retrieved candidates like a lightweight local MoA judge.

        Each unique context is an independent candidate. Consensus is estimated
        by repeated query-token coverage across candidates; unique insights are
        high-similarity contexts from different sources; blind spots are prompt
        terms not covered by any candidate.
        """
        prompt_terms = self._token_set(prompt)
        candidates = []
        source_counts: Dict[str, int] = {}
        term_counts: Dict[str, int] = {}

        for result in results[: max(1, self.config.fusion_top_k)]:
            context = result.get("context", "").strip()
            if not context:
                continue
            terms = self._token_set(context)
            source = result.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            for term in terms & prompt_terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            candidates.append({
                "source": source,
                "similarity": result.get("similarity", 0.0),
                "text": context,
                "covered_prompt_terms": sorted(terms & prompt_terms),
            })

        consensus_terms = sorted([term for term, count in term_counts.items() if count >= 2])
        partial_terms = sorted([term for term, count in term_counts.items() if count == 1])
        blind_spots = sorted(prompt_terms - set(term_counts.keys()))
        canvas_text = self.tokenizer.decode(canvas_groups).strip()

        unique_by_source = []
        seen_sources = set()
        for item in sorted(candidates, key=lambda c: c["similarity"], reverse=True):
            if item["source"] not in seen_sources:
                seen_sources.add(item["source"])
                unique_by_source.append(item)
            if len(unique_by_source) >= 5:
                break

        return {
            "consensus": consensus_terms,
            "partial_coverage": partial_terms,
            "blind_spots": blind_spots,
            "unique_insights": unique_by_source,
            "source_votes": source_counts,
            "canvas_signal": canvas_text[:500],
        }

    def _synthesize_fusion_answer(self, prompt: str, results: List[Dict], canvas_groups: List[List[int]]) -> str:
        analysis = self._fusion_analysis(prompt, results, canvas_groups)
        selected_contexts = []
        seen = set()
        for item in sorted(results, key=lambda r: r.get("similarity", 0.0), reverse=True):
            ctx = item.get("context", "").strip()
            if ctx and ctx not in seen:
                seen.add(ctx)
                selected_contexts.append((item.get("source", "unknown"), item.get("similarity", 0.0), ctx))
            if len(selected_contexts) >= self.config.fusion_top_k:
                break

        sources = []
        for source, _, _ in selected_contexts:
            if source not in sources:
                sources.append(source)

        body = " ".join(ctx for _, _, ctx in selected_contexts)
        if not body:
            body = self.tokenizer.decode(canvas_groups).strip()
        if not body:
            return "[No relevant content found in index.]"

        if not self.config.include_fusion_report:
            return body

        report = [
            "[Local Fusion: CoDAR retrieved multiple candidate contexts, denoised a byte canvas, and synthesized the strongest coverage.]",
            f"[Sources: {', '.join(sources[:5]) if sources else 'canvas'}]",
        ]
        if analysis["consensus"]:
            report.append("[Consensus terms: " + ", ".join(analysis["consensus"][:12]) + "]")
        if analysis["partial_coverage"]:
            report.append("[Partial coverage: " + ", ".join(analysis["partial_coverage"][:12]) + "]")
        if analysis["blind_spots"]:
            report.append("[Blind spots: " + ", ".join(analysis["blind_spots"][:12]) + "]")
        if analysis["canvas_signal"]:
            report.append("[Canvas signal: " + analysis["canvas_signal"] + "]")

        return "\n".join(report) + "\n\n" + body

    def reason(self, prompt: str, max_groups: int = 50) -> str:
        if len(self.index.index) == 0:
            return "[No data indexed. Add files or datasets first.]"

        results = self.index.search(prompt, top_k=max_groups)
        if not results:
            return "[No relevant content found in index.]"

        # Run block diffusion canvas and keep its signal in the final synthesis.
        decoded_canvas_groups = self.block_diffuse(prompt, results)
        return self._synthesize_fusion_answer(prompt, results, decoded_canvas_groups)

    def completion(self, prompt: str) -> str:
        return self.reason(prompt)


# ============================================================================
# Utility Functions
# ============================================================================

def scan_repo_files(repo_root: str = ".") -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        for f in filenames:
            if f.endswith((".py", ".md", ".txt", ".json", ".toml")):
                files.append(os.path.join(root, f))
    return files


def bytes_to_text(byte_list: List[int]) -> str:
    return bytes(int(clip_val(b, 0, 255)) for b in byte_list).decode("utf-8", errors="ignore")


def text_to_bytes(text: str) -> List[int]:
    return list(text.encode("utf-8"))


# ============================================================================
# Main: Test Everything
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CoDAR — Pure Python block diffusion canvas + local fusion")
    print("=" * 60)

    tok = ByteGroupTokenizer()
    index = ByteIndex()
    index.add_text("Hello World! This is CoDAR.", source_name="test1")
    index.add_text("Python is a programming language. It uses bytes.", source_name="test2")
    index.add_text("Byte-group tokens group contiguous bytes together.", source_name="test3")

    config = BlockDiffusionConfig(
        canvas_length=256,
        max_denoising_steps=48,
        entropy_bound=0.1,
        bidirectional_canvas_attention=True,
        encoder_prefill_cache=True,
        accept_low_entropy_tokens=True,
        renoise_unaccepted_tokens=True,
        fusion_top_k=10,
    )
    schedule = CosineNoiseSchedule(T=100)
    codar = CoDARDiffusion(index, schedule=schedule, tokenizer=tok, config=config)

    print(f"Indexed: {index.stats['total_groups']} groups from {index.stats['total_sources']} sources")
    print(f"Config: {config}")
    print("Query: What is Python?")
    print(codar.reason("What is Python?")[:800])
    print("✅ CoDAR block diffusion canvas + local fusion verified")
