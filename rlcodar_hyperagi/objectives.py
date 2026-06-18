"""
Masked Byte Diffusion Objective for RLCoDAR / HyperAGI.

This module adds the missing "training objective" side without introducing
Torch, NumPy, gradients, or external APIs.

Research basis:
- Masked/discrete diffusion language models train by corrupting tokens/bytes
  and learning to recover the original sequence.
- For a byte-native system, the vocabulary is exactly 256 values, so the
  corruption/recovery process can be expressed directly over raw bytes.

What this file provides:
- ByteMaskingConfig: objective knobs.
- MaskedByteDiffusionObjective: pure-Python masking, denoising evaluation,
  entropy scoring, and a prototype self_improve_step.

This is not gradient training. It is an objective harness that measures and
records whether the current byte index + CoDAR runtime can recover masked bytes.
HyperAgents can later use these metrics to decide what to index, re-index,
route, or modify.
"""

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .diffusion import ByteIndex, ByteGroupTokenizer, CoDARDiffusion


MASK_BYTE = 0


@dataclass
class ByteMaskingConfig:
    """Configuration for masked byte diffusion objective."""

    mask_ratio: float = 0.15
    random_replace_ratio: float = 0.10
    keep_original_ratio: float = 0.10
    canvas_length: int = 256
    entropy_bound: float = 0.1
    smart_masking: bool = True
    seed: Optional[int] = None


class MaskedByteDiffusionObjective:
    """
    Pure-Python masked byte diffusion objective.

    The objective is simple:
        original bytes -> corrupt/mask selected positions -> ask runtime to
        recover/denoise -> compare recovered bytes to original.

    It gives the repo an explicit diffusion-style objective while keeping the
    project byte-native and non-gradient.
    """

    def __init__(
        self,
        byte_index: ByteIndex,
        model: Optional[CoDARDiffusion] = None,
        tokenizer: Optional[ByteGroupTokenizer] = None,
        config: Optional[ByteMaskingConfig] = None,
    ):
        self.byte_index = byte_index
        self.tokenizer = tokenizer or ByteGroupTokenizer()
        self.model = model or CoDARDiffusion(byte_index=byte_index, tokenizer=self.tokenizer)
        self.config = config or ByteMaskingConfig()
        self.history: List[Dict] = []
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def _information_scores(self, byte_values: List[int]) -> List[float]:
        """
        Score positions for smart masking.

        Higher score means the byte is more structurally informative. This is a
        lightweight pure-Python approximation: rare bytes, punctuation, digits,
        uppercase letters, and code delimiters get higher priority.
        """
        if not byte_values:
            return []

        counts: Dict[int, int] = {}
        for b in byte_values:
            counts[b] = counts.get(b, 0) + 1

        scores = []
        n = len(byte_values)
        for b in byte_values:
            rarity = 1.0 - (counts[b] / n)
            structural = 0.0
            if b in b"{}[]()<>:=+-*/_.#,;\n\t":
                structural += 0.35
            if 48 <= b <= 57:  # digits
                structural += 0.15
            if 65 <= b <= 90:  # uppercase
                structural += 0.10
            if b > 127:       # non-ASCII / multimodal raw byte signal
                structural += 0.20
            scores.append(rarity + structural)
        return scores

    def choose_mask_positions(self, byte_values: List[int]) -> List[int]:
        """Choose byte positions to corrupt."""
        if not byte_values:
            return []
        k = max(1, int(len(byte_values) * self.config.mask_ratio))
        k = min(k, len(byte_values))

        if not self.config.smart_masking:
            return sorted(random.sample(range(len(byte_values)), k))

        scored = list(enumerate(self._information_scores(byte_values)))
        scored.sort(key=lambda x: -x[1])
        top_pool = [i for i, _ in scored[: max(k, min(len(scored), k * 4))]]
        return sorted(random.sample(top_pool, k)) if len(top_pool) > k else sorted(top_pool)

    def corrupt_bytes(self, byte_values: List[int]) -> Tuple[List[int], List[int]]:
        """
        Apply masked diffusion corruption.

        Selected positions are either replaced with MASK_BYTE, replaced with a
        random byte, or kept unchanged. This mirrors common masked-LM diffusion
        corruption recipes in a byte-native form.
        """
        corrupted = list(byte_values)
        positions = self.choose_mask_positions(byte_values)

        for pos in positions:
            r = random.random()
            if r < self.config.keep_original_ratio:
                continue
            if r < self.config.keep_original_ratio + self.config.random_replace_ratio:
                corrupted[pos] = random.randint(0, 255)
            else:
                corrupted[pos] = MASK_BYTE
        return corrupted, positions

    def _nearest_index_byte(self, query_byte: int) -> int:
        """
        Recover a byte using the current byte index.

        This is intentionally simple: it converts the byte to a single-byte
        embedding and finds the closest indexed byte group. Later, HyperAgents
        can replace this with better routing, model-family metadata, or tensor
        layer-aware indexing.
        """
        query_emb = self.tokenizer.group_to_embedding([query_byte])
        best_sim = -1.0
        best_byte = query_byte
        for entry in self.byte_index.index:
            sim = 0.0
            emb = entry.get("embedding")
            if emb:
                dot_val = sum(a * b for a, b in zip(query_emb, emb))
                norm_a = math.sqrt(sum(a * a for a in query_emb))
                norm_b = math.sqrt(sum(b * b for b in emb))
                if norm_a > 1e-12 and norm_b > 1e-12:
                    sim = dot_val / (norm_a * norm_b)
            if sim > best_sim and entry.get("group"):
                best_sim = sim
                best_byte = int(entry["group"][0])
        return best_byte

    def recover_bytes(self, corrupted: List[int], positions: List[int]) -> List[int]:
        """Recover masked positions using index-guided byte recovery."""
        recovered = list(corrupted)
        for pos in positions:
            left = recovered[pos - 1] if pos > 0 else MASK_BYTE
            right = recovered[pos + 1] if pos + 1 < len(recovered) else MASK_BYTE
            if corrupted[pos] == MASK_BYTE:
                # Use neighboring bytes as a minimal context signal.
                candidate = self._nearest_index_byte(left if left != MASK_BYTE else right)
            else:
                candidate = self._nearest_index_byte(corrupted[pos])
            recovered[pos] = candidate
        return recovered

    def score_recovery(self, original: List[int], recovered: List[int], positions: List[int]) -> Dict:
        """Measure byte recovery quality."""
        if not positions:
            return {"masked": 0, "correct": 0, "accuracy": 1.0, "byte_mae": 0.0}

        correct = 0
        total_error = 0
        for pos in positions:
            if int(original[pos]) == int(recovered[pos]):
                correct += 1
            total_error += abs(int(original[pos]) - int(recovered[pos]))

        return {
            "masked": len(positions),
            "correct": correct,
            "accuracy": correct / len(positions),
            "byte_mae": total_error / len(positions),
        }

    def train_step(self, text_or_bytes) -> Dict:
        """
        Run one objective step.

        This is the repo's first explicit diffusion-style objective:
        corrupt bytes, recover them, score reconstruction, and persist metrics.
        """
        if isinstance(text_or_bytes, str):
            original = list(text_or_bytes.encode("utf-8"))
        elif isinstance(text_or_bytes, (bytes, bytearray)):
            original = list(text_or_bytes)
        else:
            original = [int(x) & 0xFF for x in text_or_bytes]

        original = original[: self.config.canvas_length]
        corrupted, positions = self.corrupt_bytes(original)
        recovered = self.recover_bytes(corrupted, positions)
        metrics = self.score_recovery(original, recovered, positions)
        metrics.update({
            "objective": "masked_byte_diffusion",
            "config": asdict(self.config),
            "sample_length": len(original),
        })
        self.history.append(metrics)
        return metrics

    def self_improve_step(self, text_or_bytes, source_name: str = "objective_sample") -> Dict:
        """
        Run objective step, then add the original sample to the byte index.

        This gives HyperAgents a practical hook: if recovery is weak, add more
        source bytes and measure again.
        """
        metrics_before = self.train_step(text_or_bytes)

        if isinstance(text_or_bytes, str):
            self.byte_index.add_text(text_or_bytes, source_name=source_name)
        elif isinstance(text_or_bytes, (bytes, bytearray)):
            self.byte_index.add_text(bytes(text_or_bytes).decode("utf-8", errors="ignore"), source_name=source_name)
        else:
            raw = bytes([int(x) & 0xFF for x in text_or_bytes])
            self.byte_index.add_text(raw.decode("utf-8", errors="ignore"), source_name=source_name)

        metrics_after = self.train_step(text_or_bytes)
        return {
            "before": metrics_before,
            "after": metrics_after,
            "improved": metrics_after.get("accuracy", 0.0) >= metrics_before.get("accuracy", 0.0),
        }


if __name__ == "__main__":
    idx = ByteIndex()
    idx.add_text("Python uses bytes. CoDAR recovers masked byte positions.", source_name="seed")
    objective = MaskedByteDiffusionObjective(idx, config=ByteMaskingConfig(seed=42))
    print(objective.train_step("Python uses bytes. CoDAR recovers masked byte positions."))
