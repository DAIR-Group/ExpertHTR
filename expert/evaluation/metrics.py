"""evaluation.metrics phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def register_metrics(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    #  CER / WER + MetricTracker
    # ─────────────────────────────────────────────────────────────────────────────
    from typing import List, Dict

    try:
        import editdistance
        def _levenshtein(a, b) -> int:
            return int(editdistance.distance(a, b))
    except ImportError:
        def _levenshtein(a, b) -> int:
            a, b = tuple(a), tuple(b)
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, n + 1):
                    tmp = dp[j]
                    dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
                    prev = tmp
            return dp[n]


    def character_error_rate(reference: str, hypothesis: str) -> float:
        ref, hyp = reference.strip(), hypothesis.strip()
        if len(ref) == 0:
            return 0.0 if len(hyp) == 0 else 1.0
        return _levenshtein(tuple(ref), tuple(hyp)) / len(ref)


    def word_error_rate(reference: str, hypothesis: str) -> float:
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        return _levenshtein(tuple(ref_words), tuple(hyp_words)) / len(ref_words)


    _PAGE_HTR_STRUCTURE_PREFIX_RE = re.compile(
        r"(?m)^[ \t]*\[(?:Para|R)\d+\]:[ \t]*"
    )
    _PAGE_HTR_ATOMIC_CER_MARKERS = {
        "<del>": "\ue000",
        "<gap>": "\ue001",
    }


    def page_htr_text_for_cer(text: str) -> str:
        """Remove synthetic region labels and count each OCR marker as one character."""
        value = str(text).replace("\r\n", "\n").replace("\r", "\n")
        value = _PAGE_HTR_STRUCTURE_PREFIX_RE.sub("", value)
        for marker, atomic_character in _PAGE_HTR_ATOMIC_CER_MARKERS.items():
            value = value.replace(marker, atomic_character)
        return value.strip()


    class MetricTracker:
        """Accumulate predictions/references and compute CER/WER."""
        def __init__(self):
            self.predictions: List[str] = []
            self.references: List[str] = []
            self.losses: List[float] = []
            self.counts: int = 0

        def update(self, predictions: List[str], references: List[str],
                   loss: float = None, batch_size: int = 1):
            self.predictions.extend(predictions)
            self.references.extend(references)
            if loss is not None:
                self.losses.append(loss)
            self.counts += batch_size

        def compute(self) -> Dict[str, float]:
            if not self.predictions:
                return {'cer': 0.0, 'wer': 0.0, 'avg_loss': 0.0, 'num_samples': 0}
            total_chars = total_char_errors = 0
            total_words = total_word_errors = 0
            for ref, hyp in zip(self.references, self.predictions):
                rt, ht = ref.strip(), hyp.strip()
                total_chars       += len(rt)
                total_char_errors += _levenshtein(tuple(rt), tuple(ht))
                rw = rt.split(); hw = ht.split()
                total_words       += len(rw)
                total_word_errors += _levenshtein(tuple(rw), tuple(hw))
            return {
                'cer': total_char_errors / max(total_chars, 1),
                'wer': total_word_errors / max(total_words, 1),
                'avg_loss': np.mean(self.losses) if self.losses else 0.0,
                'num_samples': len(self.predictions),
            }

        def reset(self):
            self.predictions, self.references, self.losses, self.counts = [], [], [], 0

    return export_runtime_locals(state, locals())
