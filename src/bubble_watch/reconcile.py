"""Pure reconciliation of analyst views; the only LLM involvement is the rebuttal round the graph runs."""
from __future__ import annotations

from .models import AnalystView, MergedCatalyst, Reconciliation, most_cautious, round1

AGREE_SCORE_GAP = 0.5
BIG_MOVE = 1.0


def views_agree(a: AnalystView, b: AnalystView) -> bool:
    return abs(a.score - b.score) <= AGREE_SCORE_GAP + 1e-9 and a.verdict == b.verdict


def merge_catalysts(views: dict[str, AnalystView]) -> list[MergedCatalyst]:
    merged: dict[str, MergedCatalyst] = {}
    for agent, view in views.items():
        for c in view.catalysts:
            if c.url not in merged:
                merged[c.url] = MergedCatalyst(**c.model_dump())
            if agent not in merged[c.url].cited_by:
                merged[c.url].cited_by.append(agent)
    return list(merged.values())


def reconcile(views: dict[str, AnalystView], *, prior_score: float | None,
              rebuttal_done: bool) -> Reconciliation | None:
    """Return the reconciliation, or None when the two views disagree and no rebuttal has run yet."""
    if not views:
        raise ValueError("no analyst views to reconcile")
    flags = [f"{a}: score moved {v.score - prior_score:+.1f} from prior {prior_score}"
             for a, v in views.items() if prior_score is not None and abs(v.score - prior_score) > BIG_MOVE]
    common = dict(scores={a: v.score for a, v in views.items()}, verdicts={a: v.verdict for a, v in views.items()},
                  rebuttal_round=rebuttal_done, catalysts=merge_catalysts(views))
    if len(views) == 1:
        (agent, view), = views.items()
        return Reconciliation(mode="single", score=view.score, verdict=view.verdict,
                              flags=[*flags, f"single analyst: {agent}"], **common)
    a, b = list(views.values())[:2]
    mean = round1((a.score + b.score) / 2)
    if views_agree(a, b):
        return Reconciliation(mode="agree_after_rebuttal" if rebuttal_done else "agree", score=mean,
                              verdict=a.verdict, flags=flags, **common)
    if not rebuttal_done:
        return None
    return Reconciliation(mode="disagree", score=mean, verdict=most_cautious(a.verdict, b.verdict),
                          flags=flags, **common)
