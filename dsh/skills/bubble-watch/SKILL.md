---
name: bubble-watch
description: The daily NVDA Bubble Signal Watch procedure — call the confirmed-data tools, delegate research to the Hermes analyst, reconcile the two views, and write the Korean report. Follow this whenever you are asked to produce the report.
---

# NVDA Bubble Signal Watch — daily procedure

You are both the **orchestrator and the second analyst**. The `mcp__bubble__*` tools compute the
numbers, `mcp__hermes__analyst` does the web research, and the judgement, reconciliation and
writing are yours.

Work in English. Only the report itself is written in Korean — it is the product.

## Budget

A run should take minutes, not tens of minutes. Research is the only expensive part, so:

- **Prefer one `web_search` over many `web_fetch` calls.** Fetch a page only when you need something
  a search result does not already give you.
- **At most 8 page fetches for your own view.** Hermes is researching in parallel and returns its
  sources with URLs; verify the claims you actually dispute rather than re-reading everything.
- **Do not explore the repository.** No `glob`, `grep`, `read`, `bash` or `edit`. The only file you
  touch is the report you write. Everything you need arrives from `mcp__bubble__*` and the analyst.
- Keep `todo_write` to one plan at the start, if you use it at all.

## Absolute rules

1. **Never invent a number.** Closes, returns, IV and put changes come only from what
   `mcp__bubble__*` returned. Do not compute them yourself or fill them from memory. If a tool says
   `N/A`, the report says `N/A`.
2. **No value without a source.** Anything found by research carries a URL you actually opened and
   a freshness label.
3. **Never hide a degradation.** If `mcp__hermes__analyst` failed, or gaps remain, say so in
   the report's provenance section.

## Procedure

### 1. Context and data

```
mcp__bubble__prior_state(date)      → previous score, verdict, notes
mcp__bubble__prepare_brief(date)    → closes, puts, signals, gaps, trace_id
```

A null `signals` means the ticker's own close is missing — step 2 is then mandatory.
Keep the `trace_id`: it goes in the provenance section.

### 2. Fill the gaps (only if `gaps` is non-empty)

Delegate each `field` in `gaps`. State **exactly which value you need** and that **a source URL and
freshness label are required**.

```
mcp__hermes__analyst(task="Find the IV of the 2026-09-18 NVDA $220 put and the SOXL close.
                                 Report each value as JSON with its source URL and freshness
                                 (EOD | latest_snapshot | late_session_last).")
mcp__bubble__apply_gap_fills(date, fills=[{field, value, source_url, freshness}, ...])
```

A non-empty `rejected` means that value was not stored. Retry once; if it still fails, leave it
`N/A` and record it in the provenance.

### 3. Build two views

**(a) The Hermes view** — delegate the research. Paste the computed signals in verbatim and tell it
not to recompute them.

```
mcp__hermes__analyst(task="<computed signals JSON>. Research the catalysts published since
                                 the last record and attach a URL to each. Interpret this tape and
                                 give a Bubble Signal Score (0-10) and a verdict
                                 (NOT_TRIGGERED | TRIGGERED | TRIGGERED_DE_CONFIRMING |
                                  TRIGGERED_FURTHER_DE_CONFIRMING | CONFIRMED).")
```

**(b) Your own view** — form it independently, with your own web search, **before** you look at the
Hermes answer. Express it in the same score/verdict form.

### 4. Reconcile

- **Agreement**: score difference <= 0.5 **and** identical verdict. Final score is the mean of the
  two; the verdict is the shared one.
- **Disagreement**: anything else. Run **one rebuttal round** — show
  `mcp__hermes__analyst` the opposing view (yours) and let it reconsider, and reconsider
  yours in light of its argument. Pass the `session_id` back so it continues the same conversation
  rather than re-reading the whole brief.
  - If you still disagree after that: final score is the mean, and the verdict is the **more
    cautious** of the two. Most cautious first: `CONFIRMED` > `TRIGGERED` >
    `TRIGGERED_DE_CONFIRMING` > `TRIGGERED_FURTHER_DE_CONFIRMING` > `NOT_TRIGGERED`.
- **Hermes failed**: proceed on your own view alone and record "single judgement" in the provenance.
- If the score moved by 1.0 or more from the previous record, say why in the report.

## 5. Write the report

Write Korean markdown to the path given in the task. Structure:

1. `# NVDA Bubble Signal Watch — <date>` and a one-line headline
2. **Verdict**: score (with the delta from the previous record) and verdict, plus a paragraph of
   reasoning
3. **Tape**: close table (NVDA/SMH/SOXL, 1-day / 3-day / anchor returns, NVDA-SMH relative spread)
4. **Options**: put table per strike (bid/ask/last/IV/volume/OI, 1-day change), and the IV comparison
5. **The three re-confirmation conditions**: `nvda_underperforms`, `iv_surface_up`, `far_otm_leads`,
   each with whether it holds. Write `partial` as "can't tell" — Alpha Vantage reports IV on a
   ~0.98%p grid, so a rise within one step is indistinguishable from noise.
6. **Catalysts**: the catalysts both views cited, merged, with URLs, marking who cited what
7. **What to watch tomorrow**
8. **Provenance**: data sources and freshness, how the two views were reconciled (agreement /
   agreement after rebuttal / disagreement / single judgement), both analysts' raw scores, anything
   that failed or is still `N/A`, and the `trace_id` from `prepare_brief` on its own line as
   `trace: <32 hex characters>`. The report alone must be enough to find this run in Arize.

Every cell in every table is a value a tool returned. Field names (`iv_surface_up`, `rel_spread`
and so on) are translated into Korean prose everywhere except the condition list.

## 6. Save

```
mcp__bubble__save_run(date, score, verdict, note, report_path)
```

`note` is one or two sentences summarising the day — tomorrow's `prior_state` reads it.
