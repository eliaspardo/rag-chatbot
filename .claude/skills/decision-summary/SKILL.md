# Skill: Weekly Decision Summary

## Purpose
Summarize architectural and design decisions made in the past week.

## Input
- ADR log entries from the past 7 days (from CLAUDE.md ADR section or ADR files)
- Any notable context shifts or reversals

## Output format
Produce a markdown section with:

### Week of [DATE]
**Decisions made:** [count]

| # | Decision | Rationale | Impact |
|---|----------|-----------|--------|
| 1 | ... | ... | Low/Med/High |

**Key themes this week:** [2-3 bullet points]
**Open questions carried forward:** [if any]

## Save location
After generating the summary, save it to `docs/weekly-summary/` with filename format:
`YYYY-MM-DD_to_YYYY-MM-DD.md`

Example: `docs/weekly-summary/2026-03-13_to_2026-03-20.md`

Include a heading `# Weekly Decision Summary` at the top of the file.
