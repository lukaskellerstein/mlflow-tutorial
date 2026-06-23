---
globs: ["tutorial/**/README.md"]
---

# Lesson Content — README.md Format

Every lesson's README.md follows this structure:

## Template

```markdown
# L<level>-<Module.Lesson> — <Lesson Title>

**Level:** <Essentials | Practitioner | Expert>
**Duration:** <estimated time>

## Overview
<2-3 sentences: what the user will learn and why it matters>

## Prerequisites
- Completed: <list prior lessons, including cross-level prerequisites>
- MLFlow server running at http://127.0.0.1:5000
- <any additional requirements>

## Concepts
<Explain the key concepts before diving into code. Use clear, concise prose.
This section teaches the "why" — what problem does this solve?>

## Step-by-Step

### Step 1: <Action>
<Explain what we're doing and why>

```python
# Key code snippet (from main.py)
```

### Step 2: <Action>
...

## Running the Lesson

```bash
cd tutorial/level_<N>/<module>/<lesson>
uv sync
uv run python main.py
```

## Expected Output
<Show what the user should see in the terminal and in MLFlow UI>

## Key Takeaways
- <3-5 bullet points summarizing what was learned>

## Next Steps
<Point to the next lesson and preview what it covers.
For end-of-level lessons, point to the next level.>
```

## Writing Guidelines by Level

### Level 1 — Essentials
- Keep it short. One concept per lesson.
- Show the simplest possible working example.
- Don't go into edge cases or advanced options.
- End with: "In Level 2, we'll explore this in more depth."

### Level 2 — Practitioner
- Build real-world scenarios, not toy examples.
- Explain tradeoffs and design decisions.
- Include comparison tables (e.g., chunking strategies, model configurations).
- Cross-reference Level 1 concepts: "In L1-M4.1 you saw basic evaluation. Now we'll build custom metrics."

### Level 3 — Expert
- Production-quality code with proper error handling.
- Include architecture diagrams where appropriate.
- Discuss scalability and performance implications.
- For agent lessons, always include a section on interpreting traces in MLflow UI.
- Capstone READMEs should include a "Architecture" section with system diagrams.

## General Guidelines

- Write for the target audience: ML engineers and data scientists who know Python but may be new to MLFlow.
- Explain concepts before showing code — don't just dump code.
- Use progressive disclosure: introduce one concept at a time.
- Always explain WHY, not just WHAT.
- Keep it practical — every concept should have a working code example.
- Include "Expected Output" so users can verify they got the right result.
- Cross-reference related lessons across levels.
