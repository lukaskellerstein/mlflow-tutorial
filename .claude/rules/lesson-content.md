---
globs: ["tutorial/**/README.md"]
---

# Lesson Content — README.md Format

Every lesson's README.md follows this structure:

## Template

```markdown
# L<level>-<Module.Lesson> — <Lesson Title>

**Level:** <Models | AI Agents | Advanced>
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
cd tutorial/<level_N_domain>/<module>/<lesson>
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

### Level 1 — Models
- Each topic is covered end-to-end (basic through advanced) in one place.
- Merged lessons may be longer — use clear Part 1/Part 2 sections.
- Show working examples for every concept.
- Explain tradeoffs when covering advanced patterns.

### Level 2 — AI Agents
- Assumes L1 knowledge — no re-teaching tracking, tracing, or evaluation basics.
- Focus on agent-specific patterns, frameworks, and evaluation.
- Cross-reference L1 concepts: "In L1-M4.1 you learned evaluation. Now we'll apply it to agents."
- Include agent trace analysis sections.

### Level 3 — Advanced
- Production-quality code with proper error handling.
- Include architecture diagrams where appropriate.
- Discuss scalability and performance implications.
- Capstone READMEs should include an "Architecture" section with system diagrams.

## General Guidelines

- Write for the target audience: ML engineers and data scientists who know Python but may be new to MLFlow.
- Explain concepts before showing code — don't just dump code.
- Use progressive disclosure: introduce one concept at a time.
- Always explain WHY, not just WHAT.
- Keep it practical — every concept should have a working code example.
- Include "Expected Output" so users can verify they got the right result.
- Cross-reference related lessons across levels.
