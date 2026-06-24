"""
Code quality scorers for evaluating generated code.

Rule-based scorers that provide fast, deterministic evaluation without
requiring an LLM judge.
"""

import re


def score_code_completeness(code: str, prompt: str) -> float:
    """Score 0-1: does the code contain expected structural elements?"""
    checks = [
        "def " in code or "class " in code,      # has function or class
        "return " in code,                         # returns a value
        len(code.strip().splitlines()) >= 3,       # non-trivial length
        ":" in code,                               # has blocks
    ]
    return sum(checks) / len(checks)


def score_has_error_handling(code: str) -> float:
    """Score 0-1: does the code handle errors?"""
    patterns = ["try:", "except", "raise ", "if not ", "ValueError", "TypeError"]
    hits = sum(1 for p in patterns if p in code)
    return min(hits / 2.0, 1.0)


def score_follows_conventions(code: str) -> float:
    """Score 0-1: does the code follow Python conventions?"""
    checks = [
        bool(re.search(r'""".*?"""', code, re.DOTALL) or
             re.search(r"'''.*?'''", code, re.DOTALL)),   # has docstring
        bool(re.search(r'def \w+\(.*:\s*\w+', code)),     # type hints
        not bool(re.search(r'[A-Z]{2,}_[A-Z]', code) and  # no ALL_CAPS vars
                 "def " not in code),
        bool(re.search(r'^    ', code, re.MULTILINE)),     # 4-space indent
    ]
    return sum(checks) / len(checks)


def compute_quality_scores(code: str, prompt: str) -> dict[str, float]:
    """Compute all quality scores for generated code."""
    return {
        "code_completeness": round(score_code_completeness(code, prompt), 2),
        "has_error_handling": round(score_has_error_handling(code), 2),
        "follows_conventions": round(score_follows_conventions(code), 2),
    }
