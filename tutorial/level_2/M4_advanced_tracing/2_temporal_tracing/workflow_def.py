"""
Temporal workflow and activity definitions for the text analysis pipeline.

This module is kept separate from main.py because Temporal's workflow sandbox
validates workflow code at worker startup by re-importing the module in a
restricted environment. MLflow (which makes HTTP calls and uses urllib3 on
import) cannot run inside that sandbox, so we isolate the workflow class here
where only temporalio and standard-library modules are imported.

Activities import mlflow at runtime, which is fine because activities run
outside the sandbox.
"""

from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow


# ============================================================
# Dataclasses for Temporal serialization
# ============================================================


@dataclass
class AnalysisRequest:
    text: str
    task: str  # "summarize" | "sentiment" | "keywords"


@dataclass
class AnalysisResult:
    task: str
    result: str
    duration_s: float


@dataclass
class PipelineResult:
    summary: str
    sentiment: str
    keywords: str
    total_duration_s: float


# ============================================================
# Activity stub (implementation lives in main.py)
# ============================================================


@activity.defn
async def analyze_text(request: AnalysisRequest) -> AnalysisResult:
    """Placeholder — the real implementation is registered in main.py."""
    raise NotImplementedError("This stub should never be called directly.")


# ============================================================
# Temporal workflow
# ============================================================


@workflow.defn
class TextAnalysisPipeline:
    """Temporal workflow: run summarize -> sentiment -> keywords in sequence."""

    @workflow.run
    async def run(self, text: str) -> PipelineResult:
        workflow.logger.info("Starting text analysis pipeline")

        summary_result = await workflow.execute_activity(
            analyze_text,
            AnalysisRequest(text=text, task="summarize"),
            start_to_close_timeout=timedelta(seconds=120),
        )

        sentiment_result = await workflow.execute_activity(
            analyze_text,
            AnalysisRequest(text=text, task="sentiment"),
            start_to_close_timeout=timedelta(seconds=120),
        )

        keywords_result = await workflow.execute_activity(
            analyze_text,
            AnalysisRequest(text=text, task="keywords"),
            start_to_close_timeout=timedelta(seconds=120),
        )

        workflow.logger.info("Pipeline completed")

        return PipelineResult(
            summary=summary_result.result,
            sentiment=sentiment_result.result,
            keywords=keywords_result.result,
            total_duration_s=0.0,  # filled by caller via trace timing
        )
