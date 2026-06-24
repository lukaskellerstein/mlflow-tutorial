"""
CodeGenAgent — a simulated Codex-style code generation agent.

Mirrors the Codex SDK workflow: plan -> generate -> review -> refine.
Each stage is instrumented with MLflow tracing.
"""

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class CodeGenAgent:
    """A code generation agent that mirrors the Codex SDK workflow."""

    def __init__(self, model: str = "gemma4:e2b", temperature: float = 0.3):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.model = model
        self.temperature = temperature

    @mlflow.trace(name="codegen_plan")
    def plan(self, prompt: str) -> str:
        """Break a coding task into an implementation plan."""
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a software architect. Given a coding task, "
             "produce a short numbered plan (3-5 steps) for implementing it. "
             "Be concise — one line per step."),
            ("user", "{prompt}"),
        ])
        chain = template | self.llm
        return chain.invoke({"prompt": prompt}).content

    @mlflow.trace(name="codegen_generate")
    def generate_code(self, prompt: str, plan: str | None = None) -> str:
        """Generate Python code from a natural-language prompt."""
        system_msg = (
            "You are an expert Python developer. Write clean, well-documented "
            "Python code that solves the task. Return ONLY the code — no "
            "markdown fences, no explanation."
        )
        if plan:
            system_msg += f"\n\nFollow this plan:\n{plan}"

        template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{prompt}"),
        ])
        chain = template | self.llm
        return chain.invoke({"prompt": prompt}).content

    @mlflow.trace(name="codegen_review")
    def review_code(self, code: str, original_prompt: str) -> str:
        """Review generated code and provide improvement suggestions."""
        template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior code reviewer. Review the Python code below "
             "against the original task. List up to 3 concrete improvements. "
             "Be concise — one line per suggestion. If the code is good, "
             "say 'LGTM'."),
            ("user", "Task: {prompt}\n\nCode:\n{code}"),
        ])
        chain = template | self.llm
        return chain.invoke({"prompt": original_prompt, "code": code}).content

    @mlflow.trace(name="codegen_refine")
    def refine(self, code: str, feedback: str, original_prompt: str) -> str:
        """Refine code based on review feedback."""
        if "LGTM" in feedback.upper():
            return code
        template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert Python developer. Improve the code based on "
             "the reviewer feedback. Return ONLY the improved code — no "
             "markdown fences, no explanation."),
            ("user", "Task: {prompt}\nFeedback: {feedback}\n\nCode:\n{code}"),
        ])
        chain = template | self.llm
        return chain.invoke({
            "prompt": original_prompt, "feedback": feedback, "code": code,
        }).content

    @mlflow.trace(name="codegen_pipeline")
    def run_pipeline(self, prompt: str, *, use_plan: bool = False) -> dict:
        """Run the full code generation pipeline with MLflow tracing."""
        with mlflow.start_span(name="pipeline_orchestration") as span:
            span.set_inputs({"prompt": prompt, "use_plan": use_plan})

            plan_text = self.plan(prompt) if use_plan else None
            code = self.generate_code(prompt, plan=plan_text)
            review = self.review_code(code, prompt)
            final_code = self.refine(code, review, prompt)

            result = {
                "prompt": prompt,
                "plan": plan_text,
                "initial_code": code,
                "review_feedback": review,
                "final_code": final_code,
                "used_plan": use_plan,
            }
            span.set_outputs({"final_code_length": len(final_code)})
            return result
