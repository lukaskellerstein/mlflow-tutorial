"""
L1-M1.2 — Tracking LLM Experiments

A comprehensive showcase of every MLflow tracking/logging method:
  log_param, log_params, log_metric, log_metrics, set_tag, set_tags,
  log_text, log_dict, log_table, log_artifact, log_artifacts,
  log_figure, log_image, and step-based metric logging.
"""

import os
import tempfile
import time

import matplotlib
import mlflow
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "L1/M1_tracking/2_tracking_basics"

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-e4b"


def call_llm(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict:
    """Call the LLM and return the response with timing and usage info."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    return {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "response_time_seconds": round(elapsed, 3),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = OpenAI(base_url=LMSTUDIO_URL, api_key="lm-studio")

    prompt = "Explain what MLflow is in 2 sentences."

    with mlflow.start_run(run_name="all_logging_methods"):

        # ------------------------------------------------------------------
        # Step 1: log_param / log_params / log_metric / log_metrics
        # ------------------------------------------------------------------
        section("Step 1: Parameters and Metrics")

        mlflow.log_param("model", MODEL)
        mlflow.log_param("prompt", prompt)
        print("  log_param() — logged 'model' and 'prompt' individually")

        mlflow.log_params({"temperature": 0.7, "max_tokens": 1024})
        print("  log_params() — logged 'temperature' and 'max_tokens' as a batch")

        result = call_llm(client, prompt)

        mlflow.log_metric("response_time_seconds", result["response_time_seconds"])
        print(f"  log_metric() — logged response_time={result['response_time_seconds']}s")

        mlflow.log_metrics({
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        })
        print(f"  log_metrics() — logged token counts (total={result['total_tokens']})")

        # ------------------------------------------------------------------
        # Step 2: set_tag / set_tags
        # ------------------------------------------------------------------
        section("Step 2: Tags")

        mlflow.set_tag("model_family", "gemma")
        print("  set_tag() — tagged 'model_family'='gemma'")

        mlflow.set_tags({
            "level": "1",
            "module": "tracking",
            "lesson": "tracking_basics",
        })
        print("  set_tags() — tagged level, module, lesson as a batch")

        # ------------------------------------------------------------------
        # Step 3: log_text / log_dict
        # ------------------------------------------------------------------
        section("Step 3: Text and Dict Artifacts")

        mlflow.log_text(result["content"], "response.txt")
        print("  log_text() — saved LLM response as 'response.txt'")

        mlflow.log_dict(
            {
                "prompt": prompt,
                "model": MODEL,
                "temperature": 0.7,
                "response": result["content"],
                "tokens": result["total_tokens"],
            },
            "call_details.json",
        )
        print("  log_dict() — saved structured call details as 'call_details.json'")

        # ------------------------------------------------------------------
        # Step 4: log_table
        # ------------------------------------------------------------------
        section("Step 4: Table (temperature comparison)")

        temperatures = [0.3, 0.7, 1.0]
        temp_results = []

        for temp in temperatures:
            r = call_llm(client, prompt, temperature=temp)
            temp_results.append({
                "temperature": temp,
                "prompt": prompt,
                "response": r["content"],
                "total_tokens": r["total_tokens"],
                "response_time": r["response_time_seconds"],
            })
            print(f"  temp={temp}  tokens={r['total_tokens']:>4d}  time={r['response_time_seconds']}s")

        mlflow.log_table(
            data=pd.DataFrame(temp_results),
            artifact_file="temperature_comparison.json",
        )
        print("  log_table() — saved temperature comparison table")

        # ------------------------------------------------------------------
        # Step 5: log_artifact / log_artifacts
        # ------------------------------------------------------------------
        section("Step 5: File Artifacts")

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.md")
            with open(summary_path, "w") as f:
                f.write(f"# LLM Call Summary\n\n")
                f.write(f"**Model:** {MODEL}\n")
                f.write(f"**Prompt:** {prompt}\n\n")
                f.write(f"## Response\n{result['content']}\n")
            mlflow.log_artifact(summary_path)
            print("  log_artifact() — saved 'summary.md' (single file)")

        with tempfile.TemporaryDirectory() as tmpdir:
            responses_dir = os.path.join(tmpdir, "responses")
            os.makedirs(responses_dir)
            for row in temp_results:
                filepath = os.path.join(responses_dir, f"temp_{row['temperature']}.txt")
                with open(filepath, "w") as f:
                    f.write(row["response"])
            mlflow.log_artifacts(responses_dir, artifact_path="responses")
            print("  log_artifacts() — saved 'responses/' directory (3 files)")

        # ------------------------------------------------------------------
        # Step 6: log_figure / log_image
        # ------------------------------------------------------------------
        section("Step 6: Figure and Image")

        fig, ax = plt.subplots(figsize=(6, 4))
        temps = [r["temperature"] for r in temp_results]
        tokens = [r["total_tokens"] for r in temp_results]
        ax.bar([str(t) for t in temps], tokens, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Total Tokens")
        ax.set_title("Token Usage by Temperature")
        fig.tight_layout()
        mlflow.log_figure(fig, "token_chart.png")
        plt.close(fig)
        print("  log_figure() — saved matplotlib bar chart as 'token_chart.png'")

        img = Image.new("RGB", (400, 200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except OSError:
            font = ImageFont.load_default()
            font_small = font
        draw.text((20, 20), f"Model: {MODEL}", fill="black", font=font)
        draw.text((20, 50), f"Tokens: {result['total_tokens']}", fill="black", font=font)
        draw.text((20, 80), f"Time: {result['response_time_seconds']}s", fill="black", font=font)
        draw.text((20, 120), result["content"][:80] + "...", fill="gray", font=font_small)
        mlflow.log_image(img, artifact_file="summary_card.png")
        print("  log_image() — saved PIL-generated summary card as 'summary_card.png'")

        # ------------------------------------------------------------------
        # Step 7: Step-based metrics
        # ------------------------------------------------------------------
        section("Step 7: Step-based Metrics")

        prompts = [
            "What is a transformer model?",
            "What is attention in machine learning?",
            "What is backpropagation?",
            "What is gradient descent?",
            "What is overfitting?",
        ]

        cumulative_tokens = 0
        for step, p in enumerate(prompts):
            r = call_llm(client, p, max_tokens=1024)
            cumulative_tokens += r["total_tokens"]

            mlflow.log_metric("step_response_time", r["response_time_seconds"], step=step)
            mlflow.log_metric("step_tokens", r["total_tokens"], step=step)
            mlflow.log_metric("cumulative_tokens", cumulative_tokens, step=step)

            print(f"  Step {step}: '{p[:35]}...'  tokens={r['total_tokens']}  cumulative={cumulative_tokens}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    section("Done!")
    print(f"  Open the MLflow UI at {MLFLOW_TRACKING_URI}")
    print(f"  Navigate to experiment: {EXPERIMENT_NAME}")
    print("  Check the run 'all_logging_methods' — it has:")
    print("    - Parameters tab: model, prompt, temperature, max_tokens")
    print("    - Metrics tab: response_time, tokens, step-based charts")
    print("    - Tags: model_family, level, module, lesson")
    print("    - Artifacts: response.txt, call_details.json, temperature_comparison.json,")
    print("                 summary.md, responses/, token_chart.png, summary_card.png")


if __name__ == "__main__":
    main()
