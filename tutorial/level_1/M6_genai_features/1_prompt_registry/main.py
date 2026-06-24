"""L1-M6.1 — Prompt Registry: versioned prompt management with MLflow."""

import mlflow
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ── MLflow setup ──────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("L1/M6_genai_features/1_prompt_registry")


def main() -> None:
    prompt_name = "L1_M6_explainer_prompt"

    # ── Step 1: Register a prompt (version 1) ────────────────
    print("=" * 60)
    print("Step 1: Register a prompt template (version 1)")
    print("=" * 60)

    v1 = mlflow.genai.register_prompt(
        name=prompt_name,
        template="Explain {{topic}} to a {{audience}} in 2-3 sentences.",
        commit_message="Initial explainer prompt",
        tags={"level": "1", "style": "concise"},
    )
    print(f"  Registered: {v1.name} v{v1.version}")
    print(f"  Template:   {v1.template}")
    print(f"  Variables:  {v1.variables}")
    print(f"  URI:        {v1.uri}")

    # ── Step 2: Create version 2 (improved wording) ──────────
    print()
    print("=" * 60)
    print("Step 2: Register an improved version (version 2)")
    print("=" * 60)

    v2 = mlflow.genai.register_prompt(
        name=prompt_name,
        template=(
            "You are a friendly teacher. "
            "Explain {{topic}} in a way that a {{audience}} would understand. "
            "Use a simple analogy and keep it to 2-3 sentences."
        ),
        commit_message="Add teacher persona and analogy instruction",
        tags={"level": "1", "style": "pedagogical"},
    )
    print(f"  Registered: {v2.name} v{v2.version}")
    print(f"  Template:   {v2.template}")

    # ── Step 3: Set an alias ─────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3: Set a 'production' alias on version 2")
    print("=" * 60)

    mlflow.genai.set_prompt_alias(prompt_name, alias="production", version=v2.version)
    print(f"  Alias 'production' -> v{v2.version}")

    # ── Step 4: Load prompts by version and alias ────────────
    print()
    print("=" * 60)
    print("Step 4: Load prompts by version number and by alias")
    print("=" * 60)

    loaded_v1 = mlflow.genai.load_prompt(prompt_name, version=1)
    print(f"  Loaded v1: {loaded_v1.template[:60]}...")

    loaded_prod = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production")
    print(f"  Loaded @production (v{loaded_prod.version}): {loaded_prod.template[:60]}...")

    # ── Step 5: Format the template ──────────────────────────
    print()
    print("=" * 60)
    print("Step 5: Format the production prompt with variables")
    print("=" * 60)

    formatted = loaded_prod.format(topic="recursion", audience="10-year-old")
    print(f"  Formatted: {formatted}")

    # ── Step 6: Search registered prompts ────────────────────
    print()
    print("=" * 60)
    print("Step 6: Search all registered prompts")
    print("=" * 60)

    prompts = mlflow.genai.search_prompts()
    for p in prompts:
        print(f"  - {p.name} (created: {p.creation_timestamp})")
    print(f"  Total prompts found: {len(prompts)}")

    # ── Step 7: Use the prompt with an LLM ───────────────────
    print()
    print("=" * 60)
    print("Step 7: Use the production prompt with ChatOllama")
    print("=" * 60)

    # Convert {{var}} to {var} for LangChain compatibility
    lc_template = loaded_prod.to_single_brace_format()
    lc_prompt = ChatPromptTemplate.from_template(lc_template)

    llm = ChatOllama(model="gemma4:e2b", temperature=0.7)
    chain = lc_prompt | llm

    with mlflow.start_run(run_name="prompt_registry_demo"):
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_version", loaded_prod.version)

        response = chain.invoke({"topic": "recursion", "audience": "10-year-old"})
        answer = response.content
        print(f"  LLM response:\n  {answer}")

        mlflow.log_param("llm_response", answer[:250])

    # ── Cleanup alias ────────────────────────────────────────
    mlflow.genai.delete_prompt_alias(prompt_name, alias="production")

    print()
    print("=" * 60)
    print("Done! Check the MLflow UI at http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
