#!/usr/bin/env python3
"""
Profile Gemini Code Assist API requests to compare model latencies.

Usage:
    uv run scripts/profile_request.py                    # Test all models
    uv run scripts/profile_request.py gemini-2.5-flash   # Test specific model
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def time_prompt(model: Any, prompt_text: str, runs: int = 3, delay: float = 0.5) -> list[float]:
    """Time multiple runs of a prompt and return list of times in ms."""
    times = []
    for i in range(runs):
        start = time.perf_counter()
        response = model.prompt(prompt_text)
        _ = str(response)  # Force execution
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"    Run {i + 1}: {elapsed:.0f}ms")
        if i < runs - 1:
            time.sleep(delay)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Gemini Code Assist model latencies")
    parser.add_argument("model", nargs="?", help="Specific model to test (e.g., gemini-2.5-flash)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model (default: 3)")
    parser.add_argument(
        "--delay", type=float, default=10.0, help="Delay between models in seconds (default: 10)"
    )
    parser.add_argument(
        "--run-delay", type=float, default=0.5, help="Delay between runs in seconds (default: 0.5)"
    )
    args = parser.parse_args()

    import llm

    from llm_gemini_code_assist import GEMINI_CODE_ASSIST_MODELS

    print("=== Gemini Code Assist Model Latency Comparison ===\n")

    prompt_text = "Say 'hello' and nothing else"

    # Determine which models to test
    if args.model:
        models_to_test = [args.model]
    else:
        models_to_test = sorted(GEMINI_CODE_ASSIST_MODELS)

    print(f'Prompt: "{prompt_text}"')
    print(f"Runs per model: {args.runs}")
    print(f"Delay between models: {args.delay}s\n")

    results = {}

    for i, model_id in enumerate(models_to_test):
        full_model_id = f"gemini-ca/{model_id}"
        print(f"Testing {full_model_id}...")

        try:
            model = llm.get_model(full_model_id)
            times = time_prompt(model, prompt_text, runs=args.runs, delay=args.run_delay)

            avg = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            min_t = min(times)
            max_t = max(times)

            results[model_id] = {
                "times": times,
                "avg": avg,
                "std": std,
                "min": min_t,
                "max": max_t,
            }

            print(f"  => avg={avg:.0f}ms (min={min_t:.0f}, max={max_t:.0f}, std={std:.0f})\n")

        except Exception as e:
            print(f"  => ERROR: {e}\n")
            results[model_id] = {"error": str(e)}

        # Wait between models (except after last one)
        if i < len(models_to_test) - 1:
            print(f"Waiting {args.delay}s before next model...")
            time.sleep(args.delay)

    # Summary table
    if len(results) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'Avg (ms)':<12} {'Min':<10} {'Max':<10}")
        print("-" * 60)

        for model_id in sorted(
            results.keys(),
            key=lambda x: float(results[x].get("avg", float("inf"))),  # type: ignore[arg-type]
        ):
            r = results[model_id]
            if "error" in r:
                print(f"{model_id:<30} {'ERROR':<12}")
            else:
                print(f"{model_id:<30} {r['avg']:<12.0f} {r['min']:<10.0f} {r['max']:<10.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
