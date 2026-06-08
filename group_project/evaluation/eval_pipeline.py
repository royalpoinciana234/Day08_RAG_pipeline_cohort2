import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("VECTOR_STORE", "weaviate")
os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    os.getenv("DEEPEVAL_TIMEOUT_SECONDS", "120"),
)

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.evaluate.types import MetricData
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.task10_generation import (
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    _fallback_generate_answer,
    format_context,
    reorder_for_llm,
)
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
TOP_K = 5
EVAL_CASE_LIMIT = int(os.getenv("DEEPEVAL_CASE_LIMIT", "8"))
# Keep context large enough for Faithfulness to verify claims against full chunk content.
# Truncating too aggressively (e.g. 300 chars) causes Faithfulness=0 because the
# evaluation LLM cannot find claims that were generated from the untruncated context.
MAX_CONTEXT_CHARS = int(os.getenv("DEEPEVAL_CONTEXT_CHARS", "400"))
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_EVAL_MODEL", "gpt-4o-mini")
THRESHOLD = 0.7
# async_mode=True hangs on large contexts (>800 chars) due to metric-level event loop issues.
# Default to sync; enable async only when context is small and speed matters.
RUN_ASYNC = os.getenv("DEEPEVAL_RUN_ASYNC", "0") == "1"
MAX_CONCURRENT = int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "4"))
INCLUDE_REASON = os.getenv("DEEPEVAL_INCLUDE_REASON", "0") == "1"
CONFIG_MODE = os.getenv("DEEPEVAL_CONFIG_MODE", "ab")
# Comma-separated subset: faithfulness,answer_relevance,context_recall,context_precision
# Leave empty to run all 4
_METRICS_FILTER = {m.strip() for m in os.getenv("DEEPEVAL_METRICS", "").split(",") if m.strip()}
METRIC_ORDER = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Answer Relevance"),
    ("context_recall", "Context Recall"),
    ("context_precision", "Context Precision"),
]
ACTIVE_METRICS = [
    (key, label) for key, label in METRIC_ORDER
    if not _METRICS_FILTER or key in _METRICS_FILTER
]


def load_golden_dataset() -> list[dict]:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if EVAL_CASE_LIMIT > 0:
        return dataset[:EVAL_CASE_LIMIT]
    return dataset


def _compact_contexts(chunks: list[dict]) -> list[str]:
    contexts = []
    for chunk in chunks[:TOP_K]:
        text = chunk["content"].strip().replace("\n", " ")
        contexts.append(text[:MAX_CONTEXT_CHARS])
    return contexts


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _openai_client


def _generate_answer_with_chunks(query: str, chunks: list[dict]) -> str:
    ordered = reorder_for_llm(chunks)
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_generate_answer(query, ordered)

    try:
        client = _get_openai_client()
        context = format_context(ordered)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return _fallback_generate_answer(query, ordered)


def _run_config(question: str, config_name: str) -> dict:
    if config_name == "hybrid_rerank":
        chunks = retrieve(question, top_k=TOP_K, use_reranking=True)
    elif config_name == "dense_only":
        chunks = semantic_search(question, top_k=TOP_K)
        for chunk in chunks:
            chunk["source"] = "dense"
    else:
        raise ValueError(f"Unknown config: {config_name}")

    answer = _generate_answer_with_chunks(question, chunks)
    return {"answer": answer, "sources": chunks}


def _build_one_case(args: tuple) -> tuple[int, LLMTestCase, dict]:
    index, item, config_name = args
    result = _run_config(item["question"], config_name)
    retrieval_context = _compact_contexts(result["sources"])
    test_case = LLMTestCase(
        name=f"{config_name}-{index:02d}",
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=retrieval_context,
        context=[item["expected_context"]],
    )
    artifact = {
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "expected_context": item["expected_context"],
        "answer": result["answer"],
        "retrieval_context": retrieval_context,
    }
    return index, test_case, artifact


def _build_test_cases(
    golden_dataset: list[dict], config_name: str
) -> tuple[list[LLMTestCase], list[dict]]:
    args_list = [(i, item, config_name) for i, item in enumerate(golden_dataset, 1)]
    results: list[tuple[int, LLMTestCase, dict]] = [None] * len(args_list)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_build_one_case, args): args[0] for args in args_list}
        for future in as_completed(futures):
            index, test_case, artifact = future.result()
            results[index - 1] = (index, test_case, artifact)

    test_cases = [r[1] for r in results]
    artifacts = [r[2] for r in results]
    return test_cases, artifacts


def _create_metrics():
    _metric_builders = {
        # truths_extraction_limit caps facts extracted from context → smaller verdicts prompt → faster
        "faithfulness": lambda: FaithfulnessMetric(threshold=THRESHOLD, model=DEEPEVAL_MODEL, include_reason=INCLUDE_REASON, async_mode=RUN_ASYNC, truths_extraction_limit=3),
        "answer_relevance": lambda: AnswerRelevancyMetric(threshold=THRESHOLD, model=DEEPEVAL_MODEL, include_reason=INCLUDE_REASON, async_mode=RUN_ASYNC),
        "context_recall": lambda: ContextualRecallMetric(threshold=THRESHOLD, model=DEEPEVAL_MODEL, include_reason=INCLUDE_REASON, async_mode=RUN_ASYNC),
        "context_precision": lambda: ContextualPrecisionMetric(threshold=THRESHOLD, model=DEEPEVAL_MODEL, include_reason=INCLUDE_REASON, async_mode=RUN_ASYNC),
    }
    return [_metric_builders[key]() for key, _ in ACTIVE_METRICS]


def _normalize_metric_name(name: str) -> str:
    normalized = name.lower().replace(" ", "_")
    alias_map = {
        "answer_relevancy": "answer_relevance",
        "contextual_recall": "context_recall",
        "contextual_precision": "context_precision",
    }
    return alias_map.get(normalized, normalized)


def _extract_metric_map(metrics_data: list[MetricData] | None) -> dict:
    metric_map = {}
    for metric in metrics_data or []:
        metric_map[_normalize_metric_name(metric.name)] = metric
    return metric_map


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_with_deepeval(golden_dataset: list[dict], config_name: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY để chạy DeepEval built-in metrics.")

    test_cases, artifacts = _build_test_cases(golden_dataset, config_name)
    try:
        eval_result = evaluate(
            test_cases=test_cases,
            metrics=_create_metrics(),
            async_config=AsyncConfig(run_async=RUN_ASYNC, max_concurrent=MAX_CONCURRENT),
            display_config=DisplayConfig(
                print_results=False,
                show_indicator=True,
                inspect_after_run=False,
                truncate_passing_cases=True,
            ),
        )
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            raise RuntimeError(
                "DeepEval timed out while scoring test cases. "
                "Tăng DEEPEVAL_TIMEOUT_SECONDS hoặc giảm DEEPEVAL_CASE_LIMIT / DEEPEVAL_CONTEXT_CHARS."
            ) from exc
        raise RuntimeError(
            f"DeepEval built-in metrics failed while calling evaluation model '{DEEPEVAL_MODEL}'. "
            "Kiểm tra OPENAI_API_KEY và outbound network."
        ) from exc

    per_case = []
    for test_result, artifact in zip(eval_result.test_results, artifacts):
        metric_map = _extract_metric_map(test_result.metrics_data)
        row = {
            "question": artifact["question"],
            "expected_answer": artifact["expected_answer"],
            "expected_context": artifact["expected_context"],
            "answer": artifact["answer"],
            "contexts": artifact["retrieval_context"],
        }
        scores = []
        reasons = {}
        for key, _label in ACTIVE_METRICS:
            metric = metric_map.get(key)
            score = float(metric.score) if metric and metric.score is not None else 0.0
            row[key] = round(score, 4)
            reasons[key] = metric.reason if metric else "No metric output"
            scores.append(score)
        row["average"] = round(_safe_mean(scores), 4)
        row["reasons"] = reasons
        per_case.append(row)

    aggregate = {}
    for key, _label in ACTIVE_METRICS:
        aggregate[key] = round(_safe_mean([case[key] for case in per_case]), 4)
    aggregate["average"] = round(_safe_mean([case["average"] for case in per_case]), 4)

    return {
        "framework": "deepeval",
        "config_name": config_name,
        "num_cases": len(test_cases),
        "aggregate": aggregate,
        "per_case": per_case,
        "evaluation_model": DEEPEVAL_MODEL,
        "notes": "DeepEval built-in metrics chạy thật bằng evaluation model và retrieval bám theo VECTOR_STORE hiện tại của repo.",
    }


def compare_configs(golden_dataset: list[dict]) -> dict:
    if CONFIG_MODE == "hybrid_only":
        return {"hybrid_rerank": evaluate_with_deepeval(golden_dataset, "hybrid_rerank")}
    if CONFIG_MODE == "dense_only":
        return {"dense_only": evaluate_with_deepeval(golden_dataset, "dense_only")}
    # Run both configs concurrently to halve wall-clock time.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(evaluate_with_deepeval, golden_dataset, "hybrid_rerank")
        future_b = executor.submit(evaluate_with_deepeval, golden_dataset, "dense_only")
        results = {}
        for name, future in [("hybrid_rerank", future_a), ("dense_only", future_b)]:
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"⚠ Config '{name}' failed: {exc}")
        if not results:
            raise RuntimeError("Cả 2 configs đều thất bại.")
        return results


def _failure_stage(case: dict) -> str:
    weakest = min(
        [(key, case[key]) for key, _label in ACTIVE_METRICS if key in case],
        key=lambda item: item[1],
    )[0]
    return weakest


def _root_cause(case: dict) -> str:
    return case["reasons"].get(_failure_stage(case), "No metric reason")


def export_results(comparison: dict) -> None:
    config_a = comparison.get("hybrid_rerank")
    config_b = comparison.get("dense_only")
    primary = config_a or config_b
    if primary is None:
        raise RuntimeError("Không có kết quả evaluation để export.")
    bottom = sorted(primary["per_case"], key=lambda case: case["average"])[:3]

    rows = []
    for key, label in [*ACTIVE_METRICS, ("average", "**Average**")]:
        left = config_a["aggregate"][key] if config_a else 0.0
        right = config_b["aggregate"][key] if config_b else 0.0
        delta = left - right if config_a and config_b else 0.0
        right_cell = f"{right:.4f}" if config_b else "-"
        delta_cell = f"{delta:+.4f}" if config_a and config_b else "-"
        rows.append(f"| {label} | {left:.4f} | {right_cell} | {delta_cell} |")

    worst_rows = []
    for idx, case in enumerate(bottom, 1):
        score_cells = " | ".join(
            f"{case.get(key, 0.0):.4f}" for key, _ in ACTIVE_METRICS
        )
        worst_rows.append(
            f"| {idx} | {case['question']} | {score_cells} | "
            f"{_failure_stage(case)} | {_root_cause(case).replace(chr(10), ' ') if INCLUDE_REASON else 'Reason disabled'} |"
        )

    content = "\n".join(
        [
            "# RAG Evaluation Results",
            "",
            "## Framework sử dụng",
            "",
            "- Framework: `DeepEval`",
            f"- Dataset size: `{primary['num_cases']}`",
            f"- Evaluation model: `{primary['evaluation_model']}`",
            f"- Ghi chú: {primary['notes']}",
            f"- Run async: `{RUN_ASYNC}` | Max concurrent: `{MAX_CONCURRENT}` | Include reason: `{INCLUDE_REASON}` | Config mode: `{CONFIG_MODE}`",
            "",
            "## Overall Scores",
            "",
            "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |",
            "|--------|-----------------------------|-----------------------|---|",
            *rows,
            "",
            "## A/B Comparison Analysis",
            "",
            "**Config A:** hybrid retrieval (`semantic + BM25`) + RRF + reranking.",
            "",
            "**Config B:** dense-only retrieval (`semantic_search`) không hybrid và không rerank.",
            "",
            (
                f"**Kết luận:** Config A có điểm trung bình {config_a['aggregate']['average']:.4f}, "
                f"chênh so với Config B {config_a['aggregate']['average'] - config_b['aggregate']['average']:+.4f}."
                if config_a and config_b
                else "**Kết luận:** Đang chạy một config để debug tốc độ hoặc chất lượng eval."
            ),
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | " + " | ".join(label for _, label in ACTIVE_METRICS) + " | Failure Stage | Root Cause |",
            "|---|----------|" + "---------|" * len(ACTIVE_METRICS) + "---------------|------------|",
            *worst_rows,
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** Làm sạch markdown news trước khi chunk để giảm menu/header noise.",
            "**Expected impact:** Tăng Context Precision cho câu hỏi về tin tức.",
            "",
            "### Cải tiến 2",
            "**Action:** Thay fallback embedding hiện tại bằng multilingual embedding model thật.",
            "**Expected impact:** Tăng Context Recall cho các câu paraphrase hoặc dài.",
            "",
            "### Cải tiến 3",
            "**Action:** Giữ một đường generation thống nhất cho mọi config và ép citation chặt hơn trong prompt.",
            "**Expected impact:** Tăng Faithfulness và Answer Relevance.",
        ]
    )
    RESULTS_PATH.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    golden_dataset = load_golden_dataset()
    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print(f"Loaded {len(golden_dataset)} golden cases")
    for name, result in comparison.items():
        print(f"{name}: avg={result['aggregate']['average']:.4f}")
    print(f"Wrote report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()