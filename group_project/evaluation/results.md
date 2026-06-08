# RAG Evaluation Results

## Framework sử dụng

- Framework: `DeepEval`
- Dataset size: `16`
- Evaluation model: `gpt-4o-mini`
- Ghi chú: DeepEval built-in metrics chạy thật bằng evaluation model và retrieval bám theo VECTOR_STORE hiện tại của repo.
- Run async: `False` | Max concurrent: `4` | Include reason: `False` | Config mode: `ab`

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|-----------------------------|-----------------------|---|
| Faithfulness | 0.0000 | 1.0000 | - |
| Answer Relevance | 0.0000 | 0.1250 | - |
| Context Recall | 0.0000 | 0.1042 | - |
| Context Precision | 0.0000 | 0.0000 | - |
| **Average** | 0.0000 | 0.3073 | - |

## A/B Comparison Analysis

**Config A:** hybrid retrieval (`semantic + BM25`) + RRF + reranking.

**Config B:** dense-only retrieval (`semantic_search`) không hybrid và không rerank.

**Kết luận:** Đang chạy một config để debug tốc độ hoặc chất lượng eval.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Answer Relevance | Context Recall | Context Precision | Failure Stage | Root Cause |
|---|----------|---------|---------|---------|---------|---------------|------------|
| 1 | Điều 249 Bộ luật Hình sự quy định mức phạt cơ bản nào cho tội tàng trữ trái phép chất ma túy? | 1.0000 | 0.0000 | 0.0000 | 0.0000 | answer_relevance | Reason disabled |
| 2 | Khoản 1 Điều 249 nêu ngưỡng heroin, cocaine, methamphetamine, amphetamine, ketamine, fentanyl, MDMA hoặc XLR-11 là bao nhiêu? | 1.0000 | 0.0000 | 0.0000 | 0.0000 | answer_relevance | Reason disabled |
| 3 | Điều 255 Bộ luật Hình sự quy định mức phạt cơ bản nào cho tội tổ chức sử dụng trái phép chất ma túy? | 1.0000 | 0.0000 | 0.0000 | 0.0000 | answer_relevance | Reason disabled |

## Recommendations

### Cải tiến 1
**Action:** Làm sạch markdown news trước khi chunk để giảm menu/header noise.
**Expected impact:** Tăng Context Precision cho câu hỏi về tin tức.

### Cải tiến 2
**Action:** Thay fallback embedding hiện tại bằng multilingual embedding model thật.
**Expected impact:** Tăng Context Recall cho các câu paraphrase hoặc dài.

### Cải tiến 3
**Action:** Giữ một đường generation thống nhất cho mọi config và ép citation chặt hơn trong prompt.
**Expected impact:** Tăng Faithfulness và Answer Relevance.