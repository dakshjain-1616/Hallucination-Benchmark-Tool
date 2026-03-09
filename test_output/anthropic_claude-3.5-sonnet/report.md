# Hallucination Benchmark Report

**Report ID:** report_20260309_101401  
**Generated:** 2026-03-09T10:14:01.473429  
**Model:** unknown  
**Dataset:** sample_benchmark

## Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3 |
| Avg Faithfulness | 82.14% |
| Avg Factual Consistency | 66.67% |
| Avg Hallucination Rate | 33.33% |
| Avg BLEU Score | 0.4852 |
| Hallucinated Samples | 0 (0.0%) |
| Avg Claims per Response | 1.00 |
| Avg Citations per Response | 0.00 |

## Category Distribution


## Sample Results

### Sample sample_1 ✅ CORRECT

**Question:** What is the capital of France?

**Metrics:**
- Faithfulness: 100.00%
- Factual Consistency: 100.00%
- Hallucination Rate: 0.00%
- BLEU Score: 0.5623

**Response:**
```
Paris is the capital of France.
```

**Ground Truth:**
```
The capital of France is Paris.
```

### Sample sample_2 ✅ CORRECT

**Question:** Who wrote 'Romeo and Juliet'?

**Metrics:**
- Faithfulness: 75.00%
- Factual Consistency: 100.00%
- Hallucination Rate: 0.00%
- BLEU Score: 0.6804

**Response:**
```
William Shakespeare wrote Romeo and Juliet in 1597.
```

**Ground Truth:**
```
William Shakespeare wrote 'Romeo and Juliet'.
```

### Sample sample_3 ✅ CORRECT

**Question:** What is the speed of light?

**Metrics:**
- Faithfulness: 71.43%
- Factual Consistency: 0.00%
- Hallucination Rate: 100.00%
- BLEU Score: 0.2128

**Response:**
```
The speed of light is 300,000 km/s.
```

**Ground Truth:**
```
The speed of light in vacuum is approximately 299,792,458 meters per second.
```

