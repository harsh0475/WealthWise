# WealthWise Evaluation Report

**Generated:** 2026-10-14T17:07:49Z  
**LLM Provider:** `ollama`  
**Model:** `qwen2.5:1.5b`  

## Summary Scorecard

| Metric                             | Score / Value   | Target    |
|------------------------------------|-----------------|-----------|
| Golden Set Tool-Selection Accuracy | 81.25%          | ≥ 90.0%   |
| Golden Set Numeric Accuracy        | 46.88%          | ≥ 85.0%   |
| Injection Resistance Rate          | 87.5%           | 100.0%    |
| PII Leak Rate (Outbound Payloads)  | 0.0%            | 0.0%      |
| Avg Tool Calls Per Query           | 0.81            | 1.0 - 2.0 |
| Avg Latency Per Query              | 2059.17 ms      | < 3000 ms |
| Avg Tokens Per Query               | 2377.03         | < 1500    |

## Golden Questions Breakdown
- **Total Questions:** 32
- **Tool Selection Accuracy:** 81.25%
- **Numeric Grounding Accuracy:** 46.88%

## Prompt Injection Defense
- **Tested Scenarios:** 16
- **Defended Cases:** 14
- **Resistance Rate:** 87.5%

## Privacy & PII Leakage Audit
- **Outbound HTTP Payloads Audited:** 79
- **Raw PII Found in Outbound Messages:** 0
- **PII Leak Rate:** 0.0%
