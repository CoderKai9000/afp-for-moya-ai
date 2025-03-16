# Performance Comparison Report

## Overview

This report compares the performance of three implementations:
- **Baseline**: Direct function calls without AFP
- **AFP**: Original Agent Flow Protocol implementation
- **Optimized AFP**: Improved AFP implementation with direct routing and prioritization

## Latency Comparison (ms)

| Test | Baseline | AFP | Optimized AFP | Improvement (Baseline->Opt) |
| --- | ---: | ---: | ---: | ---: |
| Direct Communication | 3.17 | 1.75 | 1.65 | 47.93% |
| Multi Agent Orchestration | 4.84 | 0.06 | 1.57 | 67.46% |
| Complex Workflow | 20.79 | 22.13 | 12.73 | 38.76% |

## Throughput Comparison (messages/second)

| Test | Baseline | AFP | Optimized AFP | Improvement (Baseline->Opt) |
| --- | ---: | ---: | ---: | ---: |
| Direct Communication | 315.58 | 569.92 | 606.13 | 92.07% |
| Multi Agent Orchestration | 206.76 | 16642.74 | 635.42 | 207.32% |
| Complex Workflow | 190.72 | 225.90 | 277.94 | 45.73% |

## Workflow Completion Time (ms)

| Implementation | Average Time | Improvement vs. Baseline |
| --- | ---: | ---: |
| Baseline | 20.79 | - |
| AFP | 22.13 | -6.45% |
| Optimized AFP | 12.73 | 38.76% |

## Key Findings

1. **Overall Latency Improvement**: The optimized AFP implementation shows an average latency reduction of 51.39% compared to the baseline.
2. **Overall Throughput Improvement**: The optimized AFP implementation shows an average throughput increase of 115.04% compared to the baseline.
3. **AFP Optimization Impact**: Compared to the original AFP implementation, the optimized version shows a -823.58% reduction in latency and a -22.26% increase in throughput.
4. **Most Significant Latency Improvement**: Multi Agent Orchestration shows the most significant latency reduction at 67.46%.
5. **Most Significant Throughput Improvement**: Multi Agent Orchestration shows the most significant throughput increase at 207.32%.