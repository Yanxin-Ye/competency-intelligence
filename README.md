# Segment Contribution Analysis on Growth
## Overview

When comparing a target entity against a peer set, aggregate metrics
(e.g., YoY growth) often mask underlying drivers. To understand
performance gaps, we decompose the difference across key dimensions.

-   **Target YoY growth:** 29.03%
-   **Peer set YoY growth:** 62.86%
-   **Gap:** **-33.82%**

## Key Insight

The segment **(Dimension1 = B, Dimension2 = C)** is the primary driver
of underperformance:

-   Contribution: **-54.72%**
-   Share of total gap: **162%**

This segment alone more than explains the total gap, while other
segments partially offset it.

## Recommendation

-   Prioritize investigation into **(B, C)**
-   Analyze drivers such as customer mix, pricing, and product
    performance

## Detailed Breakdown

| Dimension1 | Dimension2 | amt_ty | amt_ly | amt_growth | amt_growth_ctc | peer_amt_ty | peer_amt_ly | peer_amt_growth | peer_amt_growth_ctc | growth_ctc_diff | contribution% |
|------------|------------|--------|--------|------------|----------------|-------------|-------------|-----------------|----------------------|-----------------|----------------|
| A | C | 9 | 12 | -25.00% | -9.68% | 28 | 34 | -17.65% | -5.71% | -3.96% | 12% |
| A | D | 17 | 12 | 41.67% | 16.13% | 9 | 12 | -25.00% | -2.86% | 18.99% | -56% |
| B | C | 9 | 5 | 80.00% | 12.90% | 127 | 56 | 126.79% | 67.62% | -54.72% | 162% |
| B | D | 5 | 2 | 150.00% | 9.68% | 7 | 3 | 133.33% | 3.81% | 5.87% | -17% |
|  | **Total** | 40 | 31 | 29.03% | 29.03% | 171 | 105 | 62.86% | 62.86% | -33.82% | 100% |
