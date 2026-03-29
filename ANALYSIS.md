# Regional Sector Analysis

Interpretive notes for the canonical sector panel in `data/sector_sentiment.csv`.

This document is a qualitative deep dive built on 9,422 summary-derived sector observations from 1,440 district summaries across 120 Beige Book dates (`2011-01-01` through `2025-11-01`). For the exact reproducible coefficients, p-values, sample windows, and FDR counts used elsewhere in the repo, use `output/analysis_results.json` and `output/analysis_results.md`.

## Methodology

Each district summary is split into sentences, and each sentence is classified into one of 12 economic sectors using keyword matching. VADER sentiment is then scored per sector per district per date, producing a long-format dataset: one row per (date, district, sector).

| Sector | Observations | Avg Sentiment |
|--------|-------------|---------------|
| General | 1,181 | +0.391 |
| Consumer Spending | 1,157 | +0.493 |
| Manufacturing | 1,101 | +0.490 |
| Employment | 1,022 | +0.420 |
| Prices | 987 | +0.188 |
| Real Estate | 838 | +0.518 |
| Financial Services | 759 | +0.390 |
| Agriculture | 663 | +0.188 |
| Energy | 510 | +0.411 |
| Transportation | 489 | +0.118 |
| Tourism & Hospitality | 488 | +0.380 |
| Construction | 227 | +0.165 |

## Regional Profiles

### The Northeast Corridor: Boston, New York, Philadelphia

These three districts are consistently the **most positive on nearly every sector map**. They lead in Manufacturing, Employment, Financial Services, and Real Estate sentiment.

This doesn't mean their economies are objectively better -- it reflects that Fed contacts in these districts (bankers, manufacturers, retailers) tend to describe conditions in more optimistic language. The Northeast also has the **highest sentiment volatility**: when things go bad, these districts swing hard negative, but they bounce back fastest. New York Employment sentiment went from -0.94 during COVID to +0.91 the next year -- the single largest swing in the dataset.

**Strongest sectors:** Real Estate, Manufacturing, Financial Services

**District highlights:**
- **Boston** leads the nation in Manufacturing (+0.66), Real Estate (+0.69), and Financial Services (+0.61)
- **New York** has the strongest Employment sentiment (+0.61) and second-highest Real Estate (+0.65)
- **Philadelphia** leads in Prices sentiment (+0.28) and ranks second in Manufacturing (+0.62) -- these contacts describe price changes with less alarm than other districts

### Cleveland & Richmond: The Steady Middle

These districts track the Northeast closely but with slightly less amplitude. Both have **diversified sector coverage** (low HHI concentration scores), meaning the Fed hears from a broad mix of industries rather than being dominated by any one sector.

**Cleveland** is the single most important district for prediction. Its **Employment sector has the strongest correlation with real economic activity** (r = 0.61) of any sector-district pair in the entire dataset. This likely reflects Ohio's manufacturing-heavy economy -- when the auto plants and steel mills report changes in hiring, it translates directly to the state coincident index.

**Richmond** covers VA, MD, DC, NC, SC -- a mix of government, finance, and manufacturing. Its Real Estate sentiment (+0.66) is remarkably stable and positive.

### Atlanta: Diversified South

Atlanta covers FL, GA, AL, TN, LA, MS -- a huge, diverse geography. Its sector sentiment is moderate across the board with no extreme highs or lows.

Its **most notable feature is Agriculture recovery**: Atlanta had the fastest agricultural bounce-back after COVID. Transportation is its weakest sector -- likely reflecting port and logistics stress in a region heavily dependent on freight. Atlanta Real Estate has the strongest *negative* correlation with economic activity (r = -0.37) of any sector-district pair, suggesting that when Atlanta Fed contacts are most optimistic about real estate, it may signal overheating.

**Strongest sectors:** Real Estate (+0.52), Consumer Spending (+0.48), Manufacturing (+0.46)

### Chicago: Manufacturing Heartland

Consumer Spending and Manufacturing dominate Chicago's narrative. Construction is notably weak (+0.04), reflecting the Midwest's slower building activity compared to Sun Belt and coastal regions.

The Manufacturing animated map shows Chicago tracking very closely with the national pattern -- it's a **bellwether, not a leader**. When manufacturing sentiment moves nationally, Chicago moves with it, not ahead of it.

**Strongest sectors:** Consumer Spending (+0.58), Manufacturing (+0.55), Real Estate (+0.53)

### St. Louis & Minneapolis: The Quiet Interior

Both districts are mid-pack on almost everything.

**Minneapolis** stands out for its **weak Agriculture sentiment** (+0.13, below national average) despite being in farm country. This reflects that the Fed's Beige Book tends to emphasize agricultural *challenges* (drought, low commodity prices, trade uncertainty) over successes. Minnesota, Montana, and the Dakotas face weather-dependent agricultural cycles that generate consistently cautious language.

**St. Louis** covers Missouri and Arkansas. Its Employment sector (+0.52) is its brightest spot.

### Kansas City: The Outlier

Kansas City is the most distinctive district in the dataset. Its **Construction sector is by far the most positive of any district** (+0.61, nearly 4x the national average of +0.17). This likely reflects the ongoing buildout of logistics, distribution, and data center infrastructure across Kansas, Nebraska, Oklahoma, and Colorado.

But its Manufacturing, Prices, and Agriculture are weak. Kansas City also shows the most **regional divergence** -- in July 2024, Kansas City Employment was +0.93 while neighboring Minneapolis was -0.81. These two districts, despite sharing geographic proximity, could not disagree more about labor markets.

**Strongest sectors:** Construction (+0.61), Consumer Spending (+0.56), Employment (+0.46)

### Dallas: Energy's Mixed Blessing

Despite being the oil capital, Dallas Energy sentiment (+0.17) is **well below the national Energy average** (+0.41). This is counterintuitive but makes sense: Dallas contacts talk about energy *problems* (price volatility, drilling slowdowns, layoffs in the Permian Basin) while other districts simply mention energy prices as a pass-through cost.

Dallas is also the only district where a sector is **negative on average**: Construction (-0.04). Its Tourism & Hospitality (+0.63) is its strongest sector and the highest of any district -- likely driven by the convention and business travel economy in Houston and Dallas-Fort Worth.

**Strongest sectors:** Tourism & Hospitality (+0.63), Real Estate (+0.33), Consumer Spending (+0.29)

### San Francisco: The Persistent Pessimist

San Francisco is the **palest on virtually every map**. Three sectors are outright negative on average:

| Sector | Avg Sentiment |
|--------|---------------|
| Financial Services | -0.02 |
| Transportation | -0.06 |
| Construction | -0.07 |

The 12th District covers CA, OR, WA, NV, AZ, UT, ID, HI, AK. The persistent negativity likely reflects:

- **Housing affordability crises** driving negative Real Estate language
- **Tech sector volatility** (layoffs, startup failures) coloring Financial Services and Employment
- **Wildfire, drought, and climate concerns** bleeding into Agriculture and Construction language

San Francisco also has the **lowest sentiment volatility** (std = 0.31 vs national avg of 0.45) -- it doesn't swing wildly, it's just consistently muted. This is the district where the "Fed-speak" is most guarded.

## Cross-Regional Patterns

### Sector Synchronization

Not all sectors move together nationally. Some are highly synchronized across districts (all 12 banks say the same thing), while others reflect genuinely local conditions:

| Sector | Avg Cross-District Correlation | Interpretation |
|--------|-------------------------------|----------------|
| Energy | 0.86 | Moves nationally (oil prices affect everyone) |
| Real Estate | 0.77 | Mostly national (interest rates drive housing) |
| Employment | 0.61 | Mix of national and local |
| Manufacturing | 0.59 | Most locally driven |
| Consumer Spending | 0.57 | Most locally driven |

**Energy** is essentially a national story: when oil moves, every district talks about it the same way. **Manufacturing** is the most locally driven -- regional manufacturing experiences genuinely diverge, making it the best sector for detecting geographic variation.

### COVID Impact (2020)

COVID hit **Employment hardest everywhere**. All 10 worst-hit sector-district pairs were Employment. The drops were massive (-0.70 to -0.94 vs 2019). But the recovery was almost perfectly symmetric: the same districts that fell hardest bounced back fastest by 2021.

### Rate-Hiking Era (2022-2023)

Compared to pre-hiking (2018-2019), the Fed's tightening cycle had sharply different effects by sector:

| Sector | Sentiment Change | Direction |
|--------|-----------------|-----------|
| Agriculture | +0.26 | Improved (high commodity prices) |
| Construction | +0.22 | Improved (ongoing projects) |
| Prices | +0.20 | Improved (inflation peaking = relief) |
| Transportation | -0.37 | Collapsed (freight recession) |
| Real Estate | -0.19 | Declined (mortgage rate shock) |
| Manufacturing | -0.14 | Declined (demand slowdown) |

The **Transportation collapse** (-0.37) during rate hikes is the largest sectoral shift in the data and corresponds to the well-documented "freight recession" of 2022-2023.

### Recent Distress Signals (2023-2025)

Sectors currently in negative sentiment territory:

| Sector | District | Avg Sentiment |
|--------|----------|---------------|
| Transportation | Cleveland | -0.14 |
| Agriculture | San Francisco | -0.11 |
| Transportation | Richmond | -0.11 |
| Transportation | Philadelphia | -0.09 |
| Financial Services | Dallas | -0.09 |
| Manufacturing | San Francisco | -0.09 |

Transportation stress is widespread. San Francisco continues to be the epicenter of negativity.

### The Geographic Pessimism Gradient

The single most visible pattern across all maps is a **West-to-East optimism gradient**:

**Northeast > Midwest > South > West Coast**

This gradient holds for nearly every sector. It likely reflects a combination of:
1. **Regional economic structure** -- the Northeast has more diversified, service-heavy economies that are less cyclically sensitive
2. **Fed contact selection** -- who the regional banks survey may differ systematically
3. **Linguistic norms** -- regional variation in how business leaders describe conditions

## Which Sectors Best Predict Real Economic Activity?

| Sector | Correlation with Coincident Index | Significant? |
|--------|----------------------------------|-------------|
| Tourism & Hospitality | +0.18 | Yes |
| Employment | +0.17 | Yes |
| Agriculture | +0.11 | Yes |
| Financial Services | -0.27 | Yes |
| Real Estate | -0.21 | Yes |
| Transportation | -0.13 | Yes |

**Employment** and **Tourism & Hospitality** sentiment positively predict economic activity -- optimism in these sectors signals growth ahead. **Financial Services** and **Real Estate** have *negative* correlations, suggesting that positive sentiment in these sectors may signal overheating rather than genuine strength.

### Top Predictive Sector-District Pairs

| Sector | District | Correlation | p-value |
|--------|----------|-------------|---------|
| Employment | Cleveland | +0.61 | < 0.001 |
| Manufacturing | Boston | +0.55 | < 0.001 |
| Employment | San Francisco | +0.55 | < 0.001 |
| Tourism & Hospitality | New York | +0.46 | < 0.001 |
| Tourism & Hospitality | Cleveland | +0.45 | < 0.001 |
| Employment | Boston | +0.43 | < 0.001 |
| Manufacturing | Philadelphia | +0.42 | < 0.001 |

Cleveland Employment remains the strongest single predictor. But notably, **San Francisco Employment** (r = +0.55) is highly predictive despite San Francisco being the most pessimistic district overall -- when even the pessimists turn positive, it means something real.
