# Badische Meile 2026 – Race Stats

Visual analysis of the [Badische Meile](https://www.badische-meile.de/) 2026 race results, served as a static GitHub Pages site.

**Live site:** https://hikotei.github.io/2026_05_badische_meile_stats/

---

## What's inside

| File | Purpose |
|---|---|
| `generate_plots.py` | Generates all chart images from the cleaned CSV |
| `docs/index.html` | Self-contained webpage (no build step) |
| `docs/data/race_results_cleaned.csv` | Cleaned race results loaded by the site |
| `docs/images/` | PNG charts embedded in the page |

## Charts

- **Overview** – age distribution, finishing-time distribution, top 15 clubs by finisher count
- **Age vs. Pace** – scatter + regression: does age slow you down?
- **Time by Age Group & Gender** – box plots split by age bracket and M/W
- **Fastest Clubs** – average finish time for clubs with ≥ 5 runners
- **Cumulative Finish Distribution** – ECDF by gender with 1-hour mark

## Usage

**Regenerate charts** (requires pandas, seaborn, matplotlib):

```bash
pip install pandas seaborn matplotlib
python generate_plots.py
```

Output PNGs are written to `docs/images/`.

**Run the site locally:**

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

## Data

Results scraped from the official results page and cleaned into `race_results_cleaned.csv`. Key columns: `platz`, `name`, `verein`, `gender`, `age`, `age_group`, `zeit_min`, `pace`, `percentile`.

## Deployment

The `docs/` folder is served via GitHub Pages from the `main` branch.
