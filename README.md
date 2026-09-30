# Mobile Game Monetization & Player Behavior Analytics

An end-to-end game analytics portfolio project exploring how player behavior and characteristics relate to monetization in a mobile game.

The project combines exploratory analysis, behavioral segmentation, and predictive modeling in a Streamlit dashboard.

## Business Question

> What player behaviors and characteristics are associated with higher monetization value, and how can players be segmented to support game monetization and engagement decisions?

The analysis focuses on three areas:

- Understanding the distribution and concentration of player revenue
- Identifying distinct player profiles based on engagement and monetization
- Testing whether player-level features can predict high-value players


##  Run locally

Requires Python 3.10+.

```bash
cd game-analytics
conda activate myenv
pip install -r requirements.txt
streamlit run app.py
```

The browser dashboard will open locally, typically at http://localhost:8501.

The project uses the cleaned dataset located at:

```data/game_data_clean.csv
```

The Jupyter notebooks contain the step-by-step data preparation, analysis, player segmentation, behavioral analysis, and predictive modeling behind the dashboard.
