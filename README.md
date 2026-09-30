# Mobile Game Monetization & Player Behavior Analytics

An end-to-end game analytics portfolio project exploring how player behavior and characteristics relate to monetization in a mobile game.

The project combines exploratory analysis, behavioral segmentation, and predictive modeling in a Streamlit dashboard.

## Data source
Picked a dataset available on Kaggle. Game monetization is highly universal in game analytics, and not highly context-dependent like other game metrics, for example, in-game play data, therefore it is more suitable when it comes to a personal project done for skill practice.
URL link: https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025

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

```
data/game_data_clean.csv
```

The Jupyter notebooks contain the step-by-step data preparation, analysis, player segmentation, behavioral analysis, and predictive modeling behind the dashboard.

## Project Structure
```text
game-analytics/
├── app.py
├── requirements.txt
├── data/
│   ├── mobile_game_inapp_purchases.csv
│   └── game_data_clean.csv
└── notebooks/
    ├── 01_data_understanding.ipynb
    ├── 02_data_cleaning.ipynb
    ├── 03_kpi_analysis.ipynb
    ├── 04_player_segmentation.ipynb
    ├── 05_behavior_analysis.ipynb
    └── 06_predictive_modeling.ipynb
```