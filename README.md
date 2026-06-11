# Day-Ahead Electricity Price Modeling

Predicts tomorrow's 24 hourly day-ahead spot prices for a Swedish price zone
(SE4, southern Sweden, by default) from today's prices, SMHI wind forecasts
and the calendar.

## Approach

Absolute price levels have moved enormously since 2021 - the energy crisis
pushed them up and back down again - so a model trained on absolute prices
mostly learns "which year is it". This one predicts relative to today
instead: every training row is one (day, hour) pair with today's price at
the same hour as a feature, and the model only has to learn the day-over-day
adjustment, which stays stable across price regimes.

It's hourly rather than daily because the within-day swing is bigger than
the day-to-day level: the median intraday range here (~1.11 SEK/kWh) beats
the median daily mean (~0.75). A daily average would throw away the duck
curve (cheap nights, morning ramp, midday solar dip, evening peak), and that
shape is most of what there is to predict.

## Weather data

The model's weather input is wind (the supply side: SE4 prices drop when the
wind farms produce), averaged across a few hand-picked coastal stations and
fetched from the SMHI metobs API into `./cache`, so there are no data files
to download manually. Temperature is fetched and analysed the same way but
ended up cut from the feature set - more below. One catch: SMHI's
quality-controlled corrected-archive stops about 3 months short of today,
so each station is topped up with the latest-months feed (the cached copy
refreshes daily, archives monthly). At predict time the same feed supplies
the hours of today that have already happened and the forecast covers the
rest, so the today-mean features are full-day means, same as in training.

- Wind (coastal / wind-farm points): Falsterbo, Skillinge, Hallands Väderö,
  Karlskrona, Kalmar
- Temperature (population centres, EDA only): Malmö, Halmstad, Växjö,
  Karlskrona, Kalmar

Station ids and coordinates live in `historical_data.py`. Using all active
SE4 stations instead was tried and scored worse - the extras are spatially
correlated with the curated ones and mostly add noise. Population-weighting
the temperature stations by county and adding more Skåne population centres
(Helsingborg, Lund, Kristianstad) failed the same test.

Why no temperature? Heating demand follows the season rather than the
day-to-day wiggle, and the seasonal level is already carried by the season
features and the day-before price anchor. A six-fold seasonal backtest
(4 months per fold, two winters covered, three seeds per fit) confirmed it:
dropping all three temperature features cost +0.005-0.017 MAE in the winter
and spring folds and gained more than that in the summer folds, where
temperature mostly adds noise - net roughly zero. The columns are still
built and shown in the EDA tab, and the backtest lives on as
`analysis.seasonal_backtest` (`python analysis.py --backtest`), so the call
can be re-checked as more winters accumulate. Putting temperature back is a
one-line change in `hourly_dataset.py`.

## Features

All knowable on the afternoon of day D, before tomorrow:

| Feature                                                              | Description                             |
| -------------------------------------------------------------------- | --------------------------------------- |
| `price_today_same_hour`                                              | today's price at this hour (the anchor) |
| `price_today_daily_mean`                                             | today's overall level                   |
| `wind_today_mean`                                                    | today's zone-mean wind                  |
| `wind_tomorrow_hour`                                                 | tomorrow's forecast at this hour        |
| `wind_delta_hour`                                                    | ... minus today's daily mean            |
| `hour_sin`, `hour_cos`                                               | hour of day as a cycle                  |
| `day_of_week_tomorrow`, `is_weekend_tomorrow`, `is_holiday_tomorrow` | calendar                                |
| `day_of_year_sin/cos`, `month`                                       | season                                  |

Target: `price_tomorrow_hour` (SEK/kWh). The model is a
HistGradientBoostingRegressor on the long table, ~40k rows. Prices are
grouped by local (date, hour), so the 15-minute records used since 2025
average up to hourly and DST 23/25-hour days come out right without
special-casing.

## Results

Chronological holdout: train on the past, predict the last 4 unseen months.
The baseline is hourly persistence, "tomorrow's hour h = today's hour h",
which is a strong one since the daily shape barely moves between days.

|             | MAE (SEK/kWh) | RMSE (SEK/kWh) |
| ----------- | ------------- | -------------- |
| Model       | 0.266         | 0.346          |
| Baseline    | 0.346         | 0.479          |
| Improvement | +23.1%        | +27.8%         |

(June 2026 run; the numbers drift a little as the cache refreshes.)

The model reproduces the duck curve and beats persistence, with the error
concentrated in the volatile morning-ramp and evening-peak hours.
`python hourly_model.py` regenerates `hourly_evaluation.png`, a sample
fortnight plus the average daily shape.

## Analysis and dashboard

`analysis.py` holds the supporting statistics; `dashboard.py` is a
Plotly/Dash app with three tabs.

**1. Exploratory analysis.** Target summary (mean 0.93, std 0.97, ~3.6%
negative hours, median intraday range 1.11), the price distribution and
daily shape, the two weather drivers side by side (price falls clearly with
wind, barely moves with temperature), and a records-per-day breakdown of the
feed (1423 hourly days, 252 fifteen-minute days, a handful of DST days).

**2. Feature selection.** Cleaning happens in the loaders: quality-flag
filter, type coercion, dropping incomplete rows. Negative prices are real
market outcomes, not data errors, so they stay. Predictive power is
checked three ways: Pearson r against the target, a feature correlation
matrix, and one-way ANOVA reported as eta² (variance explained) instead of
p-values, because with ~40k rows everything is "significant" and only effect
size ranks anything. Hour of day explains the most price variance, 6.4%.

Most features sit at |r| < 0.2 against the price and are kept anyway. That's
deliberate: Pearson r only measures straight-line, one-feature-at-a-time
relationships, and a boosted tree picks up non-linear and interaction
effects that r is blind to. The better test is the drop-column ablation in
`analysis.ablation_study` - retrain without each group, remeasure the
holdout error. Dropping hour-of-day (|r| ≈ 0.10) costs +0.012 MAE, because
the daily curve is cyclical and a straight line can't fit that. Temperature
is the flip side of the same logic: it failed the ablation across seasons
and was cut (see Weather data). The dashboard shows both views side by side.

**3. Model & evaluation.** KPI cards (MAE 0.266, down 23% on the baseline;
RMSE 0.346, down 28%; R² 0.598; beats the baseline on 59% of hours, by
+0.08 SEK/kWh on average), a scoreboard, permutation importances,
predicted-vs-actual, per-hour MAE against the baseline, a histogram of the
per-hour margin over the baseline (frequency of wins can hide rare big
losses; the margin shows the sizes too), residuals, and a month-by-month
forecast browser.

```bash
python analysis.py               # text version of the same statistics
python analysis.py --backtest    # + the seasonal feature backtest (slower)
python dashboard.py              # then open http://127.0.0.1:8050
```

## Setup and usage

```bash
pip install -r requirements.txt

# Holdout evaluation + graph. The first run downloads prices and station
# archives into ./cache (a few minutes); after that it's quick.
python hourly_model.py
python hourly_model.py --months 6     # longer holdout
python hourly_model.py --zone SE3     # another price zone

# Train on everything and print tomorrow's 24-hour curve.
python hourly_model.py --predict
python hourly_model.py --predict --retrain
```

## Files

| File                   | Does                                    |
| ---------------------- | --------------------------------------- |
| `data_fetchers.py`     | price API + SMHI SNOW forecast clients  |
| `historical_data.py`   | station registry + SMHI CSV parser      |
| `smhi_stations.py`     | download/cache the station archives     |
| `price_cache.py`       | incremental on-disk price cache         |
| `calendar_features.py` | Swedish holidays + calendar features    |
| `hourly_dataset.py`    | assemble the (day x hour) feature table |
| `hourly_model.py`      | train / evaluate / predict              |
| `analysis.py`          | EDA, feature stats, ablation, backtest  |
| `dashboard.py`         | the Dash app                            |

## Limitations, ideas

- Wind speed is used directly, but generation scales roughly with speed³ up
  to rated speed and flattens after; a cubed/clipped transform would
  probably be a better supply proxy.
- Hydro reservoir levels, nuclear availability, cross-border flows and gas
  prices are all missing.
- Training uses observed weather as "tomorrow's forecast" while production
  uses a real forecast, so live accuracy will be a bit worse than the
  holdout suggests.
- The 2025+ 15-minute resolution could be modelled directly instead of
  averaged away.
