# Traffic Demand Forecasting using Ensemble Gradient Boosting

## Overview

This repository contains a machine learning solution developed for the Flipkart Grid Hackathon Traffic Demand Forecasting challenge.

The objective was to predict traffic demand using location, temporal, road, weather, and environmental features. The final solution combines advanced feature engineering, Bayesian-smoothed target encoding, and a weighted ensemble of LightGBM, XGBoost, and CatBoost regressors.

## Performance

**Competition Score:** 89.43353 / 100

**Official Evaluation Metric**

```python
score = max(0, 100 * R²(actual, predicted))
```

The achieved score demonstrates the effectiveness of combining spatio-temporal feature engineering with ensemble learning techniques for traffic demand forecasting.

---

## Problem Statement

Given traffic-related attributes such as:

* Geohash location
* Timestamp
* Road type
* Number of lanes
* Weather conditions
* Nearby landmarks
* Large vehicle permissions
* Temperature

predict the traffic demand at a given location and time.

---

## Solution Pipeline

```text
Raw Data
    ↓
Feature Engineering
    ↓
Bayesian-Smoothed Target Encoding
    ↓
LightGBM
XGBoost
CatBoost
    ↓
Validation-Based Weight Optimization
    ↓
Full Dataset Retraining
    ↓
Weighted Ensemble Prediction
    ↓
Prediction Clipping
    ↓
Final Submission
```

---

## Feature Engineering

### Time Decomposition

The timestamp feature is decomposed into:

* Hour
* Minute

allowing the model to learn hourly traffic behavior.

### Cyclical Time Encoding

Traffic follows daily recurring patterns.

To preserve the cyclic nature of time:

```python
sin_time = sin(2π × time / 1440)
cos_time = cos(2π × time / 1440)
```

These features help capture periodic traffic trends throughout the day.

### Traffic Event Flags

Additional binary features are created:

* Morning Commute (07:00–09:00)
* School Rush (15:00–17:00)
* Dead Night (23:00–05:00)

These features introduce real-world traffic context into the model.

### Spatio-Temporal Interaction

A combined feature is created:

```python
geohash_hour = geohash + "_" + hour
```

This enables the model to learn location-specific traffic patterns at different times of the day.

---

## Bayesian-Smoothed Target Encoding

To capture historical traffic behavior while preventing overfitting, Bayesian-smoothed target encoding is applied to the `geohash_hour` feature.

Encoding formula:

```text
Encoded Value =
(Count × Category Mean + Weight × Global Mean)
/
(Count + Weight)
```

Benefits:

* Reduces overfitting on rare categories
* Stabilizes historical demand estimates
* Improves generalization on unseen samples

---

## Models

### LightGBM

Configuration:

```python
n_estimators = 1000
learning_rate = 0.03
max_depth = 8
num_leaves = 63
```

Used for fast and efficient gradient boosting on structured data.

### XGBoost

Configuration:

```python
n_estimators = 800
learning_rate = 0.03
max_depth = 8
tree_method = "hist"
```

Used for learning complex non-linear feature interactions with strong regularization.

### CatBoost

Configuration:

```python
iterations = 800
learning_rate = 0.04
depth = 8
```

Chosen for its native handling of categorical features and strong performance on tabular datasets.

---

## Ensemble Strategy

Three independent models are trained:

* LightGBM
* XGBoost
* CatBoost

Predictions from all models are combined through a weighted ensemble.

The ensemble prediction is:

```text
Prediction =
w1 × LightGBM
+ w2 × XGBoost
+ w3 × CatBoost
+ Bias
```

Weights are learned using validation-set predictions and gradient descent.

---

## Training Methodology

### Phase 1 – Weight Discovery

* Split dataset into 80% training and 20% validation.
* Train all base models.
* Generate validation predictions.
* Learn optimal ensemble weights.

### Phase 2 – Full Retraining

* Retrain all models on 100% of the available training data.
* Capture all observable traffic patterns before final prediction.

### Phase 3 – Final Prediction

* Generate predictions using retrained models.
* Apply optimized ensemble weights.
* Clip negative predictions to zero.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* XGBoost
* CatBoost

---

## Key Highlights

* Achieved a competition score of 89.43353 on the Flipkart Grid Traffic Forecasting Challenge.
* Engineered spatio-temporal traffic features from raw location and timestamp data.
* Implemented Bayesian-smoothed target encoding for robust historical demand estimation.
* Built a weighted ensemble of LightGBM, XGBoost, and CatBoost models.
* Optimized ensemble weights using validation-based gradient descent.
* Applied full-dataset retraining to maximize predictive performance.

---

## Future Work

* Graph Neural Networks for spatial traffic modeling.
* Transformer-based temporal forecasting.
* Dynamic road network embeddings.
* Multi-horizon traffic demand prediction.
* Integration of graph-based geospatial features.

