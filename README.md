# 2025 Infodengue-Mosqlimate Dengue Forecast Sprint

**Team: Strange Attractors**
**Contributor:** Marcio Maciel Bastos – Mosqlimate

---

## Repository Overview

This repository contains the full implementation of our dengue forecasting model for the **2024 Infodengue-Mosqlimate Forecast Sprint**. The model leverages spatial information, population data, and Bayesian inference to produce **state-level weekly dengue case forecasts** with credible intervals.

The core forecasting logic is implemented in the Jupyter Notebook **`WorkingHorsePyro.ipynb`**, which builds and trains the model using **Pyro** with **SVI optimization**.

---

## Repository Structure

```
├── WorkingHorsePyro.ipynb   # Main Jupyter notebook with model code and training pipeline
```

---

## Libraries and Dependencies

The following Python libraries are required to run the model and reproduce results:

* **Core Data Processing**

  * `pandas`
  * `numpy`
* **Probabilistic Modeling**

  * `pyro-ppl`
  * `torch`
* **Visualization & Debugging**

  * `matplotlib`
  * `seaborn`

---

## Data and Variables

### Data Sources

* **Geospatial Data**
  Official IBGE shapefiles ([link](https://www.ibge.gov.br/geociencias/downloads-geociencias.html)) used to:

  * Compute **pairwise inter-city distances** between Brazilian municipalities.
  * Map each municipality to its corresponding state.
* **Population Data**
  Weekly population estimates for each municipality, derived from yearly municipality population present in the the challenge dataset.

  * Resampled to weekly cadence.
  * Short-term projections generated to align with forecast horizons.
* **Epidemiological Data**
  Weekly dengue case counts per municipality from the challenge dataset.

---

### Variables Used

* **Distance Matrix:** Pairwise distances (in km) between municipalities.
* **Weekly Infections:** Official reported dengue cases (municipal level).
* **Weekly Population:** Aligned and forecasted population estimates.
* **Derived Ratios:** Infection rates per capita (`cases / population`).

---

## Data Preprocessing

1. **Population Resampling:**
   Annual data interpolated to weekly values, ensuring alignment with dengue time series.
2. **Spatial Aggregation:**
   Municipal data aggregated to **state level** for the model.
3. **Feature Engineering:**

   * Distance matrix computation.
   * Infection rate normalization (`Y_obs / Population`).
4. **Forecast Horizon Alignment:**
   Population projections extended to match the final prediction date.

---

## Model Description

The model is a **Bayesian state-level time-series forecasting model** combining:

* **Seasonality Component:**
  Modeled using a **Fourier-wrapped Gaussian** to capture periodic dengue patterns.
* **Gravity Component:**
  Transmission potential between states is proportional to population sizes and inversely proportional to inter-state distances.
* **Infection Probability:**
  Directly models `Y_obs / Population` rather than absolute case counts.
* **Bayesian Inference:**
  Implemented in **Pyro**, allowing full posterior sampling and credible interval estimation.

---

## Model Training

### Optimization

* **Inference Method:** Stochastic Variational Inference (SVI) in Pyro.
* **Loss Function:** Evidence Lower Bound (ELBO) minimization.
* **Optimizer:** Adam with learning rate tuning during experimentation.

### Training Flow

1. Load and preprocess dengue, population, and distance data.
2. Construct priors for seasonal and gravity parameters.
3. Define the generative model and likelihood in Pyro.
4. Fit using SVI with mini-batching (if applicable).
5. Sample from the posterior to generate forecasts.

---

## Prediction Uncertainty

* As a **fully Bayesian model**, uncertainty is quantified directly from the posterior distribution.
* **Credible Intervals** generated at:

  * 50%
  * 80%
  * 90%
  * 95%
* Intervals are computed using quantiles of the posterior predictive samples.

---

## Data Usage Restrictions

* Population forecasts are **only** short-term projections to align with dengue prediction horizons — avoiding the use of any future epidemiological information.

---

## Reproducibility

1. Install dependencies:

   ```bash
   pip install pandas numpy pyro-ppl torch matplotlib
   ```
2. Open `WorkingHorsePyro.ipynb` in Jupyter.
3. Run all cells sequentially to:

   * Load and preprocess data.
   * Train the model.
   * Generate forecasts and credible intervals.

---

## Results

Forecast evaluation scores are available in:

```
scores/scores.md
```

---

## Contact

For questions or collaboration:

* **Name:** Marcio Maciel Bastos
* **email** marciobastos@fisica.ufc.br
* **Platform:** [Mosqlimate](https://api.mosqlimate.org/)
* **Challenge:** [Infodengue-Mosqlimate Forecast Sprint](https://sprint.mosqlimate.org/)
