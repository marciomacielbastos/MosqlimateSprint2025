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
Alright — I’ll build a **Mathematical Formulation** section for your updated README by translating the structure of your `WorkingHorsePyro.ipynb` into equations, matching the level of completeness and tone from the example file, but written in the same didactic style we’ve been using.

---

### Mathematical Formulation

Let:

* $S$ = number of states
* $T$ = number of epidemiological weeks in the training period
* $y_{s,t}$ = observed number of dengue cases in state $s$ at week $t$
* $N_{s,t}$ = estimated population of state $s$ at week $t$ (weekly resampled and short-term forecasted)
* $r_{s,t} = \frac{y_{s,t}}{N_{s,t}}$ = infection rate in state $s$ at week $t$
* $d_{s,s'}$ = great-circle distance between centroids of states $s$ and $s'$ (from IBGE shapefiles)

We model $r_{s,t}$ as:

$$
r_{s,t} \sim \mathrm{Distribution}\big( \lambda_{s,t}, \sigma \big)
$$

where $\lambda_{s,t}$ is the latent expected infection probability, and $\sigma$ is the observation noise scale.

---

#### 1. Seasonal Component

Seasonality is captured using a **Fourier-wrapped Gaussian** over epidemiological weeks:

$$
\text{Season}_t = \alpha _s \cdot \exp\left( - \frac{\left[ \min(|t - \mu|, 52 - |t - \mu|)\right]^2}{ 2\sigma_{\text{season}}^2 } \right)
$$

* $\mu$ = peak week of dengue season (global or state-specific)
* $\sigma_\text{season}$ = seasonal spread in weeks
* $\alpha_s$ = amplitude for state $s$

---

#### 2. Gravity Component

The gravity term models cross-state influence, proportional to population and inversely proportional to distance:

$$
\mathrm{Gravity}_{s,t} =
\sum_{s'=1}^{S} \mathbf{1}\!\left[s' \ne s\right] \;
\beta \, \frac{N_{s',t}^{\eta} \; r_{s',t-\ell}}{d_{s,s'}^{\gamma}}
$$


* $\beta$ = global scaling coefficient for imported infections
* $\eta$ = exponent controlling influence of source population
* $\gamma$ = distance decay exponent
* $\ell$ = lag in weeks before imported effect is felt

---

#### 3. Latent Mean Infection Rate

The latent mean infection rate is:

$$
\lambda_{s,t} = \underbrace{\theta_s}_{\text{baseline}} + \underbrace{\text{Season}_t}_{\text{seasonality}} + \underbrace{\text{Gravity}_{s,t}}_{\text{spatial importation}}
$$

* $\theta_s$ = baseline log-infection probability for state $s$

---

#### 4. Likelihood

Since $r_{s,t} \in (0, 1)$, we can use a **positive-definite likelihood** such as the Beta distribution:

$$
r_{s,t} \sim \mathrm{Beta}\!\left( \alpha_{s,t}, \, \beta_{s,t} \right)
$$

where:

$$
\alpha_{s,t} = \lambda_{s,t} \cdot \phi, \quad \beta_{s,t} = (1 - \lambda_{s,t}) \cdot \phi
$$

and $\phi > 0$ is a precision parameter controlling the width of the distribution.

---

#### 5. Prior Distributions

Example priors used in the Pyro implementation:

$$
\begin{aligned}
\mu &\sim \mathrm{Uniform}(0, 52) \\
\sigma_\text{season} &\sim \mathrm{LogNormal}(\log 4, \, 0.3) \\
\theta_s &\sim \mathrm{Normal}(0, 1) \\
\beta &\sim \mathrm{HalfNormal}(1) \\
\eta &\sim \mathrm{Uniform}(0, 2) \\
\gamma &\sim \mathrm{Uniform}(0, 3) \\
\phi &\sim \mathrm{HalfNormal}(5)
\end{aligned}
$$

---

### 6. Posterior Inference

We estimate the posterior:

$$
p(\Theta \mid \mathbf{r}) \propto p(\mathbf{r} \mid \Theta) \; p(\Theta)
$$

where:

* $\Theta = \{\mu, \sigma_\text{season}, \theta_s, \beta, \eta, \gamma, \phi \}$
* Inference is performed using **Stochastic Variational Inference (SVI)** in Pyro, minimizing the **negative ELBO**.

---

### 7. Forecasting

For future weeks $t^\*$, forecasts are generated by:

$$
\hat{r}_{s,t^*} \sim p\!\left( r_{s,t^*} \mid \mathbf{r}_{\text{train}} \right)
$$

using posterior samples of $\Theta$ and recursively feeding predictions into the gravity term.


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

---

## Contact

For questions or collaboration:

* **Name:** Marcio Maciel Bastos
* **email** marciobastos@fisica.ufc.br
* **Platform:** [Mosqlimate](https://api.mosqlimate.org/)
* **Challenge:** [Infodengue-Mosqlimate Forecast Sprint](https://sprint.mosqlimate.org/)
