# Law of Large Numbers (LLN) & Central Limit Theorem (CLT) Simulator

An interactive, high-performance Streamlit dashboard designed to rigorously visualize the statistical behavior of probability distributions across the thin-tailed to fat-tailed spectrum. 

This simulator demonstrates the physical mechanics of statistical convergence—and, critically, the *failure* of convergence in extreme fat-tailed domains (like Pareto or Cauchy distributions).

## Mathematical Foundations & Core Concepts

At a fundamental level, probability and statistics rely heavily on the hidden assumptions of **stationarity** and finite moments. This simulator stress-tests what happens when those mathematical prerequisites are broken.

### The Law of Large Numbers (LLN)

**Intuitive Definition:** If you repeatedly sample from a stable environment, the average of your observations will eventually lock onto the true mathematical average of that environment. As your sample size grows, the noise of individual random events cancels out.

**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of random variables with a true expected value $\mu = E[X]$. Let $\bar{X}_n$ be the sample mean:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$

The Weak Law of Large Numbers states that for any margin of error $\epsilon > 0$, the probability that the sample mean deviates from the true mean approaches zero as $n \to \infty$:

$$
\lim_{n \to \infty} P(\vert{}\bar{X}_n - \mu\vert{} \ge \epsilon) = 0
$$

**Strict Requirements for the LLN:**
* **Finite First Moment (**$E[\vert{}X\vert{}] < \infty$**):** The mathematical expectation of the absolute value of the distribution must be finite. If the mean is undefined (e.g., Cauchy distribution), the sample average will endlessly jump when extreme outliers arrive and will never converge.
* **Identically Distributed (Stationarity):** The variables must come from the exact same probability distribution.
* **Independence:** The variables must not influence one another. 

**Convergence of Specific Sample Statistics:**

| Statistic | Does it Converge? | Requirement / Boundary Condition | 
| ----- | ----- | ----- | 
| **Sample average** | Yes | Requires a finite mean. | 
| **Sample median** | Yes | Requires the CDF to be strictly increasing at the median. | 
| **Sample percentiles** | Yes | Guaranteed by the Glivenko-Cantelli theorem. | 
| **Sample mean absolute dev.** | Yes | Requires a finite mean. | 
| **Sample standard deviation** | Yes | Requires finite variance. | 
| **Sample min & max** | **No** | Governed by Extreme Value Theory, not the LLN. | 

### The Central Limit Theorem (CLT)

While the LLN dictates *where* the sample mean heads, the CLT dictates the *shape* of the errors around that mean.

**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of independent and identically distributed (i.i.d.) random variables with a true expected value $\mu = E[X]$ and a strictly finite variance $\sigma^2 = Var(X) < \infty$. 

The Average of Sample Variables ($\bar{X}_n = \frac{S_n}{n}$) converges to a Normal distribution centered on the true mean. Its variance shrinks proportionally to $n$:

$$
\bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

The Standardized Average ($Z_n$) converges exactly in distribution ($\xrightarrow{d}$) to the Standard Normal distribution:

$$
Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

**Strict Requirements for the CLT:**
* **Finite Variance (**$\sigma^2 < \infty$**):** The core mathematical prerequisite. The variance dictates the scaling factor ($\sigma / \sqrt{n}$) in the formula.
* **Finite Mean (**$\mu$ **exists):** Centering the data is impossible without a defined mean.

### The Generalized Central Limit Theorem & Pre-Asymptotic Convergence

When the strict requirement of finite variance is broken ($\sigma^2 = \infty$), the standard CLT collapses. If the tails decay as a power law (e.g., Pareto distribution where $\alpha < 2$), the sum of the variables is governed by the **Generalized Central Limit Theorem**, converging instead to a **Lévy Alpha-Stable Distribution** retaining infinite variance.

Furthermore, we must account for the **Speed of Convergence (The Slow Law of Large Numbers)**. For fat-tailed distributions, convergence is agonizingly slow. To match the statistical stability of a sample mean derived from 1,000 Gaussian observations:
* **Borderline Fat Tail (**$\alpha = 2$**):** Requires roughly **10,000 observations**.
* **Standard Financial Tail (**$\alpha = 1.5$**):** Requires roughly **1,000,000 observations**.
* **Extreme Fat Tail (**$\alpha = 1.15$**):** Requires a sample size exceeding **100 trillion observations**.

## Supported Distributions & Tail Behavior

All distributions in this simulator are mathematically shifted to a central mean of `0` (where a mean exists) to allow for direct visual comparability.

**1. Discrete & Thin-Tailed (Binomial, Poisson, Geometric):** 
LLN & CLT apply. Convergence is generally fast. Gaussian smoothing happens rapidly.

**2. Continuous Thin-Tailed (Normal, Exponential, Weibull):** 
LLN & CLT apply. Convergence ranges from instantaneous (Normal) to moderate.

**3. Fat-Tailed (Student-t, Pareto):** 
*   If variance exists, CLT applies, but convergence is agonizingly slow (the pre-asymptotic domain). 
*   If variance is infinite but mean exists (e.g., Pareto with $\alpha$ between 1 and 2, Student-t with df between 1 and 2), the standard CLT fails completely. The LLN applies but converges too slowly to be practically relevant. 

**4. The Unruly Distribution (Cauchy):** 
Neither the mean nor the variance exists. Averages wander erratically forever. Both LLN and standard CLT fail completely.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lln-clt-simulator.git
   cd lln-clt-simulator
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage Guide

1.  **Select a Distribution:** Start with a `Normal (Thin Tail)` distribution, then switch to an extreme fat-tailed distribution like `Pareto, α=1.16 (80/20 Principle)` to witness the breakdown of the LLN.
2.  **Select a Statistic:** Toggle between tracking the Sample Mean, Sample Variance, or specific Percentiles.
3.  **Set Sample Sizes:** Adjust the $n$ for the CLT trials and the total $n$ limit for the LLN cumulative path.
4.  **Freeze the Chart Bounds:** Manually lock the Y-axis constraints (e.g., `-1` to `1`). This is critical for seeing massive outliers blast past the expected limits of the chart when examining fat-tailed behavior.

## Dependencies

*   [Streamlit](https://streamlit.io/): Web framework and UI.
*   [SciPy](https://scipy.org/): Core random variable generation and statistical functions.
*   [NumPy](https://numpy.org/): High-performance vectorization and array manipulation.
*   [Plotly](https://plotly.com/python/): Interactive charting.
*   [Pandas](https://pandas.pydata.org/): Data structuring for Plotly.