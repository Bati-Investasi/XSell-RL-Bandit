# Cross-Sell Bandit

A contextual bandit implementation for multi-product cross-selling in banking.

## Code Highlights

### Cold-start Friendly
- Starts from optional propensity scores to "warm-boot" its value estimates
- Uses an ε-greedy exploration policy (switch to UCB or Thompson Sampling later with minor tweaks)

### Online-learnable
- Incrementally updates with `partial_fit`, so you can call it after each new interaction or in mini-batches

### Interpretable
- Logistic-regression (SGD) oracle; coefficients tell you which customer features drive conversion for each product

### Easily Pluggable
- Agent is a pure-Python class with `act(state_df)` and `update(log_df)` methods
- No external RL frameworks required—just numpy, pandas, scikit-learn

## How to Integrate in Production

### 1. Load Features
Replace the toy `ctx` DataFrame with your real-time (or daily-batch) customer feature set.

### 2. Warm-start (Optional)
If you already run a propensity model, pass its predictions as `warm_propensity`.

### 3. Decision Call
```python
offers = bandit.act(feature_df)  # picks product for each customer
```
Immediately log `(customer_id, action, context_features)` so you can join with the eventual outcome.

### 4. Outcome Capture
Whenever a customer converts (or after the observation window), construct an `OfferResult` and call:
```python
bandit.update([result1, result2, ...])  # can be single or batch
```

### 5. Persistence
Dump the bandit object (e.g., `joblib.dump`) on shutdown and reload it on startup so learning continues across restarts.

## Hyper-parameters

- `epsilon` → exploration rate (start at 0.1–0.2 in cold start, decay toward 0.05)
- `alpha` / `penalty` → regularization; tweak if you see over-/under-fitting

## Advanced Customization

### Switching to UCB/Thompson
Replace the `act` logic:
- For Thompson: keep Beta priors per action and sample
- For UCB: track counts and mean reward + confidence bound

The underlying SGD oracle remains the same.

### Fairness / Guardrails
Before returning offers, filter with business-rule functions:
- Eligibility
- Frequency caps
- etc.

## Conclusion
That's all you need for a lean, interpretable RL agent that learns on every customer interaction while starting smart from your propensity model. Adapt feature engineering, reward definition, and exploration strategy to your bank's specific needs, and you'll have a scalable foundation for hyper-personalized cross-sell.