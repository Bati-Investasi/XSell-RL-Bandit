"""
cross_sell_bandit.py
Contextual-Bandit RL agent for multi-product cross-sell in banking.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

@dataclass
class OfferResult:
    """Represents the outcome of an offer made to a customer.
    
    Attributes:
        customer_id: Unique identifier for the customer
        action: The product that was offered
        reward: Binary outcome (1 for conversion, 0 for no conversion)
        context: Customer features at the time of the offer
    """
    customer_id: str
    action: str          # product offered
    reward: float        # 1 = converted, 0 = not converted
    context: pd.Series   # features at decision time

class EpsilonGreedyBandit:
    """
    Contextual ε-greedy bandit with a separate binary classifier per action.
    Each classifier estimates P(conversion | context, action).

    This implementation uses online learning to update models incrementally
    as new data becomes available. It supports warm-starting with propensity
    scores and handles cold-start scenarios gracefully.
    """

    def __init__(
        self,
        actions: List[str],
        epsilon: float = 0.1,
        warm_propensity: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        """
        Initialize the bandit algorithm.

        Parameters
        ----------
        actions : List[str]
            List of product codes/names that can be offered
        epsilon : float, optional
            Exploration probability (default: 0.1)
        warm_propensity : Optional[pd.DataFrame], optional
            Initial conversion probability estimates, indexed by customer_id
            and with columns matching actions
        random_state : int, optional
            Random seed for reproducibility (default: 42)

        Raises
        ------
        ValueError
            If actions list is empty or epsilon is not in [0,1]
        """
        if not actions:
            raise ValueError("Actions list cannot be empty")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
            
        self.actions = actions
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)

        # One online logistic reg model per action
        self.models: Dict[str, SGDClassifier] = {
            a: make_pipeline(
                StandardScaler(with_mean=False),  # sparse-safe
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    learning_rate="optimal",
                    random_state=random_state,
                ),
            )
            for a in actions
        }
        self.classes_ = np.array([0, 1])  # needed for partial_fit bootstrap
        self._bootstrap_done = {a: False for a in actions}

        # Validate warm propensity if provided
        if warm_propensity is not None:
            if not all(a in warm_propensity.columns for a in actions):
                raise ValueError("Warm propensity must contain all actions as columns")
            if not all(0 <= p <= 1 for p in warm_propensity.values.flatten()):
                raise ValueError("Warm propensity values must be between 0 and 1")
        self.warm_propensity = warm_propensity

    # ------------- Public API -------------------------------------------------

    def act(self, context_df: pd.DataFrame) -> pd.Series:
        """
        Choose an action (product) for each customer in the context DataFrame.

        Parameters
        ----------
        context_df : pd.DataFrame
            DataFrame containing customer features, indexed by customer_id

        Returns
        -------
        pd.Series
            Series of chosen actions, indexed by customer_id

        Raises
        ------
        ValueError
            If context_df is empty or missing required features
        """
        if context_df.empty:
            raise ValueError("Context DataFrame cannot be empty")
            
        scores = self._predict_scores(context_df)  # shape (n, |A|)
        chosen_actions = []

        for i in range(len(context_df)):
            if self.rng.random() < self.epsilon:
                # Explore: sample uniformly among **eligible** actions
                chosen_actions.append(self.rng.choice(self.actions))
            else:
                # Exploit: pick action with highest estimated prob
                chosen_actions.append(self.actions[int(np.argmax(scores[i]))])

        return pd.Series(chosen_actions, index=context_df.index, name="action")

    def update(self, logs: List[OfferResult]) -> None:
        """
        Update the models with newly observed offer outcomes.

        Parameters
        ----------
        logs : List[OfferResult]
            List of offer results to use for model updates

        Raises
        ------
        ValueError
            If any log entry has invalid action or reward
        """
        if not logs:
            return

        # Validate logs
        for log in logs:
            if log.action not in self.actions:
                raise ValueError(f"Invalid action {log.action} in logs")
            if not isinstance(log.reward, (int, float)) or not 0 <= log.reward <= 1:
                raise ValueError(f"Invalid reward {log.reward} in logs")

        # Build mini-batch per action
        batch_by_action: Dict[str, List[OfferResult]] = {a: [] for a in self.actions}
        for r in logs:
            batch_by_action[r.action].append(r)

        for action, batch in batch_by_action.items():
            if not batch:
                continue

            X = pd.DataFrame([res.context for res in batch])
            y = np.array([res.reward for res in batch])

            model = self.models[action]
            # First call to partial_fit must include classes
            if not self._bootstrap_done[action]:
                model.partial_fit(X, y, classes=self.classes_)
                self._bootstrap_done[action] = True
            else:
                model.partial_fit(X, y)

    # ------------- Internal helpers ------------------------------------------

    def _predict_scores(self, context_df: pd.DataFrame) -> np.ndarray:
        """
        Predict conversion probabilities for all (customer, action) combinations.

        Parameters
        ----------
        context_df : pd.DataFrame
            DataFrame containing customer features

        Returns
        -------
        np.ndarray
            Array of shape (n_customers, n_actions) containing predicted
            conversion probabilities
        """
        n = len(context_df)
        k = len(self.actions)
        scores = np.zeros((n, k))

        for j, action in enumerate(self.actions):
            model = self.models[action]

            if self._bootstrap_done[action]:
                scores[:, j] = np.clip(
                    model.predict_proba(context_df)[:, 1], 0.001, 0.999
                )
            else:
                # no data yet: use warm propensity if provided, else uniform prior
                if self.warm_propensity is not None:
                    # align on index; if missing, default 0.05
                    prop = self.warm_propensity.reindex(
                        context_df.index
                    )[action].fillna(0.05)
                    scores[:, j] = prop.values
                else:
                    scores[:, j] = 0.05  # 5 % prior

        return scores

# -------------------- Example usage ------------------------------------------

if __name__ == "__main__":
    # Fake customer context (replace with real features)
    n_customers = 500
    actions = ["FX_CARD", "FX_SAVINGS", "FX_INVEST"]

    rng = np.random.default_rng(1)
    ctx = pd.DataFrame(
        {
            "travel_freq": rng.integers(0, 5, n_customers),
            "salary_usd": rng.normal(75000, 15000, n_customers),
            "segment_student": rng.integers(0, 2, n_customers),
            "market_vol_high": rng.integers(0, 2, n_customers),
        },
        index=[f"cust_{i}" for i in range(n_customers)],
    )

    # Optional warm-start propensity guesses (here random for toy)
    warm_prop = pd.DataFrame(
        rng.uniform(0.02, 0.12, (n_customers, len(actions))),
        columns=actions,
        index=ctx.index,
    )

    bandit = EpsilonGreedyBandit(actions, epsilon=0.1, warm_propensity=warm_prop)

    # --- Simulated interaction loop ---
    logs: List[OfferResult] = []
    for day in range(30):
        # choose actions
        offers = bandit.act(ctx)

        # Simulate reward: customers who travel a lot & high salary
        reward_prob = (
            0.1 * ctx["travel_freq"]
            + 0.000002 * ctx["salary_usd"]
            + 0.05 * ctx["market_vol_high"]
        )
        reward_prob = np.clip(reward_prob, 0, 0.9)

        # Bernoulli draw
        rewards = rng.random(n_customers) < reward_prob

        # collect logs
        day_logs = [
            OfferResult(cust, act, float(rwd), ctx.loc[cust])
            for cust, act, rwd in zip(ctx.index, offers, rewards)
        ]
        logs.extend(day_logs)

        # online update every 10 k interactions
        if len(logs) > 10000:
            batch, logs = logs[:10000], logs[10000:]
            bandit.update(batch)

    # final update for leftovers
    bandit.update(logs)

    # inspect learned coefficients
    for a in actions:
        print(f"\n=== Model for {a} ===")
        coef = bandit.models[a].named_steps["sgdclassifier"].coef_.ravel()
        print(dict(zip(ctx.columns, coef)))
