"""
statshelper.py

A comprehensive set of ~30 analysis functions spanning:
1) 8 core functions
2) 12 additional general-purpose analyses
3) 5 specialized/advanced techniques
4) 5 time-change/trend functions
5) 2 advanced models (mixed-effects, generalized linear)

PLUS expansions using scikit-learn, lifelines, semopy for tasks like PCA, clustering, survival, SEM, etc.

All in one script. Feel free to refactor into separate files.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind, f_oneway, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Additional libraries for advanced tasks:
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from lifelines import KaplanMeierFitter, CoxPHFitter
import semopy

###############################################################################
# A) Your Existing 8 "Core" Functions
###############################################################################


def basic_correlation(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Purpose: Computes a simple Pearson correlation between two numeric columns (x_col, y_col).
    Use case: Quick check of linear association (e.g., "Does budget correlate with total sales?").
    """
    try:
        sub = df[[x_col, y_col]].dropna()
        r, p = pearsonr(sub[x_col], sub[y_col])
        return {
            "function": "basic_correlation",
            "x_col": x_col,
            "y_col": y_col,
            "r_value": r,
            "p_value": p,
            "n": len(sub)
        }
    except Exception as e:
        return {"error": str(e)}


def grouped_correlation(df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> dict:
    """
    Purpose: Runs basic_correlation separately for each subgroup in group_col.
    Use case: "Does budget vs. sales correlation differ by brand, or by state?"
    """
    results = {}
    try:
        grouped = df.dropna(
            subset=[x_col, y_col, group_col]).groupby(group_col)
        for g, subdf in grouped:
            if len(subdf) < 2:
                results[g] = {"r_value": None,
                              "p_value": None, "n": len(subdf)}
            else:
                r, p = pearsonr(subdf[x_col], subdf[y_col])
                results[g] = {"r_value": r, "p_value": p, "n": len(subdf)}
        return {"function": "grouped_correlation", "results_by_group": results}
    except Exception as e:
        return {"error": str(e)}


def simple_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Purpose: Regress y_col on a single predictor x_col (simple OLS).
    Use case: "What is the slope/intercept of budget -> sales?" (R², significance, etc.).
    """
    try:
        sub = df[[x_col, y_col]].dropna()
        formula = f"{y_col} ~ {x_col}"
        model = smf.ols(formula=formula, data=sub).fit()
        return {
            "function": "simple_linear_regression",
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "r_squared": model.rsquared,
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def multiple_regression(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    """
    Purpose: Multiple OLS regression with multiple predictors in x_cols.
    Use case: "How do budget, sessions, brand rating, etc., collectively affect sales?"
    """
    try:
        sub = df[[y_col] + x_cols].dropna()
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        model = smf.ols(formula=formula, data=sub).fit()
        return {
            "function": "multiple_regression",
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "r_squared": model.rsquared,
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def polynomial_regression(df: pd.DataFrame, x_col: str, y_col: str, degree: int = 2) -> dict:
    """
    Purpose: Fits a polynomial relationship (e.g., quadratic) between x_col and y_col.
    Use case: "Check if higher budgets lead to diminishing returns on sales."
    """
    try:
        sub = df[[x_col, y_col]].dropna()
        # Manually create polynomial terms
        for d in range(2, degree+1):
            sub[f"{x_col}^{d}"] = sub[x_col]**d
        poly_terms = " + ".join([f"{x_col}^{d}" for d in range(1, degree+1)])
        formula = f"{y_col} ~ {poly_terms}"
        model = smf.ols(formula=formula, data=sub).fit()
        return {
            "function": "polynomial_regression",
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "r_squared": model.rsquared,
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def lagged_correlation(df: pd.DataFrame, group_col: str, budget_col: str, sales_col: str) -> dict:
    """
    Purpose: Within each group, shift budget_col by 1 time period and correlate with current sales_col.
    Use case: "Does last month's budget drive this month's sales?"
    """
    results = {}
    try:
        if "MONTH" not in df.columns:
            return {"error": "MONTH column not found for lag operation."}
        df_sorted = df.sort_values(by=[group_col, "MONTH"])
        for g, subdf in df_sorted.groupby(group_col):
            subdf = subdf[[budget_col, sales_col, "MONTH"]].copy()
            subdf["budget_lag1"] = subdf[budget_col].shift(1)
            valid = subdf.dropna()
            if len(valid) < 2:
                results[g] = {"r_value": None,
                              "p_value": None, "n": len(valid)}
            else:
                r, p = pearsonr(valid["budget_lag1"], valid[sales_col])
                results[g] = {"r_value": r, "p_value": p, "n": len(valid)}
        return {"function": "lagged_correlation", "results_by_group": results}
    except Exception as e:
        return {"error": str(e)}


def efficiency_tiers_correlation(df: pd.DataFrame, budget_col: str, sales_col: str) -> dict:
    """
    Purpose: Splits data into high/low efficiency tiers by (sales/budget), then runs correlation by tier.
    Use case: "Do high-efficiency dealers show a different budget–sales correlation vs. low-efficiency?"
    """
    try:
        temp = df[[budget_col, sales_col]].dropna()
        temp["efficiency"] = temp[sales_col] / \
            temp[budget_col].replace(0, np.nan)
        temp = temp.dropna(subset=["efficiency"])
        median_eff = temp["efficiency"].median()
        temp["eff_tier"] = np.where(
            temp["efficiency"] >= median_eff, "high", "low")

        results = {}
        for tier, subdf in temp.groupby("eff_tier"):
            if len(subdf) < 2:
                results[tier] = {"r_value": None,
                                 "p_value": None, "n": len(subdf)}
            else:
                r, p = pearsonr(subdf[budget_col], subdf[sales_col])
                results[tier] = {"r_value": r, "p_value": p, "n": len(subdf)}
        return {"function": "efficiency_tiers_correlation", "results": results}
    except Exception as e:
        return {"error": str(e)}


def mediation_analysis(df: pd.DataFrame, x_col: str, mediator_col: str, y_col: str) -> dict:
    """
    Purpose: Classic 3-step mediation (X->Y, X->M, then X+M->Y).
    Use case: "Does website traffic (mediator) partly explain budget->sales link?"
    """
    try:
        sub = df[[x_col, mediator_col, y_col]].dropna()
        model1 = smf.ols(f"{y_col} ~ {x_col}", data=sub).fit()
        model2 = smf.ols(f"{mediator_col} ~ {x_col}", data=sub).fit()
        model3 = smf.ols(f"{y_col} ~ {x_col} + {mediator_col}", data=sub).fit()
        return {
            "function": "mediation_analysis",
            "X->Y_coefficient": model1.params.get(x_col, None),
            "X->Y_pvalue": model1.pvalues.get(x_col, None),
            "X->M_coefficient": model2.params.get(x_col, None),
            "X->M_pvalue": model2.pvalues.get(x_col, None),
            "X+M->Y_coefficient_of_X": model3.params.get(x_col, None),
            "X+M->Y_pvalue_of_X": model3.pvalues.get(x_col, None),
            "n": int(model3.nobs)
        }
    except Exception as e:
        return {"error": str(e)}

###############################################################################
# B) 12 Additional General-Purpose Analyses
###############################################################################


def partial_correlation(df: pd.DataFrame, x_col: str, y_col: str, covariates=[]) -> dict:
    """
    Purpose: Correlation between x_col and y_col while controlling for numeric covariates.
    Use case: "Is there still a correlation between budget and sales after controlling for brand size or traffic?"
    """
    try:
        sub = df[[x_col, y_col] + covariates].dropna()
        formula_x = f"{x_col} ~ {' + '.join(covariates)}" if covariates else f"{x_col} ~ 1"
        model_x = smf.ols(formula=formula_x, data=sub).fit()
        sub["resid_x"] = model_x.resid

        formula_y = f"{y_col} ~ {' + '.join(covariates)}" if covariates else f"{y_col} ~ 1"
        model_y = smf.ols(formula=formula_y, data=sub).fit()
        sub["resid_y"] = model_y.resid

        r, p = pearsonr(sub["resid_x"], sub["resid_y"])
        return {
            "function": "partial_correlation",
            "x_col": x_col,
            "y_col": y_col,
            "covariates": covariates,
            "r_value": r,
            "p_value": p,
            "n": len(sub)
        }
    except Exception as e:
        return {"error": str(e)}


def partial_regression(df: pd.DataFrame, y_col: str, main_predictor: str, other_predictors=[]) -> dict:
    """
    Purpose: Unique effect of main_predictor on y_col after partialling out other_predictors.
    Use case: "Does budget predict sales once we account for brand rating, ad clicks, etc.?"
    """
    try:
        sub = df[[y_col, main_predictor] + other_predictors].dropna()
        formula = f"{y_col} ~ {' + '.join([main_predictor]+other_predictors)}"
        model = smf.ols(formula=formula, data=sub).fit()
        return {
            "function": "partial_regression",
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "r_squared": model.rsquared,
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def logistic_regression(df: pd.DataFrame, x_cols: list, y_col: str) -> dict:
    """
    Purpose: Binary classification model (logit).
    Use case: "Predict whether a dealership is top performer (1) vs. not (0)."
    """
    try:
        sub = df[[y_col] + x_cols].dropna()
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        model = smf.glm(formula=formula, data=sub,
                        family=sm.families.Binomial()).fit()
        return {
            "function": "logistic_regression",
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def anova_test(df: pd.DataFrame, group_col: str, metric_col: str) -> dict:
    """
    Purpose: One-way ANOVA across multiple groups for metric_col.
    Use case: "Do average monthly sales differ across brands?"
    """
    try:
        groups = []
        for g, subdf in df.dropna(subset=[group_col, metric_col]).groupby(group_col):
            groups.append(subdf[metric_col].values)
        if len(groups) < 2:
            return {"error": "Need at least two groups for ANOVA."}
        F, p = f_oneway(*groups)
        return {
            "function": "anova_test",
            "group_col": group_col,
            "metric_col": metric_col,
            "F_statistic": F,
            "p_value": p
        }
    except Exception as e:
        return {"error": str(e)}


def repeated_measures_anova(df: pd.DataFrame, subject_col: str, within_col: str, metric_col: str) -> dict:
    """
    Purpose: ANOVA for repeated measures (same subject over multiple conditions).
    Use case: "Do monthly sales differ across multiple campaigns for the same dealer?"
    """
    try:
        from statsmodels.stats.anova import AnovaRM
        sub = df[[subject_col, within_col, metric_col]].dropna()
        aov = AnovaRM(sub, depvar=metric_col, subject=subject_col,
                      within=[within_col]).fit()
        return {
            "function": "repeated_measures_anova",
            "anova_table": str(aov.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def posthoc_tukey_test(df: pd.DataFrame, group_col: str, metric_col: str) -> dict:
    """
    Purpose: After an ANOVA, do pairwise comparisons (Tukey's HSD).
    Use case: "Which specific brands differ in average sales?"
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        sub = df[[group_col, metric_col]].dropna()
        tukey = pairwise_tukeyhsd(
            endog=sub[metric_col], groups=sub[group_col], alpha=0.05)
        return {
            "function": "posthoc_tukey_test",
            "summary": str(tukey)
        }
    except Exception as e:
        return {"error": str(e)}


def ttest_groups(df: pd.DataFrame, group_col: str, metric_col: str, group_a, group_b) -> dict:
    """
    Purpose: T-test of means between two groups.
    Use case: "Are average monthly sales significantly different between brand A and brand B?"
    """
    try:
        sub = df[[group_col, metric_col]].dropna()
        a_vals = sub.loc[sub[group_col] == group_a, metric_col]
        b_vals = sub.loc[sub[group_col] == group_b, metric_col]
        if len(a_vals) < 2 or len(b_vals) < 2:
            return {"error": "Insufficient data in one or both groups for t-test."}
        stat, p = ttest_ind(a_vals, b_vals, equal_var=False)
        return {
            "function": "ttest_groups",
            "group_col": group_col,
            "metric_col": metric_col,
            "group_a": group_a,
            "group_b": group_b,
            "t_statistic": stat,
            "p_value": p
        }
    except Exception as e:
        return {"error": str(e)}


def chi_square_test(df: pd.DataFrame, cat_col1: str, cat_col2: str) -> dict:
    """
    Purpose: Chi-square of independence between two categorical variables.
    Use case: "Is top vs. bottom performer distribution the same across different states?"
    """
    try:
        sub = df[[cat_col1, cat_col2]].dropna()
        contingency = pd.crosstab(sub[cat_col1], sub[cat_col2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        return {
            "function": "chi_square_test",
            "cat_col1": cat_col1,
            "cat_col2": cat_col2,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "expected": expected.tolist()
        }
    except Exception as e:
        return {"error": str(e)}


def time_series_forecasting(df: pd.DataFrame, date_col: str, metric_col: str,
                            model='ARIMA', params=None) -> dict:
    """
    Purpose: Basic approach to forecasting future values with ARIMA or similar from statsmodels.
    Use case: "Predict next month's sales based on historical monthly data."
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        if params is None:
            params = {"order": (1, 1, 1)}  # default
        sub = df[[date_col, metric_col]].dropna().sort_values(by=date_col)
        y = sub[metric_col].values
        arima_model = ARIMA(y, order=params["order"])
        fitted = arima_model.fit()
        forecast = fitted.forecast(steps=1)
        return {
            "function": "time_series_forecasting",
            "model": model,
            "order": params["order"],
            "last_value": y[-1] if len(y) else None,
            "forecast_next": forecast.tolist()
        }
    except Exception as e:
        return {"error": str(e)}


def trend_seasonality_analysis(df: pd.DataFrame, date_col: str, metric_col: str) -> dict:
    """
    Purpose: Decompose a time series into trend, seasonal, residual components.
    Use case: "Does monthly sales show cyclical patterns (winter dips, summer spikes)?"
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        sub = df[[date_col, metric_col]].dropna().sort_values(by=date_col)
        sub[date_col] = pd.to_datetime(sub[date_col])  # ensure datetime
        sub.set_index(date_col, inplace=True)
        # assume monthly frequency
        decomposition = seasonal_decompose(
            sub[metric_col], model='additive', period=12)
        return {
            "function": "trend_seasonality_analysis",
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "resid": decomposition.resid.dropna().tolist()
        }
    except Exception as e:
        return {"error": str(e)}


def cohort_analysis(df: pd.DataFrame, cohort_col: str, date_col: str, metric_col: str) -> dict:
    """
    Purpose: Groups rows by 'cohort_col' (e.g., month joined) and tracks how metric_col evolves.
    Use case: "How does sales growth differ among dealers who joined in Jan vs. June?"
    """
    try:
        sub = df[[cohort_col, date_col, metric_col]].dropna()
        # minimal example: group by (cohort_col, date_col), mean of metric
        summary = sub.groupby([cohort_col, date_col])[
            metric_col].mean().reset_index()
        return {
            "function": "cohort_analysis",
            "aggregated_example": summary.to_dict(orient='records')
        }
    except Exception as e:
        return {"error": str(e)}


def factor_analysis_or_pca(df: pd.DataFrame, cols: list, n_components=2) -> dict:
    """
    Purpose: Dimensionality reduction or factor extraction using scikit-learn's PCA.
    Use case: "Summarize many variables (ad spend, sessions, brand rating, etc.) into fewer factors."
    """
    try:
        sub = df[cols].dropna()
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(sub.values)
        explained_var = pca.explained_variance_ratio_.tolist()
        return {
            "function": "factor_analysis_or_pca",
            "n_components": n_components,
            "explained_variance_ratio": explained_var,
            "components_head": components[:5].tolist()  # sample
        }
    except Exception as e:
        return {"error": str(e)}

###############################################################################
# C) 5 Specialized/Advanced Techniques
###############################################################################


def survival_analysis(df: pd.DataFrame, time_col: str, event_col: str, group_col=None,
                      method='kaplan_meier') -> dict:
    """
    Purpose: Basic survival analysis. For method='kaplan_meier' or 'cox'.
    Using lifelines library.
    Use case: "How long do dealers stay in a program before leaving?"
    """
    try:
        if method == 'kaplan_meier':
            kmf = KaplanMeierFitter()
            # require time_col is the duration, event_col is 1/0 for event
            sub = df[[time_col, event_col]].dropna()
            kmf.fit(durations=sub[time_col], event_observed=sub[event_col])
            survival_prob = kmf.survival_function_.reset_index().to_dict(orient='records')
            return {
                "function": "survival_analysis",
                "method": "kaplan_meier",
                "survival_prob": survival_prob
            }
        elif method == 'cox':
            cph = CoxPHFitter()
            # if group_col used, we might add it as a covariate
            columns = [time_col, event_col]
            if group_col:
                columns.append(group_col)
            sub = df[columns].dropna()
            # rename for lifelines: duration_col='T', event_col='E'
            sub = sub.rename(columns={time_col: 'T', event_col: 'E'})
            cph.fit(sub, duration_col='T', event_col='E')
            return {
                "function": "survival_analysis",
                "method": "cox",
                "summary": str(cph.summary)
            }
        else:
            return {"error": f"Unknown survival method: {method}"}
    except Exception as e:
        return {"error": str(e)}


def cross_correlation_analysis(df: pd.DataFrame, x_series_col: str, y_series_col: str, max_lag=12) -> dict:
    """
    Purpose: Shifts x_series_col by multiple lags, checks correlation with y_series_col.
    Use case: "Find best lag between budget spend, sales changes."
    """
    results = {}
    try:
        sub = df[[x_series_col, y_series_col]].dropna()
        x_vals = sub[x_series_col].values
        y_vals = sub[y_series_col].values
        for lag in range(1, max_lag+1):
            if lag >= len(x_vals):
                break
            x_shifted = x_vals[:-lag]
            y_aligned = y_vals[lag:]
            if len(x_shifted) < 2:
                continue
            r, p = pearsonr(x_shifted, y_aligned)
            results[lag] = {"r_value": r, "p_value": p, "n": len(x_shifted)}
        return {
            "function": "cross_correlation_analysis",
            "max_lag": max_lag,
            "results_by_lag": results
        }
    except Exception as e:
        return {"error": str(e)}


def propensity_score_matching(df: pd.DataFrame, treatment_col: str, covariates: list, outcome_col: str,
                              match_method='nearest') -> dict:
    """
    Purpose: Controls for confounding by matching "treated" vs. "control" dealers on covariates.
    Basic approach: logistic regression for treatment ~ covariates -> propensity -> nearest neighbor match.
    We'll do a minimal demonstration with scikit-learn's logistic for propensities, manual matching.
    """
    try:
        # 1) Fit logistic to predict treatment
        from sklearn.linear_model import LogisticRegression
        sub = df[[treatment_col, outcome_col] + covariates].dropna()
        # Ensure treatment_col is 0/1
        X = sub[covariates]
        y = sub[treatment_col]
        logit = LogisticRegression()
        logit.fit(X, y)
        sub['propensity'] = logit.predict_proba(X)[:, 1]

        # 2) naive nearest neighbor match
        treated = sub[sub[treatment_col] == 1].copy()
        control = sub[sub[treatment_col] == 0].copy()
        # For each treated, find control w/ nearest propensity
        matched_pairs = []
        used_control = set()
        for i, row in treated.iterrows():
            prop = row['propensity']
            ctrl_subset = control[~control.index.isin(used_control)]
            if len(ctrl_subset) == 0:
                break
            diffs = (ctrl_subset['propensity'] - prop).abs()
            best_idx = diffs.idxmin()
            matched_pairs.append((i, best_idx))
            used_control.add(best_idx)
        return {
            "function": "propensity_score_matching",
            "pairs_found": len(matched_pairs),
            "example_pairs": matched_pairs[:5]
        }
    except Exception as e:
        return {"error": str(e)}


def structural_equation_model(df: pd.DataFrame, model_spec: str) -> dict:
    """
    Purpose: More advanced version of mediation with multiple paths, latent factors, using semopy.
    model_spec is a string specifying the SEM in semopy's format.
    """
    try:
        sub = df.dropna()  # minimal
        model = semopy.Model(model_spec)
        model.fit(sub)
        return {
            "function": "structural_equation_model",
            "model_spec": model_spec,
            "summary": str(model.inspect())
        }
    except Exception as e:
        return {"error": str(e)}


def model_selection(df: pd.DataFrame, y_col: str, candidate_x_cols: list,
                    method='lasso', alpha=1.0) -> dict:
    """
    Purpose: Automated selection of best predictors from a large set, e.g. Lasso (regularization).
    Use case: "We have 20+ possible features— which combination best predicts sales?"
    """
    try:
        sub = df[[y_col] + candidate_x_cols].dropna()
        X = sub[candidate_x_cols].values
        y = sub[y_col].values
        if method == 'lasso':
            reg = Lasso(alpha=alpha)
            reg.fit(X, y)
            coefs = dict(zip(candidate_x_cols, reg.coef_))
            intercept = reg.intercept_
            return {
                "function": "model_selection",
                "method": "lasso",
                "alpha": alpha,
                "intercept": intercept,
                "coef": coefs
            }
        else:
            return {"error": f"Unsupported method {method}; only lasso implemented."}
    except Exception as e:
        return {"error": str(e)}

###############################################################################
# (New) Additional "Time-Change" Functions
###############################################################################


def time_series_change(df: pd.DataFrame, id_col: str, value_col: str,
                       start_month: int, end_month: int,
                       agg_method='sum', freq='monthly') -> dict:
    """
    Purpose: Aggregates a metric (e.g., budget, traffic, sales) from start_month to end_month
    per dealer (id_col), then does a difference or % change. Minimal example.
    """
    try:
        sub = df.dropna(subset=[id_col, value_col, 'MONTH'])
        mask = (sub['MONTH'] >= start_month) & (sub['MONTH'] <= end_month)
        filtered = sub[mask]
        agg_df = filtered.groupby(id_col)[value_col].agg(agg_method)
        return {
            "function": "time_series_change",
            "id_col": id_col,
            "value_col": value_col,
            "start_month": start_month,
            "end_month": end_month,
            "agg_method": agg_method,
            "aggregated_example": agg_df.to_dict()
        }
    except Exception as e:
        return {"error": str(e)}


def trend_analysis(df: pd.DataFrame, id_col: str, value_col: str,
                   start_month=1, end_month=9, method='linear') -> dict:
    """
    Purpose: Minimal approach to measure slope or % difference from start to end for each id_col.
    """
    try:
        sub = df.dropna(subset=[id_col, value_col, 'MONTH'])
        results = {}
        for the_id, sdf in sub.groupby(id_col):
            sdf = sdf[(sdf['MONTH'] >= start_month)
                      & (sdf['MONTH'] <= end_month)]
            sdf = sdf.sort_values(by='MONTH')
            if len(sdf) < 2:
                results[the_id] = {"trend_measure": None}
                continue
            if method == 'linear':
                model = smf.ols(f"{value_col} ~ MONTH", data=sdf).fit()
                slope = model.params.get('MONTH', None)
                results[the_id] = {"slope_linear": slope,
                                   "r_squared": model.rsquared}
            else:
                first_val = sdf[value_col].iloc[0]
                last_val = sdf[value_col].iloc[-1]
                if first_val == 0:
                    results[the_id] = {"percent_change": None}
                else:
                    pc = (last_val - first_val)/abs(first_val)*100
                    results[the_id] = {"percent_change": pc}
        return {
            "function": "trend_analysis",
            "method": method,
            "results_by_id": results
        }
    except Exception as e:
        return {"error": str(e)}


def pre_post_comparison(df: pd.DataFrame, id_col: str, metric_col: str,
                        pre_month_range: tuple, post_month_range: tuple) -> dict:
    """
    Purpose: Splits months into 'pre' vs 'post' intervals, aggregates, and compares difference or ratio.
    Use case: "After poor mystery shops in certain months, did sales drop in subsequent months?"
    """
    try:
        sub = df.dropna(subset=[id_col, metric_col, 'MONTH'])
        (pre_start, pre_end) = pre_month_range
        (post_start, post_end) = post_month_range
        results = {}
        for the_id, sdf in sub.groupby(id_col):
            pre_df = sdf[(sdf['MONTH'] >= pre_start)
                         & (sdf['MONTH'] <= pre_end)]
            post_df = sdf[(sdf['MONTH'] >= post_start)
                          & (sdf['MONTH'] <= post_end)]
            pre_val = pre_df[metric_col].mean() if len(pre_df) else None
            post_val = post_df[metric_col].mean() if len(post_df) else None
            if pre_val is None or post_val is None:
                results[the_id] = {"pre_val": pre_val,
                                   "post_val": post_val, "difference": None}
            else:
                results[the_id] = {
                    "pre_val": pre_val,
                    "post_val": post_val,
                    "difference": post_val - pre_val
                }
        return {
            "function": "pre_post_comparison",
            "metric_col": metric_col,
            "pre_range": pre_month_range,
            "post_range": post_month_range,
            "results_by_id": results
        }
    except Exception as e:
        return {"error": str(e)}


def ranking_and_correlation(df: pd.DataFrame, rank_metric: str, correlation_metric: str,
                            group_col=None, top_n=10) -> dict:
    """
    Purpose: Ranks dealers by rank_metric, takes top_n, then correlates with correlation_metric.
    """
    try:
        sub = df.dropna(subset=[rank_metric, correlation_metric])
        sub = sub.sort_values(by=rank_metric, ascending=False)
        top_df = sub.head(top_n)
        if len(top_df) < 2:
            return {"error": f"Not enough data in top {top_n} subset."}
        r, p = pearsonr(top_df[rank_metric], top_df[correlation_metric])
        return {
            "function": "ranking_and_correlation",
            "rank_metric": rank_metric,
            "correlation_metric": correlation_metric,
            "top_n": top_n,
            "r_value": r,
            "p_value": p
        }
    except Exception as e:
        return {"error": str(e)}


def categorical_trend_analysis(df: pd.DataFrame, id_col: str, cat_col: str,
                               value_col: str, months_range: tuple,
                               measure='percent_of_total') -> dict:
    """
    Purpose: Break down value_col by cat_col over time, compute share each category occupies.
    """
    try:
        sub = df.dropna(subset=[id_col, cat_col, value_col, 'MONTH'])
        (m_start, m_end) = months_range
        sub = sub[(sub['MONTH'] >= m_start) & (sub['MONTH'] <= m_end)]
        cat_sums = sub.groupby(cat_col)[value_col].sum()
        total = cat_sums.sum()
        if total == 0:
            return {"error": "Total is zero, cannot compute percentages."}
        pct = (cat_sums/total*100).to_dict()
        return {
            "function": "categorical_trend_analysis",
            "cat_col": cat_col,
            "value_col": value_col,
            "months_range": months_range,
            "percentages": pct
        }
    except Exception as e:
        return {"error": str(e)}

###############################################################################
# cluster_analysis
###############################################################################


def cluster_analysis(df: pd.DataFrame, x_cols: list, n_clusters=5, method='kmeans') -> dict:
    """
    Purpose: Unsupervised grouping of dealerships by multiple numeric variables using scikit-learn KMeans.
    """
    try:
        sub = df[x_cols].dropna()
        if method.lower() == 'kmeans':
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(sub.values)
            centers = km.cluster_centers_
            return {
                "function": "cluster_analysis",
                "method": "kmeans",
                "n_clusters": n_clusters,
                "cluster_labels_head": labels[:10].tolist(),  # sample
                "cluster_centers": centers.tolist()
            }
        else:
            return {"error": f"Unsupported clustering method: {method}"}
    except Exception as e:
        return {"error": str(e)}

###############################################################################
# (Newest) Two Advanced Models
###############################################################################


def mixed_effects_model(df: pd.DataFrame, y_col: str, fixed_effects: list, group_col: str,
                        random_slopes=False, random_effect_col=None, family='gaussian') -> dict:
    """
    Purpose: Handle hierarchical or repeated-measures data with random intercepts.
    If family=='gaussian', statsmodels MixedLM works well.
    For random slopes, partial approach with 'exog_re' in MixedLM if needed.
    """
    try:
        sub = df[[y_col, group_col] + fixed_effects].dropna()
        formula = f"{y_col} ~ {' + '.join(fixed_effects)}"
        md = smf.mixedlm(formula, sub, groups=sub[group_col])
        mdf = md.fit()
        return {
            "function": "mixed_effects_model",
            "formula": formula,
            "random_group": group_col,
            "params": dict(mdf.params),
            "summary": str(mdf.summary())
        }
    except Exception as e:
        return {"error": str(e)}


def generalized_linear_model(df: pd.DataFrame, y_col: str, x_cols: list, family='poisson') -> dict:
    """
    Purpose: Goes beyond OLS for non-normal outcomes (counts, fractions).
    e.g. 'poisson' -> Poisson, 'binomial' -> logistic, 'gamma', etc.
    """
    try:
        sub = df[[y_col] + x_cols].dropna()
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        fam_map = {
            'poisson': sm.families.Poisson(),
            'binomial': sm.families.Binomial(),
            'gaussian': sm.families.Gaussian(),
            'gamma': sm.families.Gamma()
        }
        chosen_family = fam_map.get(family.lower(), sm.families.Poisson())
        model = smf.glm(formula=formula, data=sub, family=chosen_family).fit()
        return {
            "function": "generalized_linear_model",
            "family": family,
            "formula": formula,
            "params": dict(model.params),
            "pvalues": dict(model.pvalues),
            "n": int(model.nobs),
            "summary": str(model.summary())
        }
    except Exception as e:
        return {"error": str(e)}
