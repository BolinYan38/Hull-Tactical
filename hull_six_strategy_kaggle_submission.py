# %% [markdown]
# # Hull Tactical：六策略 OT Late Submission
#
# 这个 Notebook 将三种规则策略（趋势、均值回归、防守）与三种模型策略
#（LightGBM、CatBoost、ElasticNet）组合，并使用历史相似市场状态进行逐日配置。
#
# 提交时请把 Internet 设为 Off，并将本 Notebook 关联到
# `Hull Tactical - Market Prediction` 竞赛数据。

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
COVERAGE_START = 1006
MIN_COVERAGE = 0.80
WINDOW = 40
HORIZON = 20
K_NEIGHBORS = 20
PRESELECT_NEIGHBORS = 100
DISTANCE_TEMPERATURE = 1.0
SCORE_TEMPERATURE = 0.01
COST_RATE = 0.0001
CONT_GROUPS = ["M", "V", "S", "I", "E", "P"]
STRATEGY_NAMES = [
    "trend",
    "mean_reversion",
    "defensive",
    "lightgbm",
    "catboost",
    "elasticnet",
]


def locate_competition_path():
    """兼容 Kaggle 的不同挂载目录和本地检查路径。"""
    candidates = [
        Path("/kaggle/input/hull-tactical-market-prediction"),
        Path("archive"),
        Path("project/Hull Tactical - Market Prediction/archive"),
    ]
    for path in candidates:
        if (path / "train.csv").exists():
            return path

    # Kaggle 偶尔会调整 Input 的实际挂载层级，因此不再只依赖固定目录名。
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_file in sorted(kaggle_input.rglob("train.csv")):
            data_path = train_file.parent
            if (data_path / "test.csv").exists():
                print("Auto-detected competition files:", data_path)
                return data_path

    raise FileNotFoundError("没有找到 Hull Tactical 竞赛数据目录")


COMPETITION_PATH = locate_competition_path()
TRAIN_DF = pd.read_csv(COMPETITION_PATH / "train.csv")
print("Competition data:", COMPETITION_PATH)
print("Training rows:", len(TRAIN_DF))

# %%
def prediction_to_position(prediction, reference_prediction):
    """沿用研究 Notebook 的预测值到仓位映射。"""
    reference_prediction = np.asarray(reference_prediction, dtype=float)
    sigma = float(np.std(reference_prediction, ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.ones_like(np.asarray(prediction, dtype=float))
    z_score = (
        np.asarray(prediction, dtype=float)
        - float(np.mean(reference_prediction))
    ) / sigma
    return np.clip(1.0 + 0.5 * z_score, 0.5, 2.0)


def backtest_positions(positions, market_returns, risk_free_returns):
    """生成各策略考虑持仓漂移和交易成本后的每日净收益。"""
    market_returns = market_returns.loc[positions.index]
    risk_free_returns = risk_free_returns.loc[positions.index]
    gross = positions.mul(market_returns, axis=0) + (
        1.0 - positions
    ).mul(risk_free_returns, axis=0)
    before_trade = (
        positions.shift(1).mul(1.0 + market_returns.shift(1), axis=0)
        .div(1.0 + gross.shift(1))
    )
    before_trade.iloc[0] = 0.0
    change = positions - before_trade
    turnover = change.abs() / (
        1.0 + COST_RATE * positions * np.sign(change)
    )
    net_return = (1.0 - COST_RATE * turnover) * (1.0 + gross) - 1.0
    return net_return


def make_rule_positions(market_returns):
    """只使用决策日前已经实现的收益生成三种规则策略。"""
    past_returns = market_returns.shift(1)
    past_log_returns = np.log1p(past_returns.clip(lower=-0.999999))

    momentum_60 = np.expm1(
        past_log_returns.rolling(60, min_periods=60).sum()
    )
    log_return_5 = past_log_returns.rolling(5, min_periods=5).sum()
    log_volatility_20 = past_log_returns.rolling(
        20, min_periods=20
    ).std()
    z_score = log_return_5 / (
        log_volatility_20.clip(lower=1e-8) * np.sqrt(5.0)
    )
    annual_volatility_20 = (
        past_returns.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    )

    return pd.DataFrame(
        {
            "trend": (momentum_60 > 0).astype(float).where(
                momentum_60.notna()
            ),
            "mean_reversion": (0.5 - z_score / 2.0).clip(0.0, 1.0),
            "defensive": (
                0.05 / annual_volatility_20.clip(lower=1e-8)
            ).clip(0.0, 0.3),
        },
        index=market_returns.index,
    )


def wasserstein_distance_2(window_a, window_b):
    cost = cdist(window_a, window_b, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def previous_values(series, date_id, count):
    values = series.loc[series.index < date_id].tail(count)
    return values.astype(float)

# %%
class SixStrategyOTEngine:
    def __init__(self, train_df):
        data = train_df.sort_values("date_id").copy()
        data = data.set_index("date_id", drop=False)
        self.data = data
        self.rng = np.random.default_rng(SEED)

        self._prepare_columns()
        self._prepare_models_and_strategy_history()
        self._prepare_market_states()
        self._prepare_online_state()

    def _prepare_columns(self):
        data = self.data
        self.group_columns = {
            group: [
                col
                for col in data.columns
                if col.startswith(group) and col[1:].isdigit()
            ]
            for group in CONT_GROUPS + ["D"]
        }

        dev = data.iloc[COVERAGE_START:].copy()
        selection_end = max(1, int(len(dev) * 0.60))
        selection_data = dev.iloc[:selection_end]

        self.selected_continuous = {}
        for group in CONT_GROUPS:
            coverage = selection_data[
                self.group_columns[group]
            ].notna().mean()
            self.selected_continuous[group] = coverage[
                coverage >= MIN_COVERAGE
            ].index.tolist()

        d_columns = self.group_columns["D"].copy()
        clean_d_columns = []
        for col in d_columns:
            if not any(
                selection_data[col].equals(selection_data[kept])
                for kept in clean_d_columns
            ):
                clean_d_columns.append(col)
        self.d_columns = clean_d_columns
        self.model_columns = [
            col
            for group in CONT_GROUPS
            for col in self.selected_continuous[group]
        ] + self.d_columns

    def _make_models(self):
        lgb_model = LGBMRegressor(
            objective="regression",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            reg_lambda=2,
            reg_alpha=0.0,
            num_leaves=7,
            n_estimators=300,
            min_child_samples=80,
            max_depth=5,
            learning_rate=0.05,
            colsample_bytree=0.6,
        )
        cat_model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=SEED,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            has_time=True,
            iterations=300,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=10,
            random_strength=1.0,
        )
        elastic_model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", keep_empty_features=True
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=0.010,
                        l1_ratio=0.75,
                        max_iter=50000,
                        tol=1e-4,
                        selection="random",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        return {
            "lightgbm": lgb_model,
            "catboost": cat_model,
            "elasticnet": elastic_model,
        }

    def _prepare_models_and_strategy_history(self):
        data = self.data
        raw_features = data[self.model_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        shifted_features = raw_features.shift(1)

        market_log = np.log1p(
            data["forward_returns"].clip(lower=-0.999999)
        )
        risk_free_log = np.log1p(
            data["risk_free_rate"].clip(lower=-0.999999)
        )
        daily_log_excess = market_log - risk_free_log
        target = (
            daily_log_excess.rolling(HORIZON, min_periods=HORIZON)
            .sum()
            .shift(-(HORIZON - 1))
        )

        dev_index = data.index[COVERAGE_START:]
        valid = target.loc[dev_index].notna() & shifted_features.loc[
            dev_index
        ].notna().any(axis=1)
        model_index = dev_index[valid.to_numpy()]
        self.X_model = shifted_features.loc[model_index].copy()
        self.y_model = target.loc[model_index].copy()

        templates = self._make_models()
        splitter = TimeSeriesSplit(n_splits=5, gap=HORIZON - 1)
        oof_positions = pd.DataFrame(
            np.nan,
            index=self.X_model.index,
            columns=["lightgbm", "catboost", "elasticnet"],
        )

        split_list = list(splitter.split(self.X_model))
        for name, template in templates.items():
            for fit_pos, valid_pos in split_list:
                fold_model = clone(template)
                X_fit = self.X_model.iloc[fit_pos]
                y_fit = self.y_model.iloc[fit_pos]
                X_valid = self.X_model.iloc[valid_pos]
                fold_model.fit(X_fit, y_fit)
                reference = fold_model.predict(X_fit)
                prediction = fold_model.predict(X_valid)
                oof_positions.loc[
                    X_valid.index, name
                ] = prediction_to_position(prediction, reference)

        self.models = templates
        self.reference_predictions = {}
        for name, model in self.models.items():
            model.fit(self.X_model, self.y_model)
            self.reference_predictions[name] = model.predict(self.X_model)

        market_returns = data["forward_returns"].astype(float)
        risk_free_returns = data["risk_free_rate"].astype(float)
        rule_positions = make_rule_positions(market_returns)

        common_index = oof_positions.dropna().index
        history_positions = pd.concat(
            [
                rule_positions.loc[
                    common_index,
                    ["trend", "mean_reversion", "defensive"],
                ],
                oof_positions.loc[
                    common_index,
                    ["lightgbm", "catboost", "elasticnet"],
                ],
            ],
            axis=1,
        ).dropna()
        history_positions = history_positions[STRATEGY_NAMES]

        net_return = backtest_positions(
            history_positions, market_returns, risk_free_returns
        )
        rolling_log = np.log1p(net_return.clip(lower=-0.999999)).rolling(
            HORIZON, min_periods=HORIZON
        ).sum()
        self.strategy_forward_return = np.expm1(
            rolling_log.shift(-(HORIZON - 1))
        ).dropna()

    def _prepare_market_states(self):
        data = self.data
        dev = data.iloc[COVERAGE_START:]
        self.state_preprocessors = {}
        transformed_groups = {}

        for group in CONT_GROUPS:
            columns = self.selected_continuous[group]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            pca = PCA(n_components=0.80)

            dev_imputed = imputer.fit_transform(dev[columns])
            dev_scaled = scaler.fit_transform(dev_imputed)
            pca.fit(dev_scaled)

            all_values = pca.transform(
                scaler.transform(imputer.transform(data[columns]))
            )
            transformed_groups[group] = all_values
            self.state_preprocessors[group] = (
                columns,
                imputer,
                scaler,
                pca,
            )

        d_binary = data[self.d_columns].fillna(0).ne(0).astype(float)
        d_frequency = d_binary.rolling(
            WINDOW, min_periods=WINDOW
        ).mean().shift(1)

        date_values = data.index.to_numpy()
        date_to_pos = {int(date): pos for pos, date in enumerate(date_values)}
        valid_dates = [
            int(date)
            for date in self.strategy_forward_return.index
            if int(date) in date_to_pos and date_to_pos[int(date)] >= WINDOW
        ]
        self.history_dates = np.asarray(valid_dates, dtype=int)
        self.history_returns = self.strategy_forward_return.loc[
            self.history_dates, STRATEGY_NAMES
        ].to_numpy(dtype=float)

        self.history_windows = {}
        self.history_means = {}
        self.history_stds = {}
        for group in CONT_GROUPS:
            values = transformed_groups[group]
            windows = np.stack(
                [
                    values[date_to_pos[date] - WINDOW : date_to_pos[date]]
                    for date in self.history_dates
                ]
            )
            self.history_windows[group] = windows
            self.history_means[group] = windows.mean(axis=1)
            self.history_stds[group] = windows.std(axis=1, ddof=0)

        self.history_d_frequency = d_frequency.loc[
            self.history_dates
        ].to_numpy(dtype=float)
        self.distance_scales = self._calibrate_distance_scales()

        self.transformed_feature_store = {
            group: {
                int(date): transformed_groups[group][pos]
                for pos, date in enumerate(date_values)
            }
            for group in CONT_GROUPS
        }
        self.d_binary_store = {
            int(date): d_binary.iloc[pos].to_numpy(dtype=float)
            for pos, date in enumerate(date_values)
        }

    def _calibrate_distance_scales(self):
        pair_count = min(400, max(1, len(self.history_dates) // 2))
        distances = {group: [] for group in CONT_GROUPS + ["D"]}
        for _ in range(pair_count):
            a, b = self.rng.choice(
                len(self.history_dates), size=2, replace=False
            )
            for group in CONT_GROUPS:
                distances[group].append(
                    wasserstein_distance_2(
                        self.history_windows[group][a],
                        self.history_windows[group][b],
                    )
                )
            distances["D"].append(
                np.linalg.norm(
                    self.history_d_frequency[a]
                    - self.history_d_frequency[b]
                )
            )
        scales = {}
        for group, values in distances.items():
            scale = float(np.nanmedian(values))
            scales[group] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        return scales

    def _prepare_online_state(self):
        data = self.data
        self.raw_feature_store = {
            int(date): row[self.model_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            for date, row in data.iterrows()
        }
        self.realized_market_returns = data["forward_returns"].astype(float).copy()

    def _transform_current_features(self, row):
        transformed = {}
        for group, objects in self.state_preprocessors.items():
            columns, imputer, scaler, pca = objects
            frame = pd.DataFrame(
                [[row.get(col, np.nan) for col in columns]], columns=columns
            )
            transformed[group] = pca.transform(
                scaler.transform(imputer.transform(frame))
            )[0]
        return transformed

    def _current_market_state(self, date_id):
        prior_dates = sorted(
            date
            for date in self.d_binary_store
            if date < date_id
        )[-WINDOW:]
        if len(prior_dates) < WINDOW:
            raise ValueError("当前日期以前没有完整的40日市场窗口")

        windows = {
            group: np.stack(
                [self.transformed_feature_store[group][date] for date in prior_dates]
            )
            for group in CONT_GROUPS
        }
        d_frequency = np.stack(
            [self.d_binary_store[date] for date in prior_dates]
        ).mean(axis=0)
        return windows, d_frequency

    def _nearest_history(self, current_windows, current_d_frequency):
        group_weight = 1.0 / 7.0
        lower_bound = (
            np.linalg.norm(
                self.history_d_frequency - current_d_frequency, axis=1
            )
            / self.distance_scales["D"]
            * group_weight
        )

        for group in CONT_GROUPS:
            mean_diff = self.history_means[group] - current_windows[group].mean(
                axis=0
            )
            std_diff = self.history_stds[group] - current_windows[group].std(
                axis=0, ddof=0
            )
            group_bound = np.sqrt(
                (mean_diff**2 + std_diff**2).sum(axis=1)
            )
            lower_bound += (
                group_bound / self.distance_scales[group] * group_weight
            )

        preselect_count = min(PRESELECT_NEIGHBORS, len(lower_bound))
        candidate_pos = np.argpartition(
            lower_bound, preselect_count - 1
        )[:preselect_count]

        exact_distance = np.zeros(preselect_count, dtype=float)
        for output_pos, history_pos in enumerate(candidate_pos):
            distance = (
                np.linalg.norm(
                    self.history_d_frequency[history_pos]
                    - current_d_frequency
                )
                / self.distance_scales["D"]
                * group_weight
            )
            for group in CONT_GROUPS:
                distance += (
                    wasserstein_distance_2(
                        self.history_windows[group][history_pos],
                        current_windows[group],
                    )
                    / self.distance_scales[group]
                    * group_weight
                )
            exact_distance[output_pos] = distance

        neighbor_count = min(K_NEIGHBORS, len(candidate_pos))
        nearest_local = np.argsort(exact_distance)[:neighbor_count]
        return candidate_pos[nearest_local], exact_distance[nearest_local]

    def _rule_positions_for_day(self, date_id):
        returns = previous_values(
            self.realized_market_returns.sort_index(), date_id, 60
        )
        if len(returns) < 60:
            return np.array([1.0, 0.5, 0.3], dtype=float)

        log_returns = np.log1p(returns.clip(lower=-0.999999))
        momentum = float(np.expm1(log_returns.iloc[-60:].sum()))
        log_return_5 = float(log_returns.iloc[-5:].sum())
        log_volatility_20 = float(log_returns.iloc[-20:].std(ddof=1))
        z_score = log_return_5 / max(
            log_volatility_20 * np.sqrt(5.0), 1e-8
        )
        annual_volatility = float(
            returns.iloc[-20:].std(ddof=1) * np.sqrt(252.0)
        )
        return np.array(
            [
                float(momentum > 0),
                float(np.clip(0.5 - z_score / 2.0, 0.0, 1.0)),
                float(np.clip(0.05 / max(annual_volatility, 1e-8), 0.0, 0.3)),
            ]
        )

    def _model_positions_for_day(self, date_id):
        prior_dates = [date for date in self.raw_feature_store if date < date_id]
        if not prior_dates:
            return np.ones(3, dtype=float)
        prior_row = self.raw_feature_store[max(prior_dates)]
        X_current = pd.DataFrame([prior_row], columns=self.model_columns)
        output = []
        for name in ["lightgbm", "catboost", "elasticnet"]:
            prediction = self.models[name].predict(X_current)
            position = prediction_to_position(
                prediction, self.reference_predictions[name]
            )[0]
            output.append(float(position))
        return np.asarray(output)

    def predict_one(self, test_row):
        row = test_row.copy()
        date_id = int(row["date_id"])

        lagged_return = row.get("lagged_forward_returns", np.nan)
        if pd.notna(lagged_return):
            self.realized_market_returns.loc[date_id - 1] = float(lagged_return)

        current_windows, current_d_frequency = self._current_market_state(date_id)
        neighbor_pos, neighbor_distances = self._nearest_history(
            current_windows, current_d_frequency
        )
        neighbor_weights = softmax(
            -neighbor_distances / DISTANCE_TEMPERATURE
        )
        strategy_scores = (
            self.history_returns[neighbor_pos] * neighbor_weights[:, None]
        ).sum(axis=0)

        raw_weights = softmax(strategy_scores / SCORE_TEMPERATURE)
        ranked_pos = np.argsort(-raw_weights, kind="stable")
        ranked_weights = raw_weights[ranked_pos]
        selected_count = int(
            np.searchsorted(
                np.cumsum(ranked_weights), 0.50, side="right"
            )
            + 1
        )
        selected_pos = ranked_pos[:selected_count]
        final_weights = np.zeros(len(STRATEGY_NAMES), dtype=float)
        final_weights[selected_pos] = raw_weights[selected_pos]
        final_weights /= final_weights.sum()

        current_positions = np.concatenate(
            [
                self._rule_positions_for_day(date_id),
                self._model_positions_for_day(date_id),
            ]
        )
        allocation = float(np.dot(final_weights, current_positions))
        allocation = float(np.clip(allocation, 0.0, 2.0))

        transformed = self._transform_current_features(row)
        for group in CONT_GROUPS:
            self.transformed_feature_store[group][date_id] = transformed[group]
        self.d_binary_store[date_id] = np.asarray(
            [float(pd.notna(row.get(col)) and row.get(col) != 0) for col in self.d_columns]
        )
        self.raw_feature_store[date_id] = pd.Series(
            {
                col: (
                    np.nan
                    if not np.isfinite(row.get(col, np.nan))
                    else float(row.get(col))
                )
                for col in self.model_columns
            }
        )
        return allocation

# %% [markdown]
# 模型与 OOF 历史会在第一次调用 `predict` 时建立。这样可以确保评估服务器在
# Kaggle 规定的时间内先启动；首个预测批次允许执行初始化工作。

# %%
ENGINE = None


def predict(test):
    global ENGINE
    if ENGINE is None:
        print("Building six-strategy OT engine ...", flush=True)
        ENGINE = SixStrategyOTEngine(TRAIN_DF)
        print("Engine is ready.", flush=True)

    if hasattr(test, "to_pandas"):
        test_frame = test.to_pandas()
    elif isinstance(test, pd.DataFrame):
        test_frame = test
    else:
        test_frame = pd.DataFrame(test)

    if len(test_frame) != 1:
        raise ValueError(f"Expected one test row, received {len(test_frame)}")
    return ENGINE.predict_one(test_frame.iloc[0])

# %%
try:
    import kaggle_evaluation.default_inference_server
except ModuleNotFoundError:
    # 只用于下载到本地后的快速检查；Kaggle 环境会进入下面的 else。
    if __name__ == "__main__":
        mock_test = pd.read_csv(COMPETITION_PATH / "test.csv")
        local_allocations = [
            predict(mock_test.iloc[[row_number]])
            for row_number in range(len(mock_test))
        ]
        print(pd.DataFrame({
            "date_id": mock_test["date_id"],
            "allocation": local_allocations,
        }))
else:
    inference_server = (
        kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
    )
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (str(COMPETITION_PATH) + "/",)
        )
