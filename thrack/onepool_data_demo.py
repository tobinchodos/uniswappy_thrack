# %%
import pandas as pd
import numpy as np
from uniswappy import ERC20, UniV3Utils
from pool import UniswapPool
from env import DataDrivenEnv as Env

# from utils import plot
from copy import deepcopy
from decimal import Decimal

pd.options.display.min_rows = 100
pd.options.display.max_columns = 100
pd.options.plotting.backend = "plotly"


def row_to_action_dict(row, numeric_columns):

    in_d = row.to_dict()

    for col in numeric_columns:
        if not pd.isna(in_d[col]):
            if col == "pool_price":
                in_d[col] = float(in_d[col])
            else:
                if col in [
                    "amount_0",
                    "amount_1",
                ]:
                    in_d[col] = int(in_d[col])
                else:
                    in_d[col] = int(in_d[col])

    in_d["lwr_price"] = (
        in_d["lwr_tick"] if pd.isna(in_d["lwr_tick"]) else 1.0001 ** in_d["lwr_tick"]
    )

    in_d["upr_price"] = (
        in_d["upr_tick"] if pd.isna(in_d["upr_tick"]) else 1.0001 ** in_d["upr_tick"]
    )

    return in_d


# %%
#
if __name__ == "__main__":
    X_DEC = 18
    Y_DEC = 18

    orig_event_df = pd.read_csv(
        "/Users/harrison/Documents/GitHub/uniswappy_thrack/thrack/resources/Uniswap_V3_Pool_Mint_Burn_and_Swap_Events_Based_on_Pool_NFPM_Contracts.csv",
        dtype={
            "event": str,
            "evt_block_time": str,
            "pool_liquidity": str,
            "sqrtPriceX96": str,
            "liquidity_provider": str,
            "amount_lp_token": str,
            "tickLower": str,
            "tickUpper": str,
            "trader": str,
            "amount0": str,
            "amount1": str,
            "tick": str,
            "source": str,
        },
    )

    new_names = {
        "event": "event",
        "source": "advanced_user",
        "evt_block_time": "date",
        "pool_liquidity": "liquidity_at_tick",  # only for post analysis
        "sqrtPriceX96": "sqrt_price_X96",
        "liquidity_provider": "lp_id",
        "amount_lp_token": "amount_lp_token",  #
        "tickLower": "lwr_tick",
        "tickUpper": "upr_tick",
        "trader": "trader_id",
        "amount0": "amount_0",
        "amount1": "amount_1",
        "tick": "tick",
    }
    event_df = orig_event_df.rename(columns=new_names).filter(items=new_names.values())

    event_df["pool_id"] = "uni_0"
    event_df["token_0_name"] = "X"
    event_df["token_1_name"] = "Y"
    event_df["token_0_DEC"] = 18
    event_df["token_1_DEC"] = 18
    event_df["advanced_user"] = event_df["advanced_user"].apply(
        lambda x: x if pd.isna(x) else (x == "pool")
    )
    event_df["fee"] = Decimal(UniV3Utils.FeeAmount.MEDIUM)

    numeric_columns = [
        "liquidity_at_tick",
        "sqrt_price_X96",
        "amount_lp_token",
        "lwr_tick",
        "upr_tick",
        "amount_0",
        "amount_1",
        "tick",
        "fee",
    ]

    for col in numeric_columns:
        event_df[col] = event_df[col].apply(Decimal)

    # add fee, pool_price, total_liquidity

    # fix starting sqrt_price_X96 from internet value
    event_df.loc[0, "sqrt_price_X96"] = Decimal("1354707084081889255546596745")

    event_df["pool_price"] = event_df["sqrt_price_X96"].apply(
        lambda x: (x / 2**96) ** 2
    )
    event_df.loc[
        event_df.event.apply(lambda x: x in ["mint", "burn"]), "total_liquidity"
    ] = event_df.loc[
        event_df.event.apply(lambda x: x in ["mint", "burn"]), "amount_lp_token"
    ].cumsum()
    event_df.total_liquidity = event_df.total_liquidity.ffill()
    numeric_columns.extend(["fee", "pool_price", "total_liquidity"])

    env_config = {
        "tokens": {
            "X": (ERC20, dict(decimal=X_DEC)),
            "Y": (ERC20, dict(decimal=Y_DEC)),
        },
        "pools": {},  # not needed when we have data
    }

    env = Env(env_config)
    env.reset()
    for i, row in event_df.iterrows():
        if i > 5000:
            break
        action_dict = row_to_action_dict(row, numeric_columns)
        env.step(action_dict=action_dict)

    if True:
        env.collect_metrics()
        df = pd.DataFrame(env.metrics)
        df.loc[0, "true_pool.liquidity_at_tick"] = df.loc[
            0, "true_pool.total_liquidity"
        ]

        # convert times to times
        for col in df.columns:
            if "date" in col:
                df[col] = pd.to_datetime(df[col])

        # add lifetimes of positions
        df["lifetime"] = df.date.iloc[-1] - df.start_date
        df.loc[df.is_burned, "lifetime"] = (
            df.loc[df.is_burned, "burn_date"] - df.loc[df.is_burned, "start_date"]
        )
        df["lifetime_h"] = df["lifetime"].apply(lambda x: x.total_seconds() / 3600)

        # terminal time
        dg = df[df.date == df.date.iloc[-1]]

        # time and liq vs time
        dg.plot(
            x="total_rewards.time_and_liq",
            y="total_rewards.volume",
            kind="scatter",
            trendline="ols",
        )

        for col in df.filter(regex="^true_|^synth").columns:
            df[col] = df[col].astype(float)
        fig = (
            df.groupby("step")
            .first()
            .filter(regex="^true_|^synth|event")
            .plot(
                y=df.filter(regex="^true_|^synth").columns,
                kind="scatter",
                symbol="event",
            )
        )
        fig.layout.template = "plotly_dark"
        fig.show(renderer="browser")

        df["fees_earned_0/exp_fees_0"] = (
            df["fees_earned_0"].astype(float) / df["exp_fees_0"] / 10**18
        )
        df["fees_earned_1/exp_fees_1"] = (
            df["fees_earned_1"].astype(float) / df["exp_fees_1"] / 10**18
        )
        df[
            ~np.isclose(df["fees_earned_0/exp_fees_0"].fillna(1), 1)
            | ~np.isclose(df["fees_earned_1/exp_fees_1"].fillna(1), 1)
        ]

        p = env.pools.get("uni_0")
        for key, hp in p.positions.items():

            _hash = hash(key)
            if _hash in p._pool.positions:
                _pos = p._pool.positions[_hash]
                if not np.isclose(
                    (hp.fees_earned_0) / 10**18, hp._exp_fees_0
                ) or not np.isclose((hp.fees_earned_1) / 10**18, hp._exp_fees_1):
                    print(f"pos.liquidity={_pos.liquidity}")
                    print(f"hp.liquidity={hp.liquidity}")

                    print(f"pos.tokensOwed0={_pos.tokensOwed0}")
                    print(f"hp.fees_earned_0={hp.fees_earned_0}")
                    print(f"hp.exp_fees_0={hp._exp_fees_0}")

                    print(f"pos.tokensOwed1={_pos.tokensOwed1}")
                    print(f"hp.fees_earned_1={hp.fees_earned_1}")
                    print(f"hp.exp_fees_1={hp._exp_fees_1}")
                    print(hp)
                    print("\n")
# swap(recipient, zeroForOne, amount, limit)

# #
# zeroForOne = True <--> token_in=0, token_out=1
# zeroForOne = False <--> token_in=1, token_out=0
# amount_0 > 0 <--> token_in = 0 <-> zeroForOne = True
# amount_1 > 0 <--> token_in = 1 <--> zeroForOne = False
# #
# zeroForOne = (amount_0>0)
# #
# amount > 0 --> exact input (you get whatever you get )
# amount < 0 --> exact output (your amount in will be variable)
# #
# %%
# add winner
# # %%
# dg["winner"] = dg[
#     [
#         "total_rewards.time_and_liq",
#         "total_rewards.fee",
#         "total_rewards.volume",
#         "total_rewards.time",
#     ]
# ].idxmax(axis=1)
# dg["winner"] = dg["winner"].str.replace("total_rewards.", "", regex=False)

# dg.filter(regex="^total_rewards|winner").groupby("winner").mean().loc[
#     ["time", "time_and_liq", "volume", "fee"],
# ]
# %%
# df = dg.copy()
# for pair0, pair1 in [
#     ("fee", "volume"),
#     ("time", "volume"),
#     ("time", "fee"),
#     ("time_and_liq", "volume"),
# ]:
#     col1, col2 = f"total_rewards.{pair0}", f"total_rewards.{pair1}"
#     df_filtered = df.filter(items=[col1, col2, "lifetime_h", "liquidity_added"]).copy()
#     for col in df_filtered.columns:
#         q_low = df[col].quantile(0.05)
#         q_hi = df[col].quantile(0.95)
#         df_filtered = df_filtered[
#             (df_filtered[col] < q_hi) & (df_filtered[col] > q_low)
#         ]
#     print(df_filtered.corr())
#     fig = df_filtered.plot(
#         x=col1,
#         y=col2,
#         kind="scatter",
#         trendline="ols",
#         marginal_x="violin",
#         marginal_y="violin",
#         color="liquidity_added",
#     )
#     fig.show()

# df = dg.drop(
#     columns=[
#         "pool_id",
#         "date",
#         "last_burn_date",
#         "burn_date",
#         "lp_id",
#         "start_date",
#         "last_add_date",
#         "liquidity",
#         "lifetime",
#         "last_burn_date",
#         "in_range",
#         "lwr_tick",
#         "upr_tick",
#         "Unnamed: 0",
#         "rewards.volume",
#         "rewards.time",
#         "rewards.fee",
#         "lifetime_m",
#         "lifetime_d",
#         "lifetime_s",
#         "true_pool.price",
#         "synth_pool.price",
#         "true_pool.price/synth_pool.price",
#         "magnitude of price divergence",
#         "synth_pool.liquidity_at_tick",
#         "synth_pool.total_liquidity",
#         "true_pool.liquidity_at_tick",
#         "true_pool.total_liquidity",
#     ]
# )
