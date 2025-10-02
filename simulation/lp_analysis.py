import argparse
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import pathlib

LIQUIDITY_CUTOFF = 1e4


def load_and_preprocess(filepath: str, n_rows: int) -> pd.DataFrame:
    print(f"Loading {filepath}...")
    required_columns = [
        "amount_lp_token",
        "sqrtPriceX96",
        "liquidity_provider",
        "tickLower",
        "tickUpper",
        "event",
        "amount0",
        "amount1",
        "fee_rate",
        "evt_block_time",
        "pool_liquidity",
    ]
    if n_rows is not None:
        df = pd.read_csv(filepath, usecols=required_columns, nrows=n_rows)
    else:
        df = pd.read_csv(filepath, usecols=required_columns)
    df["evt_block_time"] = pd.to_datetime(df["evt_block_time"])

    df["amount_lp_token"] = (
        df.amount_lp_token.replace("", "nan").astype(float).fillna(0)
    )
    df["amount0"] = df.amount0.astype(float).fillna(0)
    df["amount1"] = df.amount1.astype(float).fillna(0)
    df.tickLower = df.tickLower.fillna(0).astype(float)
    df.tickUpper = df.tickUpper.fillna(0).astype(float)

    sqrtPriceX96 = df.sqrtPriceX96.astype(float)
    df["price"] = (sqrtPriceX96 / 2**96) ** 2
    df["current_price"] = df["price"].ffill().bfill()
    df["lp_tuple"] = list(zip(df.liquidity_provider, df.tickLower, df.tickUpper))
    return df


def compute_cumulative_lp_amounts(df: pd.DataFrame) -> pd.DataFrame:
    t = time.time()
    # restrict to liq ops before hand
    # df = df[df.event.isin(["mint","burn"])]
    # resultant thing should be num_liq_ops x num_lps, significantly smaller.

    df["evt_index"] = df.index.values
    sub_df = df.loc[
        df.event.isin(["mint", "burn"]), ["lp_tuple", "amount_lp_token", "evt_index"]
    ]
    d = {}
    for i, row in sub_df.iterrows():
        if row.lp_tuple in d:
            d[row.lp_tuple][0].append(row.evt_index)
            d[row.lp_tuple][1].append(d[row.lp_tuple][1][-1] + row.amount_lp_token)
        else:
            d[row.lp_tuple] = [[row.evt_index], [row.amount_lp_token]]

    print(f"cumulative df done, took{time.time()-t}")

    return d


def _process_lp(
    current_price,
    cond_0,
    cond_1,
    col,
    liquidity,
    low_price,
    high_price,
    fee_rate,
    first_alive_idx,
    last_alive_idx,
    liq_idx,
    liq_val,
    last_evt_idx,
    evt_block_time,
):
    """Compute inventory + fees for a single LP."""
    # is the position alive? liquidity>0
    # is the event a swap? event=='swap'
    vec = np.sqrt(
        np._core.umath.maximum(
            np._core.umath.minimum(current_price, high_price), low_price
        )
    )

    inv0 = liquidity * (1 / vec - 1 / np.sqrt(high_price))
    inv1 = liquidity * (vec - np.sqrt(low_price))

    inv0_diff = np.diff(inv0, prepend=0)  # cancellation errors
    inv1_diff = np.diff(inv1, prepend=0)  # cancellation errors

    fee0 = np.where(cond_0, inv0_diff * (fee_rate / (1 - fee_rate)), 0)
    fee1 = np.where(cond_1, inv1_diff * (fee_rate / (1 - fee_rate)), 0)

    volume = np.where(cond_0, inv0_diff * current_price, 0) + np.where(
        cond_1, inv1_diff, 0
    )

    # --- ROI, IL, ETC ---
    #
    #
    fee_0_raw = np.sum(fee0)
    fee_1_raw = np.sum(fee1)
    fee_0_cm2m = np.dot(fee0, current_price)  # fee_0 continuously M2M
    fee_1_cm2m = fee_1_raw  # token_1 valuation of all things "valued"
    fee_0_m2m = fee_0_raw * current_price[-1]
    fee_1_m2m = fee_1_raw

    #
    #
    # [100] [150] --> [150]
    # [ 10] [ 30] --> [40]
    # roi_0 = 10/100 = 0.1
    # roi_1 = 30/150 = 0.2
    #
    # total = 40/150 = 4/15 approx 26%
    # periodized = 13%
    # difference = convexity
    # time weighted roi = 15%
    #
    #
    # a pos' lifetime is broken into n>=1 segments of "constant investment":
    # 0^th segment: liq_val[0] x [liq_idx[0],liq_idx[1])
    # 1^th segment: liq_val[1] x [liq_idx[1],lid_idx[2])
    # ...
    # (n-2)^th seg: liq_val[-2] x [liq_idx[-2],liq_idx[-1])
    # (n-1)^th seg: liq_val[-1] x [liq_idx[-1],*** liq_idx[-1] OR T ****)
    # each of duration atleast 1 (liq_idx[i+1]>=liq_idx[i]+1) except maybe the last one.
    ##
    # liq_idx = [10, 1992, 4546] offset = 10
    # liq_val = [10**16,0,10**5]
    # last_evt_idx = 499999
    #
    total_value_burned = 0
    total_time = 0
    avg_roi = 0
    fee_roi = 0
    inv_roi = 0
    avg_il = 0
    N = len(liq_val)
    for i, val in enumerate(liq_val):
        if i == 0:
            total_value_added = inv0[0] * current_price[0] + inv1[0]
        else:
            idx = liq_idx[i] - liq_idx[0]
            if liq_val[i] > liq_val[i - 1]:  # increase in liq
                # i.e., event_df.event.iloc[liq_idx[i]] == 'mint'
                total_value_added += inv0[idx] * current_price[idx] + inv1[idx]

            elif liq_val[i] < liq_val[i - 1]:  # decrease in liq
                # total_value_burned += (
                #     (inv0[liq_idx[i - 1] - liq_idx[0]] - inv0[idx]) * current_price[idx]
                #     + inv1[liq_idx[i - 1] - liq_idx[0]]
                #     - inv1[idx]
                # )
                assert idx > 0  # should be true if liq_idx strictly increasing.
                total_value_burned += (
                    (inv0[idx - 1] - inv0[idx]) * current_price[idx]
                    + inv1[idx - 1]
                    - inv1[idx]
                )
                # i.e., event_df.event.iloc[liq_idx[i]] == 'burn' and

        if i == 0 or (val > LIQUIDITY_CUTOFF):

            # you must be at the beginning of a new period of positive liquidity. first mint can have liq_val < liq_cutoff.
            start = liq_idx[i] - liq_idx[0]
            # when does the next period start (or terminal time)... interested in [start,stop)
            stop = liq_idx[i + 1] - liq_idx[0] if i < N - 1 else len(liquidity)

            # - time_this_period
            # - want roi this period
            # - want il this period

            # TIME THIS PERIOD (SECONDS)
            time_this_period = (
                evt_block_time[stop - 1] - evt_block_time[start]
            ).item() / 10**9
            time_this_period = max(time_this_period, 1)
            # ROI THIS PERIOD
            value_at_start = inv0[start] * current_price[start] + inv1[start]
            inv_value_at_end = (
                inv0[stop - 1] * current_price[stop - 1] + inv1[stop - 1]
                if stop > start
                else value_at_start
            )
            fee_value_at_end = (
                np.sum(fee0[start:stop]) * current_price[stop - 1]
                + np.sum(fee1[start:stop])
                if stop > start
                else value_at_start
            )
            value_at_end = inv_value_at_end + fee_value_at_end  #
            roi_this_period = value_at_end / value_at_start - 1
            avg_roi += (
                roi_this_period * time_this_period
            )  # TIME WEIGHTED AVERAGE, ROR ON token1 INVESTED PER SECOND...
            total_time += time_this_period
            # IMPERMANENT LOSS THIS PERIOD
            # VALUE VS VALUE IF HELD
            value_if_held = inv0[start] * current_price[stop - 1] + inv1[start]
            il_this_period = value_at_end / value_if_held

            avg_il += il_this_period * time_this_period  # ...

            fee_roi += (fee_value_at_end / value_at_start - 1) * time_this_period
            inv_roi += (inv_value_at_end / value_at_start - 1) * time_this_period
            #######

    total_unclaimed_value = (
        inv0[-1] * current_price[-1] + inv1[-1] + fee_0_cm2m + fee_1_cm2m
    )
    roi_crude = (total_unclaimed_value + total_value_burned) / total_value_added

    avg_roi *= 1 / total_time
    avg_il *= 1 / total_time
    fee_roi *= 1 / total_time
    inv_roi *= 1 / total_time

    # vol_in_range = np.std(current_price[~np.isclose(inv0_diff, 0)])

    assert inv0[0] > 0 or inv1[0] > 0

    dt = np.diff(evt_block_time, 1, prepend=evt_block_time[0])

    time_in_range = np.sum(dt[np.abs(inv0_diff) != 0])
    data = {
        f"lp_tuple": str(col),
        f"inv_0_abs_diff": np.sum(np.abs(inv0_diff)),
        f"inv_1_abs_diff": np.sum(np.abs(inv1_diff)),
        "fee_0_raw": fee_0_raw,  # RAW
        "fee_1_raw": fee_1_raw,
        "fee_0_CM2M": fee_0_cm2m,  # CONT. M2M
        "fee_1_CM2M": fee_1_cm2m,
        f"fee_0_M2M": fee_0_m2m,  # M2M @ end_of_life
        f"fee_1_M2M": fee_1_m2m,
        f"total_fee_CM2M": fee_0_cm2m + fee_1_cm2m,  # price*tok0 + tok1
        f"total_fee_M2M": fee_0_m2m + fee_1_m2m,
        f"volume": np.sum(volume),  # no new info, see fees.
        f"lifetime_hours": max(
            (evt_block_time[-1] - evt_block_time[0]).item() / 10**9, 1
        )
        / 3600,
        f"in_range": np.count_nonzero(
            inv0_diff
        ),  # inv0_diff is subject to numerical error,(cancellation)
        "time_in_range": (time_in_range.item() / 10**9) / 3600,
        f"first_alive_idx": first_alive_idx,
        f"last_alive_idx": last_alive_idx,
        f"avg_liq": np.mean(liquidity),
        f"final_liq": liquidity[-1],
        f"liq_idx": liq_idx,
        f"liq_val": liq_val,
        "roi_time_weighted": avg_roi,
        "il_time_weighted": avg_il,
        "fee_roi": fee_roi,
        "inv_roi": inv_roi,
        "price_apprec": current_price[-1] / current_price[0],
        "price_vol": np.std(current_price),
        "roi_raw": roi_crude,
        # "vol_in_range": vol_in_range,
    }
    return data


def compute_inventory_and_fees_parallel(
    cumulative: pd.DataFrame, df: pd.DataFrame, parallelize
) -> pd.DataFrame:

    # lp_cols = [
    #     col for col in df.lp_tuple.unique() if isinstance(col, tuple) and col[0] != "0x"
    # ]
    lp_cols = list(cumulative.keys())
    # get copies of necessary cols HERE, pass them.

    args_list = []
    current_price = df["current_price"].values  # at all timesteps
    amount_0 = df.amount0.values
    amount_1 = df.amount1.values
    swaps_logical = df.event.values == "swap"
    cond_0 = np.logical_and(swaps_logical, (amount_0 > 0))
    cond_1 = np.logical_and(swaps_logical, (amount_1 > 0))
    fee_rate = df.iloc[0].fee_rate / 10**6
    last_evt_idx = df.index[-1]
    evt_block_time = df.evt_block_time.values

    # np.random.shuffle(lp_cols)
    for i, col in enumerate(lp_cols):

        # recover liquidity step function
        liq_idx, liq_val = cumulative[col]
        # if len(liq_idx)==1, then liq_val[0] is the only liquidity value until end of window
        # if len(liq_idx)>1, then liquidity = [liq_val[0]] * (liq_idx[1]- liq_idx[0] + 1) + [liq_val[1]] * (liq_idx[2]- liq_idx[1] + 1) + ... +[liq_val[-1]] * (liq_idx[-1] -liq_idx[-2] + 1)

        # first idx
        first_alive_idx = liq_idx[0]  # first mint of the lp
        # last idx: NOTE: this last_alive_idx method is not fool proof.
        #   1. the comparison liq_val[-1] ==0.0 should be something more like np.isclose(liq_val[-1],0.0).
        #   2. there's also the probably super rare edge case in which one burns a position then later mints same pos.
        assert liq_val[0] > 0
        if len(liq_idx) > 1:
            last_alive_idx = (
                liq_idx[-1] if liq_val[-1] <= LIQUIDITY_CUTOFF else last_evt_idx
            )
        else:
            last_alive_idx = last_evt_idx

        liquidity = np.repeat(liq_val[0], (last_alive_idx - first_alive_idx + 1))
        N = len(liq_idx)
        if N > 1:
            i = 0
            start = 0
            stop = liq_idx[1] - liq_idx[0] if N > 1 else 0
            while i < N:
                liquidity[start:stop] = liq_val[i]
                i += 1
                start = stop
                if i < N - 1:
                    stop += liq_idx[i + 1] - liq_idx[i]
                elif liq_val[-1] > LIQUIDITY_CUTOFF:
                    # whether or not we should fill the remaining
                    liquidity[stop:] = liq_val[-1]

        # index into the vectors over the lifetime of the pos:

        _current_price = current_price[first_alive_idx : last_alive_idx + 1]
        _cond_0 = cond_0[first_alive_idx : last_alive_idx + 1]
        _cond_1 = cond_1[first_alive_idx : last_alive_idx + 1]
        _evt_block_time = evt_block_time[first_alive_idx : last_alive_idx + 1]
        # _liquidity =

        low_price = 1.0001 ** col[1]
        high_price = 1.0001 ** col[2]
        args_list.append(
            (
                _current_price,
                _cond_0,
                _cond_1,
                col,
                liquidity,
                low_price,
                high_price,
                fee_rate,
                first_alive_idx,
                last_alive_idx,
                liq_idx,
                liq_val,
                last_evt_idx,
                _evt_block_time,
            )
        )
    if parallelize:
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(
                _process_lp, args_list, chunksize=len(args_list) // cpu_count()
            )
    else:
        results = []
        for arg in args_list:
            results.append(_process_lp(*arg))

    # Merge all LP results
    # d = {}
    # for result in results:
    #     d.update(result)
    result_df = pd.DataFrame(results)
    # result_df.columns = [str(i) for i in lp_cols]
    # cumulative = pd.concat(result_df, axis=1)

    return result_df


def main():

    parser = argparse.ArgumentParser(description="LP analysis (float + parallel)")
    parser.add_argument("--file", type=str, default="weth_dai_events_3000.csv")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--output", type=str, default="True")
    parser.add_argument("--parallel", type=str, default="True")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Loading {args.rows} rows from {args.file}...")
    df = load_and_preprocess(args.file, args.rows)
    cumulative = compute_cumulative_lp_amounts(df)
    cum_time = time.time()
    print(f"DONE with CUMULATIVE, took {cum_time-start_time}")

    result = compute_inventory_and_fees_parallel(
        cumulative, df, parallelize=(args.parallel == "True")
    )
    fee_rate = df.iloc[0].fee_rate
    # retrieve info that was discarded in processing above
    # result["price"] = df.price
    result["fee_rate"] = fee_rate
    # result["evt_block_time"] = df.evt_block_time
    # result["pool_liquidity"] = df.pool_liquidity.replace("", "nan").astype(float)
    # result["event"] = df.event
    # result["amount0"] = df.amount0
    # result["amount1"] = df.amount1

    compute_time = time.time()
    print(
        f"DONE with compute_inventory_and_fees_parallel, took {compute_time-cum_time}"
    )

    output_file = f"data/lp_analysis_output_{pathlib.Path(args.file).stem}.parquet"
    if args.output == "True":
        result.to_parquet(output_file, engine="pyarrow", compression="snappy")

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"Rows: {args.rows}")
    print(f"Time to completion: {int(m)}m {s:.2f}s")
    if args.output == "True":
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
