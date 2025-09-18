import argparse
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count



def load_and_preprocess(filepath: str, n_rows: int) -> pd.DataFrame:
    required_columns = [
        "amount_lp_token",
        "sqrtPriceX96",
        "liquidity_provider",
        "tickLower",
        "tickUpper",
        "event",
        "amount0",
        "amount1",
        'fee_rate'
    ]
    df = pd.read_csv(filepath, usecols=required_columns)[:n_rows]

    df["amount_lp_token"] = df.amount_lp_token.astype(float).fillna(0)
    df["amount0"] = df.amount0.astype(float).fillna(0)
    df["amount1"] = df.amount1.astype(float).fillna(0)
    df.tickLower = df.tickLower.fillna(0).astype(float)
    df.tickUpper = df.tickUpper.fillna(0).astype(float)

    sqrtPriceX96 = df.sqrtPriceX96.astype(float)
    df["price"] = (sqrtPriceX96 / 2**96) ** 2
    df["current_price"] = df["price"].bfill().ffill()
    df["lp_tuple"] = list(zip(df.liquidity_provider, df.tickLower, df.tickUpper))
    return df


def compute_cumulative_lp_amounts(df: pd.DataFrame) -> pd.DataFrame:
    t = time.time()
    cumulative = (
        df.groupby([df.index, "lp_tuple"])["amount_lp_token"]
        .sum()
        .unstack(fill_value=0)
        .cumsum()
    )
    print(f"cumulative df done, took{t-time.time()}")
    return cumulative





def _process_lp(
    current_price,
    cond_0,
    cond_1,
    col,
    liquidity,
    low_price,
    high_price,
    fee_rate
):
    """Compute inventory + fees for a single LP."""
    vec = np.sqrt(
        np._core.umath.maximum(
            np._core.umath.minimum(current_price, high_price), low_price
        )
    )
    inv0 = liquidity * (1 / vec - 1 / np.sqrt(high_price))
    inv1 = liquidity * (vec - np.sqrt(low_price))

    inv0_diff = np.diff(inv0, prepend=0)
    inv1_diff = np.diff(inv1, prepend=0)

    fee0 = np.where(cond_0, inv0_diff * (fee_rate / (1-fee_rate)), 0)
    fee1 = np.where(cond_1, inv1_diff * (fee_rate / (1-fee_rate)), 0)

    data = {
        f"{col}_inventory_0": inv0,
        f"{col}_inventory_1": inv1,
        f"{col}_fee_0": fee0,
        f"{col}_fee_1": fee1,
    }
    return data



def compute_inventory_and_fees_parallel(
    cumulative: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:

    lp_cols = [
        col for col in df.lp_tuple.unique() if isinstance(col, tuple) and col[0] != "0x"
    ]
    # get copies of necessary cols HERE, pass them.

    args_list = []
    current_price = df["current_price"].values
    amount_0 = df.amount0.values
    amount_1 = df.amount1.values
    swaps_logical = df.event.values == 'swap'
    cond_0 = np.logical_and(swaps_logical, (amount_0 > 0))
    cond_1 = np.logical_and(swaps_logical, (amount_1 > 0))
    fee_rate = df.iloc[0].fee_rate / 10**6

    for i, col in enumerate(lp_cols):

        liquidity = cumulative[col].values
        low_price = 1.0001 ** col[1]
        high_price = 1.0001 ** col[2]
        args_list.append(
            (
                current_price,
                cond_0,
                cond_1,
                col,
                liquidity,
                low_price,
                high_price,
                fee_rate
            )
        )

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(
            _process_lp, args_list, chunksize=len(args_list) // cpu_count()
        )

    # Merge all LP results
    d = {}
    for result in results:
        d.update(result)
    result_df = pd.DataFrame(d)
    cumulative.columns = [str(i) for i in cumulative.columns]
    cumulative = pd.concat([cumulative, result_df], axis=1)

    return cumulative


def main():

    parser = argparse.ArgumentParser(description="LP analysis (float + parallel)")
    parser.add_argument("--file", type=str, default="weth_dai_events_3000.csv")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--output", type=str, default="True")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Loading {args.rows} rows from {args.file}...")
    df = load_and_preprocess(args.file, args.rows)
    cumulative = compute_cumulative_lp_amounts(df)
    cum_time = time.time()
    print(f"DONE with CUMULATIVE, took {cum_time-start_time}")

    result = compute_inventory_and_fees_parallel(cumulative, df)
    fee_rate = df.iloc[0].fee_rate
    result['fee_rate'] = fee_rate
    compute_time = time.time()
    print(
        f"DONE with compute_inventory_and_fees_parallel, took {compute_time-cum_time}"
    )
    output_file = f"lp_analysis_output_{args.file.split('.')[0]}.parquet"
    if args.output == "True":
        result.to_parquet(output_file, engine='pyarrow',compression='snappy')

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"Rows: {args.rows}")
    print(f"Time to completion: {int(m)}m {s:.2f}s")
    if args.output == "True":
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
