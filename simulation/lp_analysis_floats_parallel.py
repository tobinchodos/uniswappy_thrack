import argparse
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count

FEE_RATE = 0.003
FEE_DIVISOR = 0.997

def load_and_preprocess(filepath: str, n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)[:n_rows]
    required_columns = [
        "amount_lp_token",
        "sqrtPriceX96",
        "liquidity_provider",
        "tickLower",
        "tickUpper",
        "event",
        "amount0",
        "amount1",
    ]
    df = df[required_columns]

    df["amount_lp_token"] = df.amount_lp_token.astype(float).fillna(0)
    df["amount0"] = df.amount0.astype(float).fillna(0)
    df["amount1"] = df.amount1.astype(float).fillna(0)
    df.tickLower = df.tickLower.fillna(0).astype(float)
    df.tickUpper = df.tickUpper.fillna(0).astype(float)

    sqrtPriceX96 = df.sqrtPriceX96.astype(float)
    df["price"] = ((sqrtPriceX96 / 2**96) ** 2)
    df["current_price"] = df["price"].bfill().ffill()
    df["lp_tuple"] = list(zip(df.liquidity_provider, df.tickLower, df.tickUpper))
    return df

def compute_cumulative_lp_amounts(df: pd.DataFrame) -> pd.DataFrame:
    lp_cumsum = df.pivot_table(
        index=df.index,
        columns="lp_tuple",
        values="amount_lp_token",
        aggfunc="sum",
    ).fillna(0)
    cumulative = lp_cumsum.cumsum()
    cumulative["current_price"] = df["current_price"]

    for col in cumulative.columns:
        if isinstance(col, tuple):
            _, lower_tick, upper_tick = col
            cumulative[f"{col}_lowPrice"] = 1.0001 ** lower_tick
            cumulative[f"{col}_uprPrice"] = 1.0001 ** upper_tick

    cumulative["event"] = df.event
    cumulative["amount_0"] = df.amount0
    cumulative["amount_1"] = df.amount1
    return cumulative

def _process_lp(col,subset):
    """Compute inventory + fees for a single LP."""
    # col = name of LP position
    # L = subset[0]
    # curr_p = subset[1]
    # low_p = subset[2]
    # high_p = subset[3]
    # amount_0 = subset[4]
    # amount_1 = subset[5]
    # events = subset[6]
    # index = subset[7]

    inv0 = subset[0] * (1 / np.sqrt(np.minimum(np.maximum(subset[1], subset[2]), subset[3])) - 1 / np.sqrt(subset[3]))
    inv1 = subset[0] * (np.sqrt(np.maximum(np.minimum(subset[1], subset[3]), subset[2])) - np.sqrt(subset[2]))

    inv0_diff = np.diff(np.concatenate([[0], inv0]))
    inv1_diff = np.diff(np.concatenate([[0], inv1]))

    cond0 = np.logical_and((subset[6] == "swap"), (subset[4] > 0))
    cond1 = np.logical_and((subset[6] == "swap"), (subset[5] > 0))

    fee0 = np.where(cond0, inv0_diff * (FEE_RATE / FEE_DIVISOR), 0)
    fee1 = np.where(cond1, inv1_diff * (FEE_RATE / FEE_DIVISOR), 0)

    df_lp = pd.DataFrame({
        f"{col}_inventory_0": inv0,
        f"{col}_inventory_1": inv1,
        f"{col}_fee_0": fee0,
        f"{col}_fee_1": fee1
    }, index=subset[7])

    return df_lp

def compute_inventory_and_fees_parallel(cumulative: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    lp_cols = [col for col in df.lp_tuple.unique() if isinstance(col, tuple) and col[0] != "0x"]
    # get copies of necessary cols HERE, pass them. 
    subsets = []
    for i, col in enumerate(lp_cols):
        subset = np.zeros((8, len(cumulative)))
        subset[0] = cumulative[col].values # lp tuple --> L over time. 
        subset[1] = cumulative["current_price"].values
        subset[2] = np.full_like(subset[0], cumulative[f"{col}_lowPrice"].iloc[0])
        subset[3] = np.full_like(subset[0], cumulative[f"{col}_uprPrice"].iloc[0])
        subset[4] = cumulative["amount_0"].values
        subset[5] = cumulative["amount_1"].values
        mapping = {"mint": 0, "burn": 1, "swap": 2}
        subset[6] = cumulative["event"].map(mapping)
        subset[7] = cumulative.index
        subsets.append(subset)

    args = [(col,subset) for col,subset in zip(lp_cols, subsets)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(_process_lp, args) # copies entire cumulative df every time?

    # Merge all LP results
    cumulative = pd.concat([cumulative] + results, axis=1)
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
    print('DONE with CUMULATIVE')
    result = compute_inventory_and_fees_parallel(cumulative, df)

    output_file = "lp_analysis_output_float_parallel.csv"
    if args.output == "True":
        result.to_csv(output_file)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"Rows: {args.rows}")
    print(f"Time to completion: {int(m)}m {s:.2f}s")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
