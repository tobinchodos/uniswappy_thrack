import argparse
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
        'fee_rate',
        'evt_block_time',
        'pool_liquidity'
    ]
    df = pd.read_csv(filepath, usecols=required_columns)[:n_rows]

    df["amount_lp_token"] = df.amount_lp_token.replace('',0).astype(float).fillna(0)
    df["amount0"] = df.amount0.astype(float).fillna(0)
    df["amount1"] = df.amount1.astype(float).fillna(0)
    df.tickLower = df.tickLower.fillna(0).astype(float)
    df.tickUpper = df.tickUpper.fillna(0).astype(float)

    sqrtPriceX96 = df.sqrtPriceX96.astype(float)
    df["price"] = (sqrtPriceX96 / 2**96) ** 2
    df["current_price"] = df["price"].bfill().ffill()
    df["lp_tuple"] = list(zip(df.liquidity_provider, df.tickLower, df.tickUpper))
    return df

def get_lp_info(subset):
    position = subset.lp_tuple.iloc[0]
    subset['L'] = 0
    subset.loc[subset.lp_tuple == position, 'L'] = subset.loc[subset.lp_tuple == position, 'amount_lp_token'].cumsum()
    L = subset['L'].ffill().fillna(0) # column of L values, for the lifetime of the position.

     # tuple for position 
    fee_rate = subset.iloc[0].fee_rate / 10**6 # e.g. 3000 --> .003

    low_price = 1.0001 ** (position[1]) # low price
    high_price = 1.0001 ** (position[2]) # high price
    assert low_price < high_price

    """Compute inventory + fees for a single LP."""
    vec = np.sqrt(
        np._core.umath.maximum(
            np._core.umath.minimum(subset.current_price, high_price), low_price
        )
    )

    inv0 = L * (1 / vec - 1 / np.sqrt(high_price)) # series representing inventory over time
    inv1 = L * (vec - np.sqrt(low_price)) #inventory over time

    inv0_diff = np.diff(inv0, prepend=0) # diffs
    inv1_diff = np.diff(inv1, prepend=0)

    swaps_logical = subset.event.values == 'swap'
    cond_0 = np.logical_and(swaps_logical, (subset.amount0 > 0))
    cond_1 = np.logical_and(swaps_logical, (subset.amount1 > 0))

    fee0 = np.where(cond_0, inv0_diff * (fee_rate / (1-fee_rate)), 0) # if amount0 is pos, get fee on it
    fee1 = np.where(cond_1, inv1_diff * (fee_rate / (1-fee_rate)), 0) # if amount1 is pos.
    dollar_fees = fee0 * subset.current_price + fee1 # M2M dollarized fees per step

    volume = np.where(cond_0,
                  np.abs(inv0_diff) * subset.current_price,
                  np.where(cond_1,
                           np.abs(inv1_diff),
                           0)
                 )

    row = (
        np.abs(inv0_diff).sum(),     # total absolute changes in inv0
        np.abs(inv1_diff).sum(),     # total absolute changes in inv1
        fee0.sum(),                  # total fee0
        fee1.sum(),                  
        dollar_fees.sum() ,
        volume.sum()
    )

    return(row)

def compute_lp_aggregate_amounts(df, bounds) -> pd.DataFrame:
    rows = []
    numeric_cols = ['inv0_sum', 'inv1_sum', 'fee0_sum', 'fee1_sum','dollar_fees','volume']

    # tqdm shows a progress bar over all unique lp_tuples
    for lp in tqdm(df.lp_tuple.unique(), desc="Computing LP aggregates for {} unique positions".format(df.lp_tuple.nunique())):
        if lp != ('0x', 0.0, 0.0):
            low = bounds.at[lp, 'first_index']
            high = bounds.at[lp, 'last_index'] # last index = last of all, if not burned. otherwise, when L > 0
            subset = df[(df.index >= low) & (df.index <= high)] # lifetime of a given position
            row = get_lp_info(subset)
            rows.append({
                'lp_tuple': lp,
                'inv0_sum': row[0],
                'inv1_sum': row[1],
                'fee0_sum': row[2],
                'fee1_sum': row[3],
                'dollar_fees' : row[4],
                'volume' : row[5]
            })

    out_df = pd.DataFrame(rows)
    out_df[numeric_cols] = out_df[numeric_cols].astype(float)

    return out_df
def get_lp_bounds(df):
    mask = df.event.isin(['mint'])
    
    first_indices = (
        df[mask]
        .groupby('lp_tuple')
        .apply(lambda x: x.index[0])
        .reset_index(name='first_index')
    )
    
    last_burns = (
        df[df.event == 'burn']
        .groupby('lp_tuple')
        .apply(lambda x: x.index.max())
        .reset_index(name='last_burn')
    )

    cumulative_lp_token = (
        df.groupby('lp_tuple')['amount_lp_token']
        .sum()
        .reset_index()
        .rename(columns={'amount_lp_token': 'cum_lp'})
    )
    
    merged = pd.merge(first_indices, last_burns, on='lp_tuple', how='outer')
    merged = pd.merge(merged, cumulative_lp_token, on='lp_tuple', how='left')
    merged['last_index'] = np.where(merged['cum_lp'] <= 0, merged['last_burn'], df.index[-1])
    merged = merged.set_index('lp_tuple')[['first_index','last_index']]
    return merged


def main():
    parser = argparse.ArgumentParser(description="LP analysis (float + parallel)")
    parser.add_argument("--file", type=str, default="weth_dai_events_3000.csv")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--output", type=str, default="True")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Loading {args.rows} rows from {args.file}...")
    df = load_and_preprocess(args.file, args.rows)
    bounds = get_lp_bounds(df) # MAP lp --> min and max indices
    result = compute_lp_aggregate_amounts(df, bounds) # use only rows within those bounds for every LP, output aggregate results for them
    cum_time = time.time()

    compute_time = time.time()
    print(
        f"DONE with compute_inventory_and_fees_parallel, took {compute_time-cum_time}"
    )
    output_file = f"lp_analysis_output_{args.file.split('/')[-1].split('.')[0]}.csv"
    if args.output == "True":
        # result.to_parquet(output_file, engine='pyarrow',compression='snappy')
        result.to_csv(output_file)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"Rows: {args.rows}")
    print(f"Time to completion: {int(m)}m {s:.2f}s")
    if args.output == "True":
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
