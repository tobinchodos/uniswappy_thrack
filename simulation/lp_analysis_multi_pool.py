import time
import argparse
import pathlib
import pandas as pd
from lp_analysis import (
    load_and_preprocess,
    compute_cumulative_lp_amounts,
    compute_inventory_and_fees_parallel,
)
from multiprocessing import Pool, cpu_count


def clamp_dfs(input_dfs):
    greatest_min = input_dfs[0].evt_block_time.min()
    least_max = input_dfs[0].evt_block_time.max()
    for df in input_dfs:
        _min, _max = df.evt_block_time.min(), df.evt_block_time.max()
        if _min > greatest_min:
            greatest_min = _min
        if _max < least_max:
            least_max = _max

    assert least_max >= greatest_min, "no data!"

    dfs = []
    for df in input_dfs:
        dfs.append(df[df.evt_block_time <= least_max].copy())
    return dfs


def analyze_df(df, file_name):
    analysis_start_time = time.time()
    cumulative = compute_cumulative_lp_amounts(df)
    stamp = time.time()
    print(f"--> {file_name} -- DONE with CUMULATIVE, took {stamp-analysis_start_time}")

    stamp = time.time()
    result = compute_inventory_and_fees_parallel(cumulative, df, parallelize=False)
    print(
        f"--> {file_name} --DONE with compute_inventory_and_fees_parallel, took {time.time()-stamp}"
    )
    stamp = time.time()
    fee_rate = df.iloc[0].fee_rate
    result["fee_rate"] = fee_rate

    output_file = f"data/lp_analysis_output_{pathlib.Path(file_name).stem}.parquet"

    result.to_parquet(output_file, engine="pyarrow", compression="snappy")
    print(
        f"--> {file_name} done with saving output to {output_file}, took {time.time()-stamp}"
    )
    print(f"--> {file_name}: DONE, took {time.time()-analysis_start_time}")


def main():
    # input:
    # --files data/weth_usdc_500.csv data/weth_usdc_3000.csv --rows 500000 500000 --clamp_time_frame True --output True --parallel=True
    parser = argparse.ArgumentParser(description="LP analysis (float + parallel)")
    parser.add_argument("--files", nargs="*", type=str)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--clamp_time_frame", type=str, default="True")
    parser.add_argument("--output", type=str, default="True")
    parser.add_argument("--parallel", type=str, default="True")
    args = parser.parse_args()

    start_time = time.time()

    dfs = []
    for file in args.files:
        dfs.append(load_and_preprocess(file, args.max_rows))
    # shave to maximal overlap window
    dfs = clamp_dfs(dfs)

    if args.parallel == "True":
        with Pool(processes=cpu_count()) as pool:
            pool.starmap(analyze_df, list(zip(dfs, args.files)), chunksize=1)
    else:
        for file_name, df in zip(args.files, dfs):
            print(f"-->{file_name} has num_rows = {df.shape[0]}")
            analyze_df(df, file_name=file_name)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"Time to completion: {int(m)}m {s:.2f}s")


if __name__ == "__main__":
    main()
