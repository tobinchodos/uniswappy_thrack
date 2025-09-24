# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

pd.options.plotting.backend = "plotly"

high_fee = pd.read_parquet("lp_analysis_output_weth_dai_events_3000.parquet")
low_fee = pd.read_parquet("lp_analysis_output_weth_dai_events_500.parquet")

high_fee["date"] = pd.to_datetime(high_fee.evt_block_time).dt.normalize()
low_fee["date"] = pd.to_datetime(low_fee.evt_block_time).dt.normalize()
high_fee["evt_block_time"] = pd.to_datetime(high_fee.evt_block_time)
low_fee["evt_block_time"] = pd.to_datetime(low_fee.evt_block_time)
end_date = min(high_fee.date.max(), low_fee.date.max())
high_fee = high_fee[high_fee.date <= end_date]
low_fee = low_fee[low_fee.date <= end_date]
high_fee.drop(columns=["('0x', 0.0, 0.0)"], inplace=True)
low_fee.drop(columns=["('0x', 0.0, 0.0)"], inplace=True)
# %%
# for _df in [high_fee,low_fee]:
#     # lp stuff
#     _df['']


h_lps = [pos.lstrip("(").split(",")[0] for pos in high_fee.filter(regex="\\)$").columns]
l_lps = [pos.lstrip("(").split(",")[0] for pos in low_fee.filter(regex="\\)$").columns]
common_lps = list(set(h_lps).intersection(set(l_lps)))
common_df = pd.concat(
    [
        high_fee.filter(regex="evt_block_time|" + "|".join(common_lps))
        .set_index("evt_block_time")
        .add_prefix("high_fee."),
        low_fee.filter(regex="evt_block_time|" + "|".join(common_lps))
        .set_index("evt_block_time")
        .add_prefix("low_fee."),
    ]
).sort_index()

# common_df.filter(regex=np.random.choice(common_lps)).plot()

# pos_info = dict.fromkeys(common_lps, 0)
# for lp_id in common_lps:
#     pos_info[lp_id] = len(common_df.filter(regex=lp_id).columns) / 5
