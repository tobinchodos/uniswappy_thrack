This project's purpose is to take a v3-style pool's sequence of events (mints/swaps/burns) and analyze all liquidity positions. In plain language, the primary use case goes like this:

- you pick a token pair, say WETH-USDC
- find a V3-style pool and fee tiers, say 0.03% (high) and 0.005% (low)
- for both tiers, you download the events data (say from Dune via the query below) from the beginning of the pool's lifetime to some stopping period (like number of events = 500k or 1 year's worth of events).
- run the command:
  $python simulation/lp_analysis_multipool.py --files <path_to_your_low_fee_events.csv> <path_to_your_high_fee_events.csv> --output=True --parallel=False

You will get two output files from the above which hold the analysis of the lp positions:

lp*analysis_output*<path_to_your_low_fee_events>.parquet
lp_analysis_output\*<path_to_your_high_fee_events>.parquet

The analysis done and output created is done according to the following:

- find the approximately common period of events for the two pools (e.g., start_date = 05/04/2021 - stop_date = 08/16/2021)
- "integrate forward" all lp's mints/burns over the time period and calculate all lp's fees, volume, ROI, etc.
- output a dataframe for each pool which holds the calculated lp quantities of interest over the window of time

For example usage, see analysis.ipynb, all_pools.ipynb, etc.

For a set of 8 event csv's (four pairs, two fee tiers each), see (https://drive.google.com/drive/folders/1axcOahumYPrMx4OyXLAe8LFLLCteibzq)

**DUNE QUERY**

WITH
NFTPositions AS (
select
chain
, nft_manager_address
, pool_address
, tokenId
, first_recipient
, tickUpper
, tickLower
from dune.thrackle_team.result_uniswap_v3_nft_positions T
where chain = 'ethereum'
and pool_address = {{pool_contract}}
),

LiquidityAdds AS (
-- LiquidityAdd events from NFPM contract
select
N.chain
, tokenId
, N.pool_address  
 , N.first_recipient
, N.tickUpper
, N.tickLower
, L.\*
-- , L.liquidity as amount

        from (select *
             from uniswap_v3_ethereum.nonfungiblepositionmanager_evt_increaseliquidity
             ) L
        left join NFTPositions N
            using (tokenId)
        where N.pool_address = {{pool_contract}}

),

Mints AS (
SELECT
'mint' AS event,
CASE
WHEN ((L.evt_tx_hash IS NOT NULL) AND (M.evt_tx_hash IS NOT NULL))
THEN 'pool-and-nfpm'
WHEN L.evt_tx_hash IS NULL
THEN 'pool'
WHEN M.evt_tx_hash IS NULL
THEN 'nfpm'
ELSE
'error'
END AS source,

        -- Identity / tx
        COALESCE(M.contract_address, L.pool_address)                AS pool,
        COALESCE(M.evt_tx_hash,     L.evt_tx_hash)                  AS evt_tx_hash,
        COALESCE(M.evt_tx_from,     L.evt_tx_from)                  AS evt_tx_from,
        COALESCE(M.evt_index,       L.evt_index)                    AS evt_index,
        COALESCE(M.evt_block_time,  L.evt_block_time)               AS evt_block_time,
        COALESCE(M.evt_block_number,L.evt_block_number)             AS evt_block_number,

        -- Amounts
        COALESCE(M.amount,          L.liquidity)                    AS amount_lp_token,
        COALESCE(M.amount0,         L.amount0)                      AS amount0,
        COALESCE(M.amount1,         L.amount1)                      AS amount1,
        COALESCE(L.tokenId,         0)                              AS tokenId,
        COALESCE(L.first_recipient, TRY_CAST('' AS VARBINARY))      AS first_recipient,

        -- CUSTOM HEURISTIC TO FIND ADVANCED USERS
        CASE
            -- If LiquidityAdds table has an original_recipient, prefer that
            WHEN L.first_recipient IS NOT NULL
              THEN L.first_recipient

            -- Otherwise, if the tx target was NFPM, attribute to tx sender
            WHEN M.evt_tx_to = 0xc36442b4a4522e871399cd717abdd847ab11fe88
                THEN M.evt_tx_from

            -- Or, if the owner field was NFPM, also attribute to tx sender
            WHEN M.owner = 0xc36442b4a4522e871399cd717abdd847ab11fe88
                THEN M.evt_tx_from

            -- Otherwise, fall back to the owner field
            ELSE M.owner
        END AS liquidity_provider,

        -- Raw position fields from either side
        M.owner                                                    AS owner,
        COALESCE(M.tickLower,        L.tickLower)                  AS tickLower,
        COALESCE(M.tickUpper,        L.tickUpper)                  AS tickUpper,

        -- Not available on mint join (placeholders)
        NULL                                                       AS pool_liquidity,
        TRY_CAST('' AS VARBINARY)                                  AS trader,
        -- COALESCE(L.first_recipient, TRY_CAST('' AS VARBINARY))     AS first_recipient,

        -- Sender (who called pool.swap/mint) if present on either side
        M.sender                                                   AS sender,

        -- Price state placeholders (not emitted on Mint)
        NULL                                                       AS sqrtPriceX96,
        NULL                                                       AS tick
    FROM (
        -- Mint events from Pool contract
        select *
        from uniswap_v3_ethereum.uniswapv3pool_evt_mint
        where contract_address = {{pool_contract}}
    ) M
    -- LiquidityAdd events from NFPM contract
    FULL OUTER JOIN LiquidityAdds L
        ON M.evt_tx_hash = L.evt_tx_hash
            AND M.amount = L.liquidity
            AND M.amount0 = L.amount0
            AND M.amount1 = L.amount1
            AND M.contract_address = L.pool_address
            AND L.evt_index BETWEEN M.evt_index AND M.evt_index + 6
        -- USING (evt_tx_hash, amount, amount0, amount1, contract_address) --contract_address is pool_adddress, here we could additionally join by evt_index range

),

LiquidityRemovals AS (
-- LiquidityRemoval events from NFPM contract
select
N.chain
, tokenId
, N.pool*address  
 , N.first_recipient
, N.tickUpper
, N.tickLower
, L.*
from (select \_
from uniswap_v3_ethereum.nonfungiblepositionmanager_evt_decreaseliquidity
where liquidity > 0
-- where contract_address = {{pool_contract}}
) L
left join NFTPositions N
using (tokenId)
where N.pool_address = {{pool_contract}}
),

Burns AS (
SELECT
'burn' AS event,
CASE
WHEN ((L.evt_tx_hash IS NOT NULL) AND (M.evt_tx_hash IS NOT NULL))
THEN 'pool-and-nfpm'
WHEN L.evt_tx_hash IS NULL
THEN 'pool'
WHEN M.evt_tx_hash IS NULL
THEN 'nfpm'
ELSE
'error'
END AS source,  
 -- Identity / tx
COALESCE(M.contract_address, L.contract_address) AS pool,
COALESCE(M.evt_tx_hash, L.evt_tx_hash) AS evt_tx_hash,
COALESCE(M.evt_tx_from, L.evt_tx_from) AS evt_tx_from,
COALESCE(M.evt_index, L.evt_index) AS evt_index,
COALESCE(M.evt_block_time, L.evt_block_time) AS evt_block_time,
COALESCE(M.evt_block_number,L.evt_block_number) AS evt_block_number,

        -- Amounts
        COALESCE(-M.amount,        -L.liquidity)                    AS amount_lp_token,
        COALESCE(-M.amount0,       -L.amount0)                      AS amount0,
        COALESCE(-M.amount1,       -L.amount1)                      AS amount1,
        COALESCE(L.tokenId,         0)                              AS tokenId,
        COALESCE(L.first_recipient, TRY_CAST('' AS VARBINARY))      AS first_recipient,


        -- CUSTOM HEURISTIC TO FIND ADVANCED USERS
        CASE
            -- If LiquidityAdds table has an original_recipient, prefer that
            WHEN L.first_recipient IS NOT NULL
              THEN L.first_recipient

            -- Otherwise, if the tx target was NFPM, attribute to tx sender
            WHEN M.evt_tx_to = 0xc36442b4a4522e871399cd717abdd847ab11fe88
                THEN M.evt_tx_from

            -- Or, if the owner field was NFPM, also attribute to tx sender
            WHEN M.owner = 0xc36442b4a4522e871399cd717abdd847ab11fe88
                THEN M.evt_tx_from

            -- Otherwise, fall back to the owner field
            ELSE M.owner
        END AS liquidity_provider,

        -- Raw position fields from either side
        M.owner                                                    AS owner,
        COALESCE(M.tickLower,        L.tickLower)                  AS tickLower,
        COALESCE(M.tickUpper,        L.tickUpper)                  AS tickUpper,

        -- Not available on burn join (placeholders)
        NULL                                                       AS pool_liquidity,
        TRY_CAST('' AS VARBINARY)                                  AS trader,
        TRY_CAST('' AS VARBINARY)                                  AS sender,

        -- Price state placeholders (not emitted on BURN)
        NULL                                                       AS sqrtPriceX96,
        NULL                                                       AS tick
    FROM (
        -- Burn events from Pool contract
        select *
        from uniswap_v3_ethereum.uniswapv3pool_evt_burn
        where contract_address = {{pool_contract}}
            and amount > 0
    ) M
    -- LiquidityRemovals events from NFPM contract
    FULL OUTER JOIN LiquidityRemovals L
    -- USING (evt_tx_hash, amount, amount0, amount1, contract_address) --contract_address is pool_adddress, here we could additionally join by evt_index range
        ON M.evt_tx_hash = L.evt_tx_hash
        AND M.amount = L.liquidity
        AND M.amount0 = L.amount0
        AND M.amount1 = L.amount1
        AND M.contract_address = L.pool_address
        AND L.evt_index BETWEEN M.evt_index AND M.evt_index + 6

),

Swaps AS (
SELECT
'swap' AS event
, NULL as source
, S.contract_address as pool
, S.evt_tx_hash
, S.evt_tx_from
, S.evt_index
, S.evt_block_time
, S.evt_block_number
, NULL AS amount_lp_token
, S.amount0
, S.amount1
, NULL as tokenId
, TRY_CAST('' AS VARBINARY) AS first_recipient
, TRY_CAST('' AS VARBINARY) AS liquidity_provider
-- , NULL AS tokenId
, TRY_CAST('' AS VARBINARY) AS owner
, NULL AS tickLower
, NULL AS tickUpper
, S.liquidity AS pool_liquidity
, T."from" AS trader --S.recipient
-- , S.recipient
, S.sender
, S.sqrtPriceX96
, S.tick
FROM (
select \*
from uniswap_v3_ethereum.uniswapv3pool_evt_swap
where contract_address = {{pool_contract}}
) S
LEFT JOIN (select hash, "from" from ethereum.transactions) T
ON S.evt_tx_hash = T.hash

),

Combined AS (
select _ from Burns
UNION ALL
select _ from Mints
UNION ALL
select \* from Swaps
)
select
S.event
, S.source
, S.evt_tx_hash
, S.evt_index
, S.evt_block_time
, S.evt_block_number
, CAST(S.amount_lp_token AS VARCHAR) AS amount_lp_token
, CAST(S.amount0 AS VARCHAR) AS amount0
, CAST(S.amount1 AS VARCHAR) AS amount1
, S.tokenId
, S.first_recipient
, S.liquidity_provider
-- , S.owner
, S.tickLower
, S.tickUpper
, CAST(S.pool_liquidity AS VARCHAR) AS pool_liquidity
, S.trader
-- , S.recipient
-- , S.sender
, S.sqrtPriceX96
, S.tick
from Combined S
-- INNER JOIN Pool P ON S.contract_address = P.pool
-- LEFT JOIN NFTTransfers N ON N.amount_lp_token_abs = abs(S.amount_lp_token)
order by S.evt_block_number ASC, S.evt_index
LIMIT 500000
