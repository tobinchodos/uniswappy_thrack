from uniswappy.utils.tools.v3.UniV3Utils import UniV3Utils
from uniswappy.erc.IndexERC20 import ERC20
from uniswappy.cpt.factory.UniswapFactory import UniswapFactory
from uniswappy.utils.data.UniswapExchangeData import UniswapExchangeData

from uniswappy.process.join.Join import Join
from uniswappy.process.swap.Swap import Swap
from uniswappy.process.liquidity.AddLiquidity import AddLiquidity

import pandas as pd
import random
import math
from humanposition import HumanPosition
from uniswappy import UniV3Helper

helper = UniV3Helper()
from decimal import Decimal


class UniswapPool:

    def __init__(
        self,
        env=None,
        pool_id=None,
        address=None,
        token_0=None,
        amount_0=None,
        token_1=None,
        amount_1=None,
        lwr_tick=None,
        upr_tick=None,
        fee=None,
        date=None,
        **kwargs,
    ):
        self.env = env
        self.pool_id = pool_id
        self.address = address
        self.token_0 = self.env.tokens.get(token_0)
        self.amount_0 = amount_0  # not needed anymore
        self.token_1 = self.env.tokens.get(token_1)
        self.amount_1 = amount_1  # not needed anymore
        self.fee = fee
        self.tick_spacing = UniV3Utils.TICK_SPACINGS[fee]
        self.lwr_tick = (
            int(lwr_tick)
            if lwr_tick is not None
            else UniV3Utils.getMinTick(self.tick_spacing)
        )
        self.upr_tick = (
            int(upr_tick)
            if upr_tick is not None
            else UniV3Utils.getMaxTick(self.tick_spacing)
        )
        self._factory = UniswapFactory(
            "Base token pool factory", self.env.get_fake_address()
        )
        self._pool = None
        self.positions = {}
        self.date = date
        self.orig_kwargs = kwargs
        return

    def log(self, message):
        self.env.log(f"pool_id={self.pool_id}" + message)

    def reset(self, seed=None):
        self.create_pool()
        return

    def create_pool(self):
        self._pool = self._factory.deploy(
            UniswapExchangeData(
                tkn0=self.token_0,
                tkn1=self.token_1,
                symbol="uni",
                address=self.address,
                version="V3",
                tick_spacing=self.tick_spacing,
                fee=self.fee,
                precision="GWEI",
            )
        )

        self.positions = {}  # key = {(lp_id,lwr_tick,upr_tick): val= HumanPosition()}

        self._pool.initialize(self.orig_kwargs.get("sqrt_price_X96"))

        inventory = self.add_liquidity_position(
            lp_id=self.orig_kwargs.get("lp_id"),
            token_in="X",  # not needed with nonnull liq now
            amount_in=self.amount_0,  # not needed with nonnull liq now
            lwr_tick=self.lwr_tick,
            upr_tick=self.upr_tick,
            liquidity=self.orig_kwargs.get("liquidity"),
            date=self.date,
            advanced_user=self.orig_kwargs.get("advanced_user"),
        )

        self.inventory_on_init = inventory

        return

    def get_token(self, idx):

        assert idx in [0, 1] or idx in self.env.tokens, f"got {idx}"
        if idx in [0, 1]:
            token_obj = list(self._pool.factory.token_from_exchange.values())[0][
                self._pool.token0 if idx == 0 else self._pool.token1
            ]
        else:
            token_obj = list(self._pool.factory.token_from_exchange.values())[0][idx]

        return token_obj

    def add_liquidity_position(
        self,
        lp_id,  #'BS_id'
        token_in,  # 'Y'
        amount_in,  # $1.00
        lwr_price=None,  # 3451
        upr_price=None,  # 3455
        lwr_tick=None,
        upr_tick=None,
        liquidity=None,
        date=None,
        advanced_user=None,
    ):
        lwr_tick = (
            int(lwr_tick)
            if lwr_tick is not None
            else helper.get_price_tick(self._pool, 0, lwr_price, self.tick_spacing)
        )
        upr_tick = (
            int(upr_tick)
            if upr_tick is not None
            else helper.get_price_tick(self._pool, 0, upr_price, self.tick_spacing)
        )

        if liquidity is not None:
            ix, iy = self._pool.mint(lp_id, lwr_tick, upr_tick, liquidity)
            inventory = {
                self._pool.token0: ix,
                self._pool.token1: iy,
            }

        else:
            inventory = AddLiquidity().apply(
                lp=self._pool,
                user_nm=lp_id,
                token_in=self.get_token(token_in),
                amount_in=amount_in,
                lwr_tick=lwr_tick,
                upr_tick=upr_tick,
            )
        if (lp_id, lwr_tick, upr_tick) in self.positions:
            # put burned warning maybe...
            hp = self.positions.get((lp_id, lwr_tick, upr_tick))
            hp.on_liquidity_update(
                kind="add",
                date=date,
                liquidity=liquidity,
                amount_0=ix,
                amount_1=iy,
                price=self.get_price(),
            )
        else:
            hp = HumanPosition(
                pool=self,
                owner=lp_id,
                lwr_tick=lwr_tick,
                upr_tick=upr_tick,
                token_in=token_in,
                amount_in=amount_in,
                inventory_0_init=inventory.get(self._pool.token0),
                inventory_1_init=inventory.get(self._pool.token1),
                start_date=date,
                advanced_user=advanced_user,
            )
            hp.on_liquidity_update(
                kind="add",
                date=date,
                liquidity=liquidity,
                amount_0=inventory.get(self._pool.token0),
                amount_1=inventory.get(self._pool.token1),
                price=self.get_price(),
            )
            self.positions.update({(lp_id, lwr_tick, upr_tick): hp})

        return hp

    def burn(self, lp_id, lwr_tick, upr_tick, liquidity_amount, date):
        hp = self.positions.get((lp_id, int(lwr_tick), int(upr_tick)))
        assert hp is not None
        out = self._pool.burn(
            recipient=lp_id,
            tickLower=int(lwr_tick),
            tickUpper=int(upr_tick),
            amount=liquidity_amount,
        )
        amount_0, amount_1 = out[-2::]
        # should be put inside the hp
        hp.on_liquidity_update(
            kind="burn",
            date=date,
            liquidity=liquidity_amount,
            amount_0=amount_0,
            amount_1=amount_1,
            price=self.get_price(),
        )

        if hp.is_burned:
            # more precisely, hp.liquidity <= 0, not sure about fees.
            self.log(f"position extinguished: {(lp_id, int(lwr_tick), int(upr_tick))}")

        return amount_0, amount_1

    def get_price(self):
        return self._pool.get_price(self.token_0)

    def swap(self, token_in, amount_in, trader_id=None):
        # pre
        alive_positions = [pos for pos in self.positions.values() if not pos.is_burned]

        for pos in alive_positions:
            pos.pre_swap(ctx={"token_in": token_in, "amount_in": amount_in})

        out = Swap().apply(
            lp=self._pool,
            user_nm=trader_id or "Somebody",
            token_in=self.env.tokens[token_in],
            amount_in=amount_in,
            sqrt_price_limit=None,
        )
        # out = self._pool.swap(
        #     recipient=trader_id or "Somebody",
        #     zeroForOne=(token_in == self.token_0),
        #     amount=amount_in, (positive -> exact input, negative -> exact output)
        #     limit=None,
        # )
        # post
        total_volume_this_swap = 0
        for pos in alive_positions:
            pos.post_swap(ctx={"token_in": token_in, "amount_in": amount_in})
            total_volume_this_swap += pos.volume_this_swap
        for pos in alive_positions:
            pos.pool_volume += total_volume_this_swap

        return out

    def get_state(self):
        state = {
            "price": Decimal(self.get_price()),
            "liquidity_at_tick": Decimal(self.liquidity_at_price()),
            "total_liquidity": Decimal(self._pool.total_supply),
            "positions": self.get_all_positions(),
        }

        return state

    def liquidity_at_price(self, price=None):
        # find the left most stored tick not exceeding. could make more efficient with splitting or whatever, don't care rn.
        price = price if price is not None else self.get_price()
        tick = UniV3Helper().price_to_tick(price)
        best_tick = None

        active_liq = 0
        for tick_, pos in sorted(self._pool.ticks.items(), key=lambda item: item[0]):

            if tick >= tick_:
                best_tick = tick_
                active_liq += pos.liquidityNet
            else:
                break
        # liq = self._pool.ticks[best_tick].liquidityGross if best_tick else None
        return active_liq

    def prices_to_liquidity(self):
        d = {
            UniV3Helper().tick_to_price(tick): (pos.liquidityGross, pos.liquidityNet)
            for tick, pos in sorted(self._pool.ticks.items(), key=lambda item: item[0])
        }
        cum_sum = 0
        d_ = dict.fromkeys(d.keys(), 0)
        for i, (k, v) in enumerate(d.items()):
            cum_sum += v[1]
            d_[k] = cum_sum

        return d, d_

    def get_lp_ids(self):
        return list(map(lambda x: x[0], self.positions.keys()))

    def get_positions_for(self, lp_id):
        return [hp for k, hp in self.positions.items() if k[0] == lp_id]

    def get_lp_dict(self):
        return {
            lp_id: self.get_positions_for(lp_id=lp_id) for lp_id in self.get_lp_ids()
        }

    def get_all_positions(self, as_dicts=True):
        if as_dicts:
            all_positions = [hp.to_dict() for hp in self.positions.values()]
        else:
            all_positions = list(self.positions.values())
        return all_positions

    def get_pos_df(self):
        df = pd.DataFrame.from_records(self.get_all_positions())
        return df

    def disburse_rewards(self):
        # only called on swaps.
        alive_positions = [pos for pos in self.positions.values() if not pos.is_burned]
        totals = {"time": 0, "time_and_liq": 0, "volume": 0, "fee": 0}

        for pos in alive_positions:
            # time reward
            pos.rewards["time"] = 1 if pos.in_range_pre_swap else 0
            totals["time"] += pos.rewards["time"]
            # time and liq
            pos.rewards["time_and_liq"] = pos.liquidity * pos.rewards["time"]

            totals["time_and_liq"] += pos.rewards["time_and_liq"]
            # volume reward

            pos.rewards["volume"] = pos.volume_this_swap
            totals["volume"] += pos.rewards["volume"]

            # fee reward
            pos.rewards["fee"] = pos.fee_value_earned_this_swap
            totals["fee"] += pos.rewards["fee"]

        # NORMALIZATION PRO_RATA IN SPACE
        for pos in alive_positions:
            for reward_key, reward_total in totals.items():
                pos.rewards[reward_key] = (
                    pos.rewards[reward_key] / reward_total if reward_total > 0 else 0
                )
                pos.total_rewards[reward_key] += pos.rewards[reward_key]

        return
