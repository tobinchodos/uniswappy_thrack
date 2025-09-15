from uniswappy import UniV3Helper
from copy import deepcopy
import numpy as np


class HumanPosition:

    def __init__(
        self,
        pool,
        owner,
        lwr_tick,
        upr_tick,
        token_in,
        amount_in,
        inventory_0_init=0,
        inventory_1_init=0,
        start_date=None,
        burn_date=None,
        advanced_user=None,
    ):
        self.pool = pool
        self.owner = owner
        self.deposit_0 = inventory_0_init
        self.deposit_1 = inventory_1_init
        self.price_at_deposit = self.pool.get_price()
        self.lwr_tick = lwr_tick
        self.upr_tick = upr_tick

        self.lwr_price = UniV3Helper().tick_to_price(lwr_tick)
        self.upr_price = UniV3Helper().tick_to_price(upr_tick)
        self.token_in = token_in  # not needed anymore
        self.amount_in = amount_in  # not needed anymore
        self.fees_earned_0 = 0
        self.fees_earned_1 = 0
        self._num_steps_in_range = 0
        self._num_swaps_in_range = 0
        self._total_value_0_added = 0
        self._total_value_1_added = 0
        self._total_value_added = 0
        self.volume_this_swap = 0
        self.total_volume = 0
        self.pool_volume = 0  # set by pool.post_swap(). see self.prop_volume
        self.num_swaps_during_lifetime = (
            0  # for the POOL. to compute relative prop of swaps in range
        )
        self.start_date = start_date
        self.updates = {
            "adds": [],
            "burns": [],
        }  # will be updated on init by on_liquidity_update() below by the pool
        self.burn_date = burn_date
        self.rewards = {"time": 0, "time_and_liq": 0, "volume": 0, "fee": 0}
        self.total_rewards = {"time": 0, "time_and_liq": 0, "volume": 0, "fee": 0}
        self.advanced_user = advanced_user
        self.in_range_pre_swap = False  # reset on pre_swap
        self.total_fee_value_earned = 0

        self.raw_pos_info = list(
            filter(
                lambda x: x[0] == self.lwr_tick and x[1] == self.upr_tick,
                self.pool._pool.get_positions_for_owner(self.owner),
            )
        )[0][-1]
        self._exp_fees_0 = 0
        self._exp_fees_1 = 0

    def pre_swap(self, ctx=None):
        # called by pool pre swap, only those alive ones
        if not self.is_burned:
            # self.collect_fees()
            self.num_swaps_during_lifetime += (
                1  # swaps seen by pool during this guys life
            )
            self._num_swaps_in_range += 1 if self.in_range else 0
            self._pre_swap_inv_0 = self.inventory_0
            self._pre_swap_inv_1 = self.inventory_1
            self._pre_swap_fees_0 = self.fees_earned_0
            self._pre_swap_fees_1 = self.fees_earned_1
            self.in_range_pre_swap = self.in_range

    def post_swap(self, ctx=None):
        # called by pool, only those alive ones.
        if not self.is_burned:  #
            # self.collect_fees()
            # inventory changes & volume
            delta_inv_0 = self.inventory_0 - self._pre_swap_inv_0
            delta_inv_1 = self.inventory_1 - self._pre_swap_inv_1
            assert delta_inv_1 >= 0 or delta_inv_0 >= 0, "something wrong"
            dollar_vol_0 = delta_inv_0 * self.pool.get_price()
            dollar_vol_1 = delta_inv_1
            volume = dollar_vol_0 if dollar_vol_0 >= 0 else dollar_vol_1
            self.volume_this_swap = volume
            self.total_volume += self.volume_this_swap
            # fees

            fees_earned_this_swap_0 = self.fees_earned_0 - self._pre_swap_fees_0
            fees_earned_this_swap_1 = self.fees_earned_1 - self._pre_swap_fees_1

            assert (
                fees_earned_this_swap_0 >= 0 and fees_earned_this_swap_1 >= 0
            ), "something wrong"
            # if self.in_range_pre_swap:
            #     if not (fees_earned_this_swap_0 > 0 or fees_earned_this_swap_1 > 0):
            #         assert ctx.get("amount_in") < 10**6  # miniscule purchase

            self.fee_value_earned_this_swap = (
                fees_earned_this_swap_0 * self.pool.get_price()
                + fees_earned_this_swap_1
            )
            self.total_fee_value_earned += self.fee_value_earned_this_swap

            self._exp_fees_0 += (
                delta_inv_0 * 0.003 / (1 - 0.003) if delta_inv_0 > 0 else 0
            )
            self._exp_fees_1 += (
                delta_inv_1 * 0.003 / (1 - 0.003) if delta_inv_1 > 0 else 0
            )

        return

    @property
    def prop_swaps_in_range(self):

        return (
            self.num_swaps_in_range / self.num_swaps_during_lifetime
            if self.num_swaps_during_lifetime > 0
            else 0
        )

    @property
    def prop_volume(self):

        return self.total_volume / self.pool_volume if self.pool_volume > 0 else 0

    def on_liquidity_update(
        self,
        kind=None,
        date=None,
        liquidity=None,
        amount_0=None,
        amount_1=None,
        price=None,
    ):
        self.updates.get("adds" if kind == "add" else "burns").append(
            dict(
                date=date,
                liquidity=liquidity,
                amount_0=amount_0,
                amount_1=amount_1,
                price=price,
            )
        )
        if kind == "add":
            self._total_value_0_added += amount_0 * price
            self._total_value_1_added += amount_1

        if self.is_burned:
            self.burn_date = date
        return

    @property
    def num_steps_in_range(self):
        return self._num_steps_in_range

    @property
    def num_swaps_in_range(self):
        return self._num_swaps_in_range

    @property
    def total_value_0_added(self):
        return self._total_value_0_added

    @property
    def total_value_1_added(self):
        return self._total_value_1_added

    @property
    def total_value_added(self):
        return self.total_value_0_added + self.total_value_1_added

    def get_state(self):
        return self.to_dict()

    def to_dict(self):

        d = {}
        d.update(
            {
                "pool_id": self.pool.pool_id,
                "lp_id": self.owner,
                "start_date": self.start_date,
                "lwr_tick": self.lwr_tick,
                "upr_tick": self.upr_tick,
                "lwr_price": self.lwr_price,
                "upr_price": self.upr_price,
                "price_at_deposit": self.price_at_deposit,
                "width": self.width,
                "fees_earned_0": self.fees_earned_0,
                "fees_earned_1": self.fees_earned_1,
                "in_range": self.in_range,
                "inventory_0": self.inventory_0,
                "inventory_1": self.inventory_1,
                "liquidity": self.liquidity,
                "is_burned": self.is_burned,
                "burn_date": self.burn_date,
                "num_adds": self.num_adds,
                "num_burns": self.num_burns,
                "last_add_date": self.last_add_date,
                "last_burn_date": self.last_burn_date,
                "last_add_amount": self.last_add_amount,
                "last_burn_amount": self.last_burn_amount,
                "liquidity_added": self.liquidity_added,
                "liquidity_burned": self.liquidity_burned,
                "inventory_value": self.inventory_value,  # amount_0 * price +amount_1
                "fee_value": self.fee_value,
                "total_value": self.total_value,
                "num_steps_in_range": self.num_steps_in_range,  # counting mints and burns
                "num_swaps_in_range": self.num_swaps_in_range,  # swaps only
                "total_value_0_added": self.total_value_0_added,  # amount_0 * price
                "total_value_1_added": self.total_value_1_added,
                "total_value_added": self.total_value_added,
                "prop_swaps_in_range": self.prop_swaps_in_range,  # number of swaps we were in range / number of swaps in the pool
                "total_volume": self.total_volume,  # volume this pos saw while alive
                "prop_volume": self.prop_volume,  # volume this pos saw relative to pool while alive
                "rewards.time": self.rewards.get("time"),
                "total_rewards.time": self.total_rewards.get("time"),
                "rewards.time_and_liq": self.rewards.get("time_and_liq"),
                "total_rewards.time_and_liq": self.total_rewards.get("time_and_liq"),
                "rewards.volume": self.rewards.get("volume"),
                "total_rewards.volume": self.total_rewards.get("volume"),
                "rewards.fee": self.rewards.get("fee"),
                "total_rewards.fee": self.total_rewards.get("fee"),
                "advanced_user": self.advanced_user,
                "total_fee_value_earned": self.total_fee_value_earned,
                "exp_fees_0": self._exp_fees_0,
                "exp_fees_1": self._exp_fees_1,
            }
        )

        return d

    @property
    def last_add_date(self):
        return self.updates.get("adds")[-1].get("date")

    @property
    def last_burn_date(self):
        last_burn = self.updates.get("burns")
        if last_burn:
            return last_burn[-1].get("date")
        else:
            return None

    @property
    def last_add_amount(self):
        return self.updates.get("adds")[-1].get("liquidity")

    @property
    def last_burn_amount(self):
        burns = self.updates.get("burns")
        if burns:
            return burns[-1].get("liquidity")
        else:
            return None

    @property
    def liquidity(self):

        # liquidity = self._get_liquidity() / (
        #     10
        #     ** (
        #         (
        #             self.pool.get_token(0).token_decimal
        #             + self.pool.get_token(1).token_decimal
        #         )
        #         / 2
        #     )
        # )
        return self._get_liquidity() / 10**18

    @property
    def width(self):
        return self.upr_price - self.lwr_price

    @property
    def in_range(self):
        return self.lwr_price <= self.pool.get_price() <= self.upr_price

    @property
    def inventory(self):
        # [x]: check inv scaling... using both token_1_dec
        price = self.pool.get_price()
        # I_x(p;L,p_min,p_max) = \sqrt{L}\left(\sqrt{\frac{1}{\min\left(\max\left(p,p_{min}\right),p_{max}\right)}}-\sqrt{\frac{1}{p_{max}}}\right)

        i_0 = self.liquidity * (
            1 / np.sqrt(min(max(price, self.lwr_price), self.upr_price))
            - 1 / np.sqrt(self.upr_price)
        )
        # I_{y}(p;L,p_min,p_max)=\sqrt{L}\left(\sqrt{\max\left(\min\left(p,p_{max}\right),p_{min}\right)}-\sqrt{p_{min}}\right)
        i_1 = self.liquidity * (
            np.sqrt(max(min(price, self.upr_price), self.lwr_price))
            - np.sqrt(self.lwr_price)
        )

        return [i_0, i_1]

    @property
    def inventory_value(self):
        return self.inventory_0 * self.pool.get_price() + self.inventory_1

    @property
    def fee_value(self):
        return self.fees_earned_0 * self.pool.get_price() + self.fees_earned_1

    @property
    def total_value(self):
        return self.inventory_value + self.fee_value

    @property
    def inventory_0(self):
        return self.inventory[0]

    @property
    def inventory_1(self):
        return self.inventory[1]

    @property
    def is_burned(self):
        return self.liquidity <= 0

    def _get_liquidity(self):
        # only need to store once

        if self.raw_pos_info:
            liq = self.raw_pos_info.liquidity
        else:
            liq = 0
        return liq

    def update(self):
        if not self.is_burned:
            self._num_steps_in_range += 1 if self.in_range else 0
        return

    def collect_fees(self):
        BIG_NUM = 2**64 - 2
        self.pool._pool.burn(self.owner, self.lwr_tick, self.upr_tick, 0)
        fee_data = self.pool._pool.collect(
            self.owner,
            self.lwr_tick,
            self.upr_tick,
            BIG_NUM,
            BIG_NUM,
        )
        x_fees = fee_data[-2]
        y_fees = fee_data[-1]

        self.fees_earned_0 += x_fees
        self.fees_earned_1 += y_fees

        return x_fees, y_fees

    def __repr__(self):
        return f"HumanPosition({self.to_dict()})"

    @property
    def num_adds(self):
        return len(self.updates.get("adds"))

    @property
    def num_burns(self):
        return len(self.updates.get("burns"))

    @property
    def liquidity_added(self):
        return sum([add.get("liquidity") for add in self.updates.get("adds")])

    @property
    def liquidity_burned(self):
        burns = self.updates.get("burns")
        if burns:
            ret = sum([burn.get("liquidity") for burn in self.updates.get("burns")])
        else:
            ret = 0

        return ret
