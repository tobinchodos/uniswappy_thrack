from humanposition import HumanPosition
from pool import UniswapPool
import pandas as pd


class Env:

    def __init__(self, config):
        self.config = config
        self._fake_address_counter = 0
        self._address_to_obj = {}
        self.tokens = {}
        self.pools = {}
        self.metrics = []
        # all this is pretty much unnecessary now
        for token_str, (token_cls, token_kwargs) in config.get("tokens").items():
            fake_address = self.get_fake_address()
            token_kwargs.update(
                {
                    "name": token_kwargs.get("name", token_str),
                    "addr": fake_address,
                    "decimal": token_kwargs.get("decimal"),
                }
            )
            token = token_cls(**token_kwargs)
            self.tokens.update({token_str: token})
            self.register(fake_address, token)

        for pool_str, (pool_cls, pool_kwargs) in config.get("pools").items():
            fake_address = self.get_fake_address()
            pool = pool_cls(
                pool_str=pool_str, address=fake_address, env=self, **pool_kwargs
            )
            self.pools.update({pool_str: pool})
            self.register(fake_address, pool)

    def reset(self, seed=None):
        # not necessary nwo
        for pool in self.pools.values():
            pool.reset(seed=seed)
        self.step_count = 0

    def get_fake_address(self):
        # not even necessary now
        fake_address = hex(self._fake_address_counter)
        self._fake_address_counter += 1
        return fake_address

    def register(self, address, obj):
        # not even necessary now
        address = address or self.get_fake_address()
        self._address_to_obj.update({address: obj})
        return address


class DataDrivenEnv(Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log(self, message):
        # make better
        if not message.startswith("STEP"):
            message = "\t" + message
        else:
            message = "\n" + message
        print(message)

    def step(self, **kwargs):
        orig_kwargs = action_dict = kwargs.get("action_dict")
        pool_id = action_dict.get("pool_id")
        self.pre_step(ctx=orig_kwargs)

        match action_dict.get("event"):
            case "mint":
                if pool_id in self.pools:
                    kwargs = {
                        "lp_id": action_dict.get("lp_id"),
                        "token_in": (
                            action_dict.get("token_0_name")
                            if action_dict.get("amount_0") > 0
                            else action_dict.get("token_1_name")
                        ),  # not needed anymore
                        "amount_in": (
                            action_dict.get("amount_0")
                            if action_dict.get("amount_0") > 0
                            else action_dict.get("amount_1")
                        ),  # not needed anymore
                        "lwr_tick": action_dict.get("lwr_tick"),
                        "upr_tick": action_dict.get("upr_tick"),
                        "liquidity": action_dict.get("amount_lp_token"),
                        "date": action_dict.get("date"),
                        "advanced_user": action_dict.get("advanced_user"),
                    }
                    out = self.mint(
                        pool_id=pool_id, kwargs=kwargs, orig_kwargs=orig_kwargs
                    )

                else:

                    kwargs = {
                        "env": self,
                        "pool_id": action_dict.get("pool_id"),
                        "lp_id": action_dict.get("lp_id"),
                        "token_0": action_dict.get("token_0_name"),
                        "amount_0": action_dict.get("amount_0"),  # not needed anymore
                        "token_1": action_dict.get("token_1_name"),
                        "amount_1": action_dict.get("amount_1"),  # not needed anymore
                        "lwr_tick": action_dict.get("lwr_tick"),
                        "upr_tick": action_dict.get("upr_tick"),
                        "fee": action_dict.get("fee"),
                        "sqrt_price_X96": action_dict.get(
                            "sqrt_price_X96"
                        ),  ## INIT PRICE SET FROM THIS
                        "liquidity": action_dict.get(
                            "amount_lp_token"
                        ),  # INIT LIQ DEPOSIT SET FROM THIS
                        "date": action_dict.get("date"),
                        "advanced_user": action_dict.get("advanced_user"),
                    }
                    out = self.create_pool(
                        pool_id=pool_id, kwargs=kwargs, orig_kwargs=orig_kwargs
                    )

            case "burn":
                assert pool_id in self.pools, "make sure first a mint happened first"
                kwargs = {
                    "lp_id": action_dict.get("lp_id"),
                    "lwr_tick": action_dict.get("lwr_tick"),
                    "upr_tick": action_dict.get("upr_tick"),
                    "liquidity_amount": -action_dict.get("amount_lp_token"),
                    "date": action_dict.get("date"),
                }
                out = self.burn(pool_id=pool_id, kwargs=kwargs, orig_kwargs=orig_kwargs)

            case "swap":
                assert pool_id in self.pools, "make sure a mint happened first"

                kwargs = {
                    "trader_id": action_dict.get("trader"),
                    "token_in": (
                        action_dict.get("token_0_name")
                        if action_dict.get("amount_0") > 0
                        else action_dict.get("token_1_name")
                    ),
                    "amount_in": (
                        action_dict.get("amount_0")
                        if action_dict.get("amount_0") > 0
                        else action_dict.get("amount_1")
                    ),
                }
                out = self.swap(pool_id=pool_id, kwargs=kwargs, orig_kwargs=orig_kwargs)
        out = self.post_step(
            out, ctx=dict(pool_id=pool_id, kwargs=kwargs, orig_kwargs=orig_kwargs)
        )
        return out

    ### MAKE ALL THESE WRAPPERS ACTUAL WRAPPERS OR SMTH EASIER

    def pre_step(self, ctx):

        message = (
            "STEP {}, ".format(self.step_count)
            + f"PRE_STEP: trying event={ctx.get('event').upper()} on pool_id={ctx.get('pool_id')} with ctx={ctx}"
        )
        self.log(message)
        return

    def post_step(self, out, ctx):

        message = "POST_STEP: expected v actual"
        self.log(message)
        expected_price = ctx.get("orig_kwargs").get("pool_price")
        actual_price = self.pools.get(ctx.get("pool_id")).get_price()
        message = f"expected_price={expected_price}, actual_price={actual_price}"
        self.log(message)
        expected_sqrtpriceX96 = ctx.get("orig_kwargs").get("sqrt_price_X96")
        actual_sqrtpriceX96 = self.pools.get(
            ctx.get("pool_id")
        )._pool.slot0.sqrtPriceX96
        message = f"expected_sqrtpriceX96={expected_sqrtpriceX96}, actual_sqrtpriceX96={actual_sqrtpriceX96}"
        self.log(message)
        expected_liquidity = ctx.get("orig_kwargs").get("liquidity_at_tick")
        actual_liquidity = self.pools.get(ctx.get("pool_id")).liquidity_at_price()
        message = f"expected_liquidity={expected_liquidity}, actual_liquidity={actual_liquidity}"
        self.log(message)
        expected_total_liquidity = ctx.get("orig_kwargs").get("total_liquidity")
        actual_total_liquidity = self.pools.get(ctx.get("pool_id"))._pool.total_supply
        message = f"expected_total_liquidity={expected_total_liquidity}, actual_liquidity={actual_total_liquidity}"
        self.log(message)

        pool = self.pools.get(ctx.get("pool_id"))

        for pos in pool.positions.values():
            pos.update()  # set is_burned
        # self.collect_metrics(ctx)
        self.last_ctx = ctx
        self.step_count += 1
        return out

    def create_pool(self, **kwargs):
        self.pre_create_pool(kwargs)
        pool = UniswapPool(**kwargs.get("kwargs"))
        pool.reset()
        out = self.pools.update({pool.pool_id: pool})
        out = self.post_create_pool(out, kwargs)
        return out

    def pre_create_pool(self, ctx):
        message = "creating pool_id={pool_id} with {kwargs}".format(**ctx)
        self.log(message)

    def post_create_pool(self, out, ctx):
        pool_id = ctx.get("pool_id")
        kwargs = ctx.get("kwargs")
        message = f"pool created: {self.pools.get(pool_id).get_state()}"
        self.log(message)
        return out

    def mint(self, **kwargs):
        self.pre_mint(kwargs)
        out = self.pools.get(kwargs.get("pool_id")).add_liquidity_position(
            **kwargs.get("kwargs")
        )
        out = self.post_mint(out, kwargs)
        return out

    def pre_mint(self, ctx):
        print(ctx.get("kwargs"))
        return

    def post_mint(self, out, ctx):
        if isinstance(out, HumanPosition):
            self.log(f"Position Created: {out.to_dict()}")
            self.log(
                f"expected amount_0={ctx.get('orig_kwargs').get('amount_0')}, amount_1={ctx.get('orig_kwargs').get('amount_1')}"
            )
            self.log(f"actual amount_0={out.deposit_0}, amount_1={out.deposit_1}")
        return out

    def swap(self, **kwargs):
        self.pre_swap(kwargs)
        out = self.pools.get(kwargs.get("pool_id")).swap(**kwargs.get("kwargs"))
        out = self.post_swap(out, kwargs)
        return out

    def pre_swap(self, ctx):
        pass

    def post_swap(self, out, ctx):

        # pool_id = ctx.get("pool_id")
        kwargs = ctx.get("kwargs")
        orig_kwargs = ctx.get("orig_kwargs")
        expected_amount_out = (
            abs(orig_kwargs.get("amount_0"))
            if kwargs.get("token_in") == "Y"
            else abs(orig_kwargs.get("amount_1"))
        )
        actual_amount_out = out
        self.log(
            f"expected amount_out = {expected_amount_out}, actual amount out = {actual_amount_out}"
        )
        self.pools.get("uni_0").disburse_rewards()
        return out

    def burn(self, **kwargs):
        self.pre_burn(kwargs)
        out = self.pools.get(kwargs.get("pool_id")).burn(**kwargs.get("kwargs"))
        out = self.post_burn(out, ctx=kwargs)
        return out

    def pre_burn(self, ctx):
        pass

    def post_burn(self, out, ctx):
        self.log(
            f"expected amount_0={ctx.get('orig_kwargs').get('amount_0')}, amount_1={ctx.get('orig_kwargs').get('amount_1')}"
        )
        self.log(f"actual amount_0={out[0]}, amount_1={out[1]}")

        return out

    def collect_metrics(self, ctx=None):
        if ctx is None:
            ctx = self.last_ctx
        # this sucks but whatever.. trying to make this work for multiple pools with this cross-sectional kind of data..
        date = ctx.get("orig_kwargs").get("date")
        event = ctx.get("orig_kwargs").get("event")
        states = {}
        global_positions = []

        #### test
        for pos in self.pools.get("uni_0").positions.values():

            pos.collect_fees()

        ####

        for pool_id, pool in self.pools.items():
            state = pool.get_state()
            global_positions.extend(state.pop("positions"))
            states.update({pool_id: state})

        global_info_dict = {}
        for pool_id, state in states.items():
            global_info_dict.update(
                {
                    (
                        "synth_pool"
                        + f"{('.' + pool_id) if len(self.pools)>1 else ''}"
                        + f".{k}"
                    ): v
                    for k, v in state.items()
                }
            )
            global_info_dict.update(
                {
                    "true_pool.price": ctx.get("orig_kwargs").get("pool_price"),
                    "true_pool.liquidity_at_tick": ctx.get("orig_kwargs").get(
                        "liquidity_at_tick"
                    ),
                    "true_pool.total_liquidity": ctx.get("orig_kwargs").get(
                        "total_liquidity"
                    ),
                    "synth_pool.liq_at_true_tick": (
                        ctx.get("orig_kwargs").get("pool_price")
                        if pd.isna(ctx.get("orig_kwargs").get("pool_price"))
                        else self.pools.get(pool_id).liquidity_at_price(
                            price=ctx.get("orig_kwargs").get("pool_price")
                        )
                    ),
                }
            )
            global_info_dict.update(
                {"date": date, "event": event, "step": self.step_count}
            )

        for position in global_positions:
            position.update(global_info_dict)

        self.metrics.extend(global_positions)
