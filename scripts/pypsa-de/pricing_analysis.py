import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds 'scripts/' to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Adds repo root

import multiprocessing as mp
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from pypsa.descriptors import get_switchable_as_dense as as_dense
from tqdm import tqdm

logger = logging.getLogger(__name__)

date_format = "%Y-%m-%d %H:%M:%S"


def timestep_before(timestep, n):
    if timestep == "2013-01-01 00:00:00" or timestep == "2019-01-01 00:00:00":
        return timestep
    dt = datetime.strptime(timestep, date_format)
    timestep_b = (
        dt - pd.Timedelta(n.snapshot_weightings.generators.iloc[0], unit="h")
    ).strftime(date_format)
    return timestep_b


def supply_price_link(n, link, timestep, bus, co2_add_on=False, print_steps=False):
    buses = ["bus0", "bus1", "bus2", "bus3", "bus4"]
    cols = [
        col
        for col in n.links.columns
        if col.startswith("bus") and n.links.loc[link, col] != ""
    ]
    buses = buses[: len(cols)]
    loc_buses = n.links.loc[link, buses].tolist()
    j = loc_buses.index(bus)
    efficiency_link = (
        1 if j == 0 else n.links.loc[link, f"efficiency{j}" if j > 1 else "efficiency"]
    )
    supply_price = 0

    for i, loc_bus in enumerate(loc_buses):
        efficiency_i = (
            1
            if i == 0
            else n.links.loc[link, f"efficiency{i}" if i > 1 else "efficiency"]
        )

        if loc_bus == bus:
            # Cost for the specified bus
            price = n.buses_t.marginal_price.loc[timestep, n.links.loc[link, "bus0"]]
            mc = n.links.loc[link, "marginal_cost"]
            cost = (price + mc) / efficiency_link
            supply_price += cost
            if print_steps:
                logger.warning(
                    f"Cost at bus {loc_bus}: {cost} from price {price} + mc {mc} / efficiency {efficiency_link}"
                )
        elif i > 0:
            # Revenue / costs at other buses
            price = price = n.buses_t.marginal_price.loc[timestep, loc_bus]
            if (
                n.buses.loc[loc_bus, "carrier"] in ["co2"]
            ) & co2_add_on:  # "co2 stored"
                price += n.global_constraints.loc["CO2LimitUpstream"].mu
            rev = price * efficiency_i / efficiency_link
            supply_price -= rev
            if print_steps:
                logger.warning(
                    f"Revenue at bus {loc_bus}: {rev} from price {price} * efficiency {efficiency_i} / efficiency {efficiency_link}"
                )

    return supply_price


def demand_price_link(n, link, timestep, bus, co2_add_on=False, print_steps=False):  
    
    buses = ["bus0", "bus1", "bus2", "bus3", "bus4"]
    cols = [col for col in n.links.columns if col.startswith("bus") and n.links.loc[link, col] != ""]
    buses = buses[:len(cols)]
    loc_buses = n.links.loc[link, buses].tolist()
    demand_price = 0

    # calc demand price for demand at bus 0

    if bus == n.links.loc[link, "bus0"]:
        for i, loc_bus in enumerate(loc_buses):
            if i > 0:
                    # Revenue at other buses (other from bus0)
                    efficiency_i = 1 if i == 0 else as_dense(n, "Link", f"efficiency{i}" if i > 1 else 'efficiency').loc[timestep, link] # watch out for time dependent efficiencies of heat pumps: 
                    price = n.buses_t.marginal_price.loc[timestep, loc_bus]
                    if (n.buses.loc[loc_bus, "carrier"] in ["co2", "co2 stored"]) & co2_add_on:
                        price += n.global_constraints.loc["co2_limit_upstream-DE"].mu
                    rev = price * efficiency_i
                    if print_steps:
                        print(f"Revenue at bus {loc_bus}: {rev} from price {price} * efficiency {efficiency_i}")
                    demand_price += rev
     
        demand_price -= n.links.loc[link, 'marginal_cost']
        if print_steps:
            print(f"Marginal cost of {link}: {n.links.loc[link, 'marginal_cost']}")

    # calc for demand price of demand at other bus than bus 0 (e.g. methanolisation for electricity)
    else:
        for i, loc_bus in enumerate(loc_buses):
                if bus != loc_bus:
                    # Revenue at other buses (other from bus)
                    efficiency_i = -1 if i == 0 else as_dense(n, "Link", f"efficiency{i}" if i > 1 else 'efficiency').loc[timestep, link] # watch out for time dependent efficiencies of heat pumps: 
                    price = n.buses_t.marginal_price.loc[timestep, loc_bus]
                    if (n.buses.loc[loc_bus, "carrier"] in ["co2", "co2 stored"]) & co2_add_on:
                        price += n.global_constraints.loc["co2_limit_upstream-DE"].mu
                    rev = price * efficiency_i
                    if print_steps:
                        print(f"Revenue at bus {loc_bus}: {rev} from price {price} * efficiency {efficiency_i}")
                    demand_price += rev
                else:
                    bus_no = loc_buses.index(bus)
                    efficiency_bus = n.links.loc[link, f"efficiency{bus_no}"] * -1

        demand_price -= n.links.loc[link, 'marginal_cost']
        if print_steps:
            print(f"Marginal cost of {link}: {n.links.loc[link, 'marginal_cost']}")
        
        demand_price = demand_price / efficiency_bus # adjust for efficiency of the demand bus (only needed if it is not bus 0 bc for bus 0 the efficiencfy is 1)
        if print_steps:
            print(f"Adjust demand price by efficiency of {bus} = {efficiency_bus}")

    
    return demand_price


def get_supply_demand(n, buses, timestep, co2_add_on=False):

    if not isinstance(buses, list):
        buses = [buses]

    # Initialize DataFrames with the required columns
    supply = pd.DataFrame(
        columns=[
            "mc",
            "capex_add_on",
            "p_nom_opt",
            "p",
            "volume_bid",
            "mc_final",
            "carrier",
        ]
    )
    demand = pd.DataFrame(
        columns=["bidding_price", "p", "p_nom_opt", "volume_demand", "carrier"]
    )

    # Supply Calculation
    for gen in n.generators[n.generators.bus.isin(buses)].index:
        mc = n.generators.loc[gen].marginal_cost
        capex_add_on = -n.generators_t.mu_upper.loc[timestep, gen]
        p_nom_opt = n.generators.loc[gen].p_nom_opt
        p = n.generators_t.p.loc[timestep, gen]
        p_max_pu = (
            n.generators_t.p_max_pu.loc[timestep, gen]
            if gen in n.generators_t.p_max_pu.columns
            else n.generators.loc[gen].p_max_pu
        )
        volume_bid = p_nom_opt * p_max_pu
        mc_final = mc
        carrier = n.generators.loc[gen].carrier

        if n.generators.loc[gen].carrier in [
            "load-shedding",
            "load",
        ]:  # alter values for load shedding
            volume_bid = p
            if "load-shedding" in n.generators_t.marginal_cost_quadratic:
                mcq = n.generators_t.marginal_cost_quadratic.loc[timestep, gen]
            else: 
                mcq = n.generators.loc[gen].marginal_cost_quadratic
            mc = p * mcq + n.generators.loc[gen].marginal_cost
            mc_final = mc

        supply.loc[gen] = [
            mc,
            capex_add_on,
            p_nom_opt,
            p,
            volume_bid,
            mc_final,
            carrier,
        ]

    for su in n.storage_units[n.storage_units.bus.isin(buses)].index:
        mc = (
            n.storage_units.loc[su].marginal_cost
            + n.storage_units_t.mu_energy_balance.loc[timestep, su]
            * 1
            / n.storage_units.efficiency_dispatch.loc[su]
        )
        capex_add_on = -n.storage_units_t.mu_upper.loc[timestep, su]
        p_nom_opt = n.storage_units.loc[su].p_nom_opt
        p = n.storage_units_t.p_dispatch.loc[timestep, su]
        p_max_pu = n.storage_units.p_max_pu.loc[su]
        volume_bid = min(
            p_nom_opt * p_max_pu,
            n.storage_units_t.state_of_charge.loc[timestep_before(timestep, n), su],
        )  #  * n.storage_units.efficiency_dispatch.loc[su] is alread in p_nom_opt integrated
        mc_final = mc
        carrier = n.storage_units.loc[su].carrier

        supply.loc[su] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    for st in n.stores[n.stores.bus.isin(buses)].index:
        mc = (
            n.stores.loc[st].marginal_cost
            + n.stores_t.mu_energy_balance.loc[timestep, st]
        )
        capex_add_on = -n.stores_t.mu_upper.loc[timestep, st]
        p_nom_opt = n.stores.loc[st].e_nom_opt
        p = n.stores_t.p.loc[timestep, st]
        volume_bid = min(
            n.stores.e_max_pu.loc[st] * n.stores.e_nom_opt.loc[st],
            n.stores_t.e.loc[timestep, st],
        )
        mc_final = mc
        carrier = n.stores.loc[st].carrier

        supply.loc[st] = [mc, capex_add_on, p_nom_opt, p, volume_bid, mc_final, carrier]

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index:
        for bus in buses:
            # exclude if the bus is bus0 as there is no supply only demand for bus0
            if bus in n.links.loc[link, loc_buses[1:]].tolist():
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = (
                    1
                    if i == 0
                    else (
                        n.links.loc[link, "efficiency"]
                        if i == 1
                        else n.links.loc[link, f"efficiency{i}"]
                    )
                )
                bus_from = n.links.loc[link, "bus0"]
                # assumption that the marginal costs only apply to bus 1 (main output)
                mc = n.links.loc[
                    link
                ].marginal_cost  # - (n.links_t.mu_upper.loc[timestep, link] + n.links.loc[link].marginal_cost if i == 1 else 0)
                capex_add_on = -n.links_t.mu_upper.loc[timestep, link]
                p_nom_opt = n.links.loc[link].p_nom_opt * efficiency
                p = -n.links_t[f"p{i}"].loc[timestep, link]
                p_max_pu = (
                    n.links_t.p_max_pu.loc[timestep, link]
                    if link in n.links_t.p_max_pu.columns
                    else n.links.loc[link].p_max_pu
                )
                volume_bid = (
                    p_nom_opt * p_max_pu
                )  # efficiency is already included in p_nom_opt
                # cost from buying one unit of input energy (bus_from);
                mc_final = supply_price_link(
                    n, link, timestep, bus, co2_add_on
                )  # or: mc + supply_price_link_old(n, link, timestep, bus)
                carrier = n.links.loc[link].carrier

                supply.loc[link] = [
                    mc,
                    capex_add_on,
                    p_nom_opt,
                    p,
                    volume_bid,
                    mc_final,
                    carrier,
                ]

    loc_buses = ["bus0", "bus1"]
    for line in n.lines.index:
        for bus in buses:
            if bus in n.lines.loc[line, loc_buses].tolist():
                i = n.lines.loc[line, loc_buses].tolist().index(bus)
                bus_from = (
                    n.lines.loc[line, "bus0"]
                    if bus == n.lines.loc[line, "bus0"]
                    else n.lines.loc[line, "bus1"]
                )
                mc = n.buses_t.marginal_price.loc[timestep, bus_from]
                capex_add_on = -n.lines_t.mu_upper.loc[timestep, line]
                p_nom_opt = n.lines.loc[line].s_nom_opt
                p = -n.lines_t[f"p{i}"].loc[timestep, line]
                volume_bid = n.lines.s_max_pu.loc[line] * n.lines.s_nom_opt.loc[line]
                mc_final = mc
                carrier = n.lines.loc[line].carrier

                supply.loc[line] = [
                    mc,
                    capex_add_on,
                    p_nom_opt,
                    p,
                    volume_bid,
                    mc_final,
                    carrier,
                ]

    # Demand Calculation
    for load in n.loads[n.loads.bus.isin(buses)].index:
        bidding_price = np.inf
        p = n.loads_t.p.loc[timestep, load]
        p_nom_opt = p
        volume_demand = p
        carrier = n.loads.loc[load].carrier

        demand.loc[load] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    for su in n.storage_units[n.storage_units.bus.isin(buses)].index:
        bidding_price = (
            -n.storage_units_t.mu_upper.loc[timestep, su]
            + n.storage_units_t.mu_energy_balance.loc[timestep, su]
            * n.storage_units.efficiency_store.loc[su]
        )
        p = n.storage_units_t.p_store.loc[timestep, su]
        p_nom_opt = n.storage_units.loc[su].p_nom_opt
        storage_space = (
            n.storage_units.loc[su].max_hours * p_nom_opt
            - n.storage_units_t.state_of_charge.loc[timestep_before(timestep, n), su]
        )
        volume_demand = min(
            abs(p_nom_opt * n.storage_units.p_min_pu.loc[su]), storage_space
        )  # this value is negative  * n.storage_units.efficiency_store.loc[su] already integrated in p_nom_opt
        carrier = n.storage_units.loc[su].carrier

        demand.loc[su] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    for st in n.stores[n.stores.bus.isin(buses)].index:
        bidding_price = (
            -n.stores_t.mu_upper.loc[timestep, st]
            + n.stores_t.mu_energy_balance.loc[timestep, st]
        )
        p = -n.stores_t.p.loc[timestep, st]
        p_nom_opt = n.stores.loc[st].e_nom_opt
        volume_demand = abs(
            p_nom_opt * n.stores.e_nom_min.loc[st]
        )  # this value is negative
        carrier = n.stores.loc[st].carrier

        demand.loc[st] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    # only working for demand bid at bus0 ( are there links with several inputs and how to handle them? Yes, methanolisation)
    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index:
        for bus in buses:
            if (bus in n.links.loc[link, loc_buses].tolist()) & (
                bus == n.links.loc[link, "bus0"]
            ):
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = (
                    1
                    if i == 0
                    else (
                        n.links.loc[link, "efficiency"]
                        if i == 1
                        else n.links.loc[link, f"efficiency{i}"]
                    )
                )
                bidding_price = demand_price_link(n, link, timestep, bus, co2_add_on)
                p = n.links_t[f"p{i}"].loc[timestep, link]
                p_nom_opt = n.links.loc[link].p_nom_opt  # * efficiency
                p_min_pu = (
                    n.links_t.p_min_pu.loc[timestep, link]
                    if link in n.links_t.p_min_pu.columns
                    else n.links.loc[link].p_min_pu
                )
                if n.links.loc[link].carrier == "BEV charger":
                    volume_demand = min(
                        abs(p_nom_opt),
                        n.loads_t.p.loc[timestep, "DE0 0 land transport EV"],
                    )
                if n.links.loc[link].carrier == "urban decentral resistive heater":
                    volume_demand = min(
                        abs(p_nom_opt),
                        n.loads_t.p.loc[timestep, "DE0 0 urban decentral heat"]
                        / efficiency,
                    )
                if n.links.loc[link].carrier == "rural resistive heater":
                    volume_demand = min(
                        abs(p_nom_opt),
                        n.loads_t.p.loc[timestep, "DE0 0 rural heat"] / efficiency,
                    )
                else:
                    volume_demand = abs(p_nom_opt)
                carrier = n.links.loc[link].carrier
                demand.loc[link] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

            elif bus in n.links.loc[link, loc_buses].tolist():
                # if there is another bus, the efficiency has to be negative
                i = n.links.loc[link, loc_buses].tolist().index(bus)
                efficiency = (
                    1
                    if i == 0
                    else (
                        n.links.loc[link, "efficiency"]
                        if i == 1
                        else n.links.loc[link, f"efficiency{i}"]
                    )
                )
                if efficiency > 0:
                    continue
                bidding_price = demand_price_link(n, link, timestep, bus, co2_add_on)
                p = n.links_t[f"p{i}"].loc[timestep, link]
                p_nom_opt = abs(n.links.loc[link].p_nom_opt * efficiency)
                volume_demand = p_nom_opt
                carrier = n.links.loc[link].carrier
                demand.loc[link] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    loc_buses = ["bus0", "bus1"]
    for line in n.lines.index:
        for bus in buses:
            if bus in n.lines.loc[line, loc_buses].tolist():
                i = n.lines.loc[line, loc_buses].tolist().index(bus)
                bidding_price = n.buses_t.marginal_price.loc[timestep, bus]
                p = n.lines_t[f"p{i}"].loc[timestep, line]
                p_nom_opt = n.lines.loc[line].s_nom_opt
                volume_demand = p_nom_opt
                carrier = n.lines.loc[line].carrier

                demand.loc[line] = [bidding_price, p, p_nom_opt, volume_demand, carrier]

    return supply, demand


def get_compressed_demand(demand, th):
    demand.loc[:, "carrier"] = demand["carrier"].replace(
        {
            "electricity": "electricity load",
            "industry electricity": "electricity load",
            "agriculture electricity": "electricity load",
        }
    )
    df = demand.sort_values(by=["carrier", "bidding_price"]).reset_index(drop=True)
    group = (df["bidding_price"].diff().abs() > th).cumsum()
    df.loc[:, "group"] = group

    grouped_df = (
        df.groupby(["carrier", "group"])
        .agg(
            {
                "bidding_price": "mean",
                "p": "sum",
                "p_nom_opt": "sum",
                "volume_demand": "sum",
                "carrier": "first",
            }
        )
        .reset_index(drop=True)
    )

    return grouped_df


def price_setter(
    n, bus, timestep, supply=None, demand=None, minimum_generation=1e-3, co2_add_on=False, suppress_warnings=False
):
    mp = n.buses_t.marginal_price.loc[timestep, bus]
    if not isinstance(bus, list):
        bus = [bus]
    supply, demand = get_supply_demand(n, bus, timestep, co2_add_on)

    if supply is None or demand is None:
        supply, demand = get_supply_demand(n, bus, timestep, co2_add_on)
    
    # Filter where supply (p) is greater than 0.1 (using 1 omits the marginal generator at some times)
    th_p = minimum_generation
    drop_c_s = ["electricity distribution grid", "load", "BEV charger"]
    supply = supply[(supply.p > th_p) & ~(supply.carrier.isin(drop_c_s))]
    demand = demand[demand.p > th_p]

    # Find supply items where 'mc_final' is equal to the marginal price
    supply_closest = supply[supply["mc_final"] == mp]
    demand_closest = demand[demand["bidding_price"] == mp]

    # If no exact match, find the closest supply item to the marginal price
    if supply_closest.empty:
        closest_index = (supply["mc_final"] - mp).abs().argsort()[:1]
        supply_closest = supply.iloc[closest_index]
    else:
        closest_index = (supply["mc_final"] - mp).abs().argsort()[:1]
        supply_closest = pd.concat(
            [supply_closest, supply.iloc[closest_index]]
        ).drop_duplicates()

    if demand_closest.empty:
        closest_index = (demand["bidding_price"] - mp).abs().argsort()[:1]
        demand_closest = demand.iloc[closest_index]
    else:
        closest_index = (demand["bidding_price"] - mp).abs().argsort()[:1]
        demand_closest = pd.concat(
            [demand_closest, demand.iloc[closest_index]]
        ).drop_duplicates()

    sc = supply_closest.copy()
    sc["bus"] = bus[0]
    sc["timestep"] = timestep
    sc["supply_price"] = supply.mc_final[sc.index]
    sc["marginal price @ bus"] = mp
    sc["sp - mp"] = sc.loc[:, "supply_price"] - sc.loc[:, "marginal price @ bus"]
    sc["capacity_usage"] = sc.loc[:, "p"] / sc.loc[:, "volume_bid"]
    sc["valid"] = True
    msg_s = ""

    ### checks
    # diff of marginal price at bus and supply price
    if abs(sc["sp - mp"].values[0]) > 0.1:
        sc["valid"] = False
        msg_s = f"Warning: Supply price differs from market clearing price by {sc['sp - mp'].values[0]}; supply_price: {supply.mc_final[sc.index].iloc[0]}, marginal_price @ bus: {mp} (timestep {timestep}) \n"

    # check if capacity is used more than 0.99
    if sc["capacity_usage"].values[0] > 0.99:
        sc["valid"] = False
        msg_s += f"Warning: Marginal generator uses full capacity {sc['capacity_usage'].values[0]} (timestep {timestep}) \n"

    # check if generation is less than 1e-2
    if sc["capacity_usage"].values[0] < 1e-2:
        sc["valid"] = False
        msg_s += f"Warning: Marginal generator generates very low amount: amount = {sc.loc[:, 'p']}; capacity usage = {sc['capacity_usage'].values[0]} (timestep {timestep}) \n"

    # supply until marginal generator differs from real supply (with tolerance)
    # p_s = supply[supply.p > th_p].sort_values(by="mc_final", ascending=True)[:supply_closest.index[0]].p.sum()
    p_s = supply[(supply.p > th_p) & (supply.mc_final <= (mp + 0.1))].p.sum()
    p_s_true = n.statistics.supply(bus_carrier="AC", aggregate_time=False)[
        timestep
    ].sum()
    if abs(p_s - p_s_true) > 10:
        # sc["valid"] = False
        msg_s += f"Warning: Supply until marginal generator plus tolerance of {0.1} €/MWh does not match the total supply {p_s} != {p_s_true} (timestep {timestep}) \n"

    # check if mg is the one with the highest mc which is running (what is running?) with tolerance

    dc = demand_closest.copy()
    dc["bus"] = bus[0]
    dc["timestep"] = timestep
    dc["bidding_price"] = demand.bidding_price[dc.index]
    dc["marginal price @ bus"] = mp
    dc["bp - mp"] = dc.loc[:, "bidding_price"] - dc.loc[:, "marginal price @ bus"]
    dc["capacity_usage"] = dc.loc[:, "p"] / dc.loc[:, "volume_demand"]
    dc["valid"] = True
    msg_d = ""

    ### checks
    # diff of marginal price at bus and supply price
    if abs(dc["bp - mp"].values[0]) > 0.1:
        dc["valid"] = False
        msg_d = f"Warning: Demand price differs from market clearing price by {dc['bp - mp'].values[0]}; demand_price: {demand.bidding_price[dc.index].iloc[0]}, marginal_price @ bus: {mp} (timestep {timestep}) \n"

    # check if capacity is used more than 0.99
    if dc["capacity_usage"].values[0] > 0.99:
        dc["valid"] = False
        msg_d += f"Warning: Marginal consumer uses full capacity {dc['capacity_usage'].values[0]} (timestep {timestep})\n"

    # check if consumption is less than 1e-2
    if dc["capacity_usage"].values[0] < 1e-2:
        dc["valid"] = False
        msg_d += f"Warning: Marginal consumer consumes very low amount: amount = {dc.loc[:, 'p']}; capacity usage = {dc['capacity_usage'].values[0]} (timestep {timestep})\n"

    # demand until least price taker differs from real demand
    d_s = demand[(demand.p > th_p) & (demand.bidding_price >= (mp - 0.1))].p.sum()
    d_s_true = n.statistics.withdrawal(bus_carrier="AC", aggregate_time=False)[
        timestep
    ].sum()
    if abs(d_s - d_s_true) > 10:
        # dc["valid"] = False
        msg_d += f"Warning: Demand until least price taker minus tolerance of {0.1} €/MWh does not match the total demand {d_s} != {d_s_true} (timestep {timestep}) \n"

    if not suppress_warnings:
        if not (sc["valid"].any() or dc["valid"].any()):
            logger.warning(
                f"Warning: No valid price setting technology found for bus {bus} at timestep {timestep}"
            )
            logger.warning(msg_s)
            logger.warning(msg_d)

    # check if supply and demand are equal
    if abs(p_s - d_s) > 10:
        if not suppress_warnings:
            logger.info(
                f"Info: Supply until marginal gen ({p_s}) and demand until least price taker ({d_s})differs by {abs(p_s - d_s)} (timestep {timestep})"
            )

    return sc, dc


def get_all_supply_prices(n, bus, period=None, carriers=None):
    index = n.snapshots if period is None else period

    # Initialize a dictionary to store results temporarily
    res_dict = {ts: {} for ts in index}

    for gen in n.generators.index[
        (n.generators.bus == bus)
        & ((n.generators.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        marginal_cost = n.generators.loc[gen].marginal_cost
        for ts in index:
            res_dict[ts][gen] = marginal_cost

    for su in n.storage_units.index[
        (n.storage_units.bus == bus)
        & ((n.storage_units.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][su] = (
                n.storage_units.loc[su].marginal_cost
                + n.storage_units_t.mu_energy_balance.loc[ts, su]
                * 1
                / n.storage_units.efficiency_dispatch.loc[su]
            )

    for st in n.stores.index[
        (n.stores.bus == bus)
        & ((n.stores.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][st] = (
                n.stores.loc[st].marginal_cost
                + n.stores_t.mu_energy_balance.loc[ts, st]
            )

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index[
        (n.links.bus0 != bus)
        & (n.links[loc_buses].isin([bus]).any(axis=1))
        & ((n.links.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][link] = supply_price_link(n, link, ts, bus)

    # Convert the dictionary to a DataFrame
    res = pd.DataFrame.from_dict(res_dict, orient="index")

    return res


def get_all_demand_prices(n, bus, period=None, carriers=None):
    index = n.snapshots if period is None else period

    # Initialize a dictionary to store results temporarily
    res_dict = {ts: {} for ts in index}

    for su in n.storage_units.index[
        (n.storage_units.bus == bus)
        & ((n.storage_units.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][su] = (
                -n.storage_units_t.mu_upper.loc[ts, su]
                + n.storage_units_t.mu_energy_balance.loc[ts, su]
            )

    for st in n.stores.index[
        (n.stores.bus == bus)
        & ((n.stores.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][st] = (
                -n.stores_t.mu_upper.loc[ts, st]
                + n.stores_t.mu_energy_balance.loc[ts, st]
            )

    loc_buses = ["bus" + str(i) for i in np.arange(0, 5)]
    for link in n.links.index[
        (n.links[loc_buses].isin([bus]).any(axis=1))
        & ((n.links.carrier.isin(carriers)) if carriers is not None else True)
    ]:
        for ts in index:
            res_dict[ts][link] = demand_price_link(n, link, ts, bus)

    # Convert the dictionary to a DataFrame
    res = pd.DataFrame.from_dict(res_dict, orient="index")

    return res


### parellelisation methods ###

def process_bus_snapshot(args):
    """Process a single bus-snapshot combination for a specific network"""
    n, bus, snapshot, suppress_warnings = args
    
    # Get supply and demand outside of price_setter
    supply, demand = get_supply_demand(n, bus, str(snapshot))
    
    # Pass supply and demand to price_setter
    s, d = price_setter(
        n, 
        bus, 
        str(snapshot), 
        supply=supply, 
        demand=demand, 
        suppress_warnings=suppress_warnings
    )
    
    # Return the results including supply and demand for saving
    return (bus, str(snapshot), s, d, supply, demand)


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "pricing_analysis",
            simpl="",
            clusters=1,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="KN2045_Bal_v4_365H",
            planning_horizons="2025",
            lt_st="lt",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons
    nhours = int(snakemake.params.hours[:-1])
    nyears = nhours / 8760

    # Load data
    network = pypsa.Network(snakemake.input.network)
    investment_year = int(snakemake.wildcards.planning_horizons)

    # create results dict to collect all supply and demand data
    supply_all = {}
    demand_all = {}

    # create results df for price setting data
    res_s = pd.DataFrame()
    res_d = pd.DataFrame()

    # without parallelisation
    if not snakemake.params.pricing["parallel"]:
        
        logger.info("")  # Adds a blank line to the log file
        logger.info(f"Calculating price setter for year {investment_year}")

        for bus in network.buses.query("carrier == 'AC'").index:
            for snapshot in tqdm(network.snapshots.strftime(date_format), desc=f"Bus {bus}", leave=True):

                supply, demand = get_supply_demand(network, bus, snapshot)
                supply_all[snapshot] = supply
                demand_all[snapshot] = demand

                s, d = price_setter(network, bus, snapshot, supply=supply, demand=demand,  suppress_warnings=False)
                res_s = pd.concat([res_s, s])
                res_d = pd.concat([res_d, d])


    # with parallelisation
    else:

        # Use 'spawn' instead of 'fork' for better compatibility on cluster
        mp.set_start_method(snakemake.params.pricing["parallel_method"], force=True)
        nprocesses = snakemake.threads

        logger.info("")
        logger.info(f"Calculating price setter for year {investment_year}")

        # For each bus in the year
        for bus in network.buses.query("carrier == 'AC'").index:
            # Prepare arguments for each snapshot
            snapshot_args = [(network, bus, snapshot, False) for snapshot in network.snapshots]

            # Process all snapshots for this bus in parallel
            with mp.Pool(processes=nprocesses) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_bus_snapshot, snapshot_args),
                        total=len(snapshot_args),
                        desc=f"Bus {bus} in year {investment_year}",
                        unit="snapshots",
                    )
                )

            # Collect results for this bus
                for bus_result, snapshot_result, s, d, supply, demand in results:
                    res_s = pd.concat([res_s, s])
                    res_d = pd.concat([res_d, d])
                    
                    # Store supply and demand data
                    supply_all[snapshot_result] = supply
                    demand_all[snapshot_result] = demand
   

    # Save results
    with open(snakemake.output.supply, "wb") as file:
        pickle.dump(supply_all, file)
    with open(snakemake.output.demand, "wb") as file:
        pickle.dump(demand_all, file)

    with open(snakemake.output.price_setter_s, "wb") as file:
        pickle.dump(res_s, file)
    with open(snakemake.output.price_setter_d, "wb") as file:
        pickle.dump(res_d, file)
