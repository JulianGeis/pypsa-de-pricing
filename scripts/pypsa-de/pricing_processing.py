import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds 'scripts/' to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Adds repo root

import pickle

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

date_format = "%Y-%m-%d %H:%M:%S"


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "pricing_processing",
            simpl="",
            clusters=1,
            opts="",
            ll="vopt",
            sector_opts="none",
            lt_st="lt",
            run="KN2045_Bal_v4_365H",
        )

    configure_logging(snakemake)
    config = snakemake.config
    planning_horizons = snakemake.params.planning_horizons

    # Load networks
    networks = {year: pypsa.Network(fn) for year, fn in zip(planning_horizons, snakemake.input.networks)}

    # Load price setting data
    results_s = {}
    results_d = {}
    for year, file_s, file_d in zip(planning_horizons, snakemake.input.price_setter_s_all, snakemake.input.price_setter_d_all):
        with open(file_s, "rb") as file:
            results_s[year] = pickle.load(file)
        with open(file_d, "rb") as file:
            results_d[year] = pickle.load(file)

    # convert to datetime
    for year in planning_horizons:
        results_s[year].timestep = pd.to_datetime(
            results_s[year].timestep, format=date_format
        )
        results_d[year].timestep = pd.to_datetime(
            results_d[year].timestep, format=date_format
        )

    # save combined price setter files
    with open(snakemake.output.price_setter_s_all, "wb") as file:
        pickle.dump(results_s, file)
    with open(snakemake.output.price_setter_d_all, "wb") as file:
        pickle.dump(results_d, file)

    # Load supply and demand data
    supply_all = {}
    demand_all = {}
    for year, file_s, file_d in zip(planning_horizons, snakemake.input.supply_all, snakemake.input.demand_all):
        with open(file_s, "rb") as file:
            supply_all[year] = pickle.load(file)
        with open(file_d, "rb") as file:
            demand_all[year] = pickle.load(file)

    # calc bid / ask
    bid = {}
    ask = {}

    for year in np.arange(2020,2050,5):
        n = networks[year]

        bid_year = pd.DataFrame(columns=supply_all[year][str(n.snapshots[0])].index)
        ask_year = pd.DataFrame(columns=demand_all[year][str(n.snapshots[0])].index.drop(['DE0 0', 'DE0 0 industry electricity', 'DE0 0 agriculture electricity']))

        for sn in n.snapshots:
            bid_year.loc[str(sn)] = supply_all[year][str(sn)]["mc_final"] 
            ask_year.loc[str(sn)] = demand_all[year][str(sn)]["bidding_price"]

        bid_year.index = pd.to_datetime(list(bid_year.index))
        ask_year.index = pd.to_datetime(list(ask_year.index))

        bid[year] = bid_year
        ask[year] = ask_year

    # save combined bid / ask
    with open(snakemake.output.bid, "wb") as file:
        pickle.dump(bid, file)
    with open(snakemake.output.ask, "wb") as file:
        pickle.dump(ask, file)

    # calc mapped bid / ask
    mapped_bid = {}
    mapped_ask = {}

    for year in  np.arange(2020,2050,5):
        n = networks[year]
        carrier_mapping = pd.concat([
            n.generators.carrier,
            n.storage_units.carrier,
            n.links.carrier,
            n.stores.carrier
        ])

        carrier_mapping = carrier_mapping[~carrier_mapping.index.duplicated(keep='first')]
        mapped_bid[year] = bid[year].copy()
        mapped_bid[year].columns = bid[year].columns.to_series().map(carrier_mapping)
        mapped_ask[year] = ask[year].copy()
        mapped_ask[year].columns = ask[year].columns.to_series().map(carrier_mapping)

    # save combined mapped bid / ask
    with open(snakemake.output.mapped_bid, "wb") as file:
        pickle.dump(mapped_bid, file)
    with open(snakemake.output.mapped_ask, "wb") as file:
        pickle.dump(mapped_ask, file)

    # price setting prioritization
    results_s_unique = {}
    results_d_unique = {}

    # ranked by marginal cost / bidding price in 2045
    supply_candidates_generators_ranking = ['ror', 'solar', 'solar-hsat', 'onwind',  'offwind-ac', 'offwind-dc', ]
    # ranked by  bidding price in 2020 (has no impact as they are never competing candidates)
    supply_candidates_constant_links = ["nuclear", "lignite", "coal", "CCGT", "OCGT", "biogas", "solid biomass","oil"]

    logger.info("Price setter prioritization:")
    
    for year in planning_horizons:

        df_s = results_s[year].copy()
        df_d = results_d[year].copy()

        # Drop duplicates in timestep and carrier and keep plant with highest p
        df_s = df_s.sort_values("p", ascending=False)
        df_d = df_d.sort_values("p", ascending=False)
        df_s = df_s.drop_duplicates(subset=["timestep", "carrier"], keep="first").reset_index(drop=True)
        df_d = df_d.drop_duplicates(subset=["timestep", "carrier"], keep="first").reset_index(drop=True)

        # candidate carriers
        supply_carriers = results_s[year].carrier.unique()
        demand_carriers = results_d[year].carrier.unique()
        
        # Specify prioritization
        manual_ranking = supply_candidates_generators_ranking + supply_candidates_constant_links
        manual_ranking = [item for item in manual_ranking if item in supply_carriers]
        var_ranking = list(mapped_bid[year].T.groupby(level=0).mean().var(axis=1)[supply_carriers].sort_values().index)
        var_ranking = [item for item in var_ranking if item not in manual_ranking]
        supply_ranking = manual_ranking + var_ranking
        supply_ranking_dict = {tech: i for i, tech in enumerate(supply_ranking)}
        
        demand_ranking = list(mapped_ask[year].T.groupby(level=0).mean().var(axis=1)[demand_carriers].sort_values().index)
        demand_ranking_dict = {tech: i for i, tech in enumerate(demand_ranking)}

        logger.info(f"Year {year}")
        logger.info("Supply ranking")
        logger.info(supply_ranking)
        logger.info("Demand ranking")
        logger.info(demand_ranking)
    

        # Choose price setter according to prioritization
        df_s["priority"] = df_s["carrier"].map(supply_ranking_dict).fillna(len(supply_ranking_dict))
        df_d["priority"] = df_d["carrier"].map(demand_ranking_dict).fillna(len(demand_ranking_dict))
        df_s = df_s.sort_values("priority")
        df_d = df_d.sort_values("priority")
        df_s = df_s.drop_duplicates(subset=["timestep"], keep="first").reset_index(drop=True)
        df_d = df_d.drop_duplicates(subset=["timestep"], keep="first").reset_index(drop=True)

        # set demand price setting to invalid if a supply price setter is valid in the same timestep
        valid_supply_ts = df_s[df_s["valid"] == True]["timestep"].unique()
        df_d.loc[df_d["timestep"].isin(valid_supply_ts), "valid"] = False

        # sort according to timestep
        df_s = df_s.sort_values("timestep")
        df_d = df_d.sort_values("timestep")

        results_s_unique[year] = df_s.copy()
        results_d_unique[year] = df_d.copy()

        # save combined mapped bid / ask
        with open(snakemake.output.price_setter_s, "wb") as file:
            pickle.dump(results_s_unique, file)
        with open(snakemake.output.price_setter_d, "wb") as file:
            pickle.dump(results_d_unique, file)