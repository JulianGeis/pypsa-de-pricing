# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.
"""

import logging
import os
import sys

import numpy as np
import pypsa
from _benchmark import memory_logger

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds 'scripts/' to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Adds repo root

from solve_network import prepare_network, solve_network

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_network_myopic",
            simpl="",
            clusters=1,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2045",
            run="KN2045_Bal_v4_voll",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    logger.info("Solving again with fixed capacities")

    solve_opts = snakemake.params.options
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    # # co2 constraint can become infeasible bc of numerical issues (2 ways to handle it)
    
    # # (1) round co2 constraint to make feasible (rounding with <= softens the constraint)
    # # n.global_constraints.loc["CO2Limit" , "constant"] = round(n.global_constraints.loc["CO2Limit" , "constant"])

    # # (2) multiply co2 store e_nom_opt by 2
    # n.stores.loc[n.stores.carrier == "co2", "e_nom_opt"] *= 2

    n.optimize.fix_optimal_capacities()

    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
        snakemake=snakemake,
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            planning_horizons=planning_horizons,
            rule_name=snakemake.rule,
            snakemake=snakemake,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)
