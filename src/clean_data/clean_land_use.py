# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:42:36 2024

@author: StefanoMerciai
"""
import pandas as pd

import transformation.data_elaborations as elab
from pathlib import Path
from paths_makers import Paths
from config_file import Config_input


def import_n_clean_lucas_greenhouse(config: Config_input, paths: Paths) -> pd.DataFrame:
    """import and clean the LUCA db on Greenhouses"""
    # LUCAS Land Cover and Land Use Statistics: Greenhouses
    lucas = {}
    lucas["Greenhouses"] = pd.read_excel(
        paths.greenhouse,
        sheet_name="Sheet 12",
        header=1,
        index_col=0,
        skiprows=7,
        skipfooter=7,
        na_values=[":", "bu", "b", "u"],
    )

    ###########################################
    # clean data
    ##########################################
    lucas["Greenhouses"].dropna(axis=1, how="all", inplace=True)
    lucas["Greenhouses"].dropna(axis=0, how="all", inplace=True)
    lucas_cols = pd.DataFrame(
        columns=[
            str(i)
            for i in range(
                int(lucas["Greenhouses"].columns.min()),
                int(lucas["Greenhouses"].columns.max()) + 1,
                1,
            )
        ]
    )
    lucas["Greenhouses"] = (
        pd.merge(lucas["Greenhouses"], lucas_cols, how="outer")
        .reindex(lucas_cols.columns, axis=1)
        .set_index(lucas["Greenhouses"].index)
        .astype(float)
    )
    lucas["Greenhouses"] = lucas["Greenhouses"].interpolate(
        method="linear", limit_direction="both", axis=1
    )

    # conversion to final database
    # INITIATE DATAFRAMES
    fd = pd.DataFrame(
        columns=[
            "Year",
            "Country",
            "Structure",
            "Value",
            "Value_source",
            "Value_process",
            "Unit",
        ]
    )
    lifetime_greenhouse = 25  # years, lifetime of a greenhouse, based on ecoinvent ##############################################
    fd = elab.add_structure(
        config,
        fd,
        lucas,
        ["Greenhouses"],
        "replacement",
        "m2",
        "EU LUCAS interpolated",
        1000000,
        lifetime_greenhouse,
    )
    fd = elab.add_structure(
        config,
        fd,
        lucas,
        ["Greenhouses"],
        "expansion",
        "m2",
        "EU LUCAS interpolated",
        1000000,
    )  # construction of new greenhouses

    return fd
