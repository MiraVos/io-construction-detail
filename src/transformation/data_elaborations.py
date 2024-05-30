# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:22:00 2024

@author: StefanoMerciai
"""
import pandas as pd
import country_converter as coco

from config_file import Config_input


def add_structure(
    config: Config_input,
    fd,
    df,
    structure_labels,
    process,
    unit,
    source,
    scalar=1,
    lifetime=1,
):
    fd_new = pd.DataFrame(columns=fd.columns)

    for i in structure_labels:
        if process == "expansion":
            df_flat = df[i].diff(axis=1).dropna(axis=1, how="all")
        elif (
            process == "replacement"
        ):  # assumes constant linear replacement of existing stock and linear construction
            df_flat = df[i] / lifetime
        else:
            df_flat = df[i]
        df_flat.index = coco.convert(names=df_flat.index, to="ISO3")
        if config.only_eu:
            eu_labels = coco.CountryConverter().EU27as("ISO3").ISO3.to_list()
            df_flat = df_flat[df_flat.index.isin(eu_labels)]
        df_flat = df_flat.unstack().reset_index()
        df_flat.columns = ["Year", "Country", "Value"]
        df_flat = df_flat[df_flat["Value"].notna()]
        df_flat.Year = df_flat.Year.astype(str)
        df_flat["Structure"] = i
        fd_new = pd.merge(df_flat, fd_new, how="outer")

    fd_new["Value"] = fd_new["Value"] * scalar
    fd_new["Value_source"] = source
    fd_new["Value_process"] = process
    fd_new["Unit"] = unit
    fd = pd.concat([fd, fd_new])

    return fd
