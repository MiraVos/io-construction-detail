# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:46:52 2024

@author: StefanoMerciai
"""

from dataclasses import dataclass
from config_file import Config_input
from pathlib import Path


@dataclass
class Paths:
    config = Config_input()
    if config.user == "SM":
        path_home = (
            Path.home()
            / "OneDrive - 2.-0 LCA Consultants ApS/Documenten/IIOA/CML/Mira/construction"
        )
        greenhouse = (
            path_home
            / "external_data/240214_LUCAS_LandCover_Greenhouses_lan_lcv_art__custom_9878113_spreadsheet.xlsx"
        )

    else:
        greenhouse = Path(
            "external_data/240214_LUCAS_LandCover_Greenhouses_lan_lcv_art__custom_9878113_spreadsheet.xlsx"
        )
