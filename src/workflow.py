# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:23:59 2024

@author: StefanoMerciai
"""
import pandas as pd

import transformation.data_elaborations as elab

import clean_data.clean_land_use as clean_lu
from pathlib import Path
from paths_makers import Paths
from config_file import Config_input


###############################
# config Path
###############################
paths = Paths()
config = Config_input()


##############################
# tasks
##############################

fd = clean_lu.import_n_clean_lucas_greenhouse(config, paths)
