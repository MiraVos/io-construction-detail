For use for GVCtools, please save this folder as
commondata\Data_histdb\ICIO2021andExt_year2018

# OECD ICIO data for 2018, from OECDICIO Edition 2021
aggregated MEX and CHN
see ReadMe_ICIO2021_R.xlsx for more information

# Extensions data in F
names(F)
  [1] "VA"                       OECD ICIO                                              
  [2] "PROD_CO2"                 OECD ICIO                                              
  [3] "EMPN"                     OECD ICIO                                              
  [4] "GHG"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                             
  [5] "CO2"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
  [6] "CH4"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
  [7] "N2O"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
  [8] "CO"                       SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
  [9] "NMVOC"                    SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [10] "NOX"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [11] "SOX"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [12] "NH3"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [13] "PM2_5"                    SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [14] "PM10"                     SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [15] "BC"                       SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [16] "OC"                       SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [17] "O3PR"                     SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [18] "ACG"                      SINTEF estimates based on SEEA Air Emission Accounts from Eurostat & OECD and EDGAR data for remaining countries                                              
 [19] "EMP"                      SINTEF estimates based on ILO LabourForce Statistics Employment by sex and economic activity https://db.nomics.world/ILO/EMP_TEMP_SEX_ECO_NB                                               
 [20] "EMP_M"                    SINTEF estimates based on ILO LabourForce Statistics Employment by sex and economic activity https://db.nomics.world/ILO/EMP_TEMP_SEX_ECO_NB                                                                                             
 [21] "EMP_F"                    SINTEF estimates based on ILO LabourForce Statistics Employment by sex and economic activity https://db.nomics.world/ILO/EMP_TEMP_SEX_ECO_NB                                                                                             
 [22] "GLORIA_Emissions_c4f8"    GLORIA_SatelliteAccounts_055_2018  converted to ICIO classification by SINTEF                                            
 [23] "GLORIA_Emissions_c2f6"    GLORIA_SatelliteAccounts_055_2018  converted to ICIO classification by SINTEF
  ...
 [180] "GLORIA_Blue_water_consumption_Non-agriculture blue water consumption"  

# Intensities in S
S = F/x

i.e. also for Gloria extensions in F, we have calculated the intensities in S by dividing F with exonomic output x from OECD ICIO