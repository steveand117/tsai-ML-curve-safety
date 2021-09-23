if __name__ == '__main__':
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from utils.BBI_utils import mergeData, computeBBI, simpleLowpassFilter, getSuperelevation, getAdvisorySpeed, \
        alignData
    from utils.LRS_utils import NCATgetReferencePoints, getDistToMid, getReferenceCurve, NCATgetRadius
    from utils.NCAT_processing import NCAT_processing
    from utils.SR_processing import SR_processing

    loc = pd.read_csv(r'2021_03_11_07_36_21_506_loc.csv')
    acc = pd.read_csv(r'2021_03_11_07_36_21_506_acc.csv')
    crash = pd.read_csv(r'Crashdata.csv')
    # Chooses certain columns
    crash = crash.filter(items=['Road_Name', 'KABCO_Seve', 'Manner_of_', 'Location_a', 'Latitude', 'Longitude'])
    # Filters out all collision based crashes
    crash = crash[crash['Manner_of_'].eq('Not a Collision with Motor Vehicle')]
    # Filters out intersection crashes
    crash = crash[crash['Location_a'].str.contains('Non-Intersection') | crash['Location_a'].eq('Off Roadway')]
    print(crash)
    print(crash.shape)
    SR_obj = SR_processing(inFiles=[r'2021_03_11_07_36_21_506_loc.csv',r'2021_03_11_07_36_21_506_acc.csv'])
    SR_obj.gdf.to_file("smartphone.shp")
    print(acc.columns)
    print(loc.columns)
    print(loc)
    # Not sure why this merge doesn't work...
    # all_csv_data = pd.merge(left=loc, right=acc, how='left', left_on='timestamp_utc_local', right_on='timestamp_nanosecond')
    all_csv_data = pd.concat([loc, acc], axis=1)
    print(all_csv_data)
    road_17 = gpd.read_file('0017_D1_2/0017_D1_2.shp')
    # print(SR_obj.gdf)