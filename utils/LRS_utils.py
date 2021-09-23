# LRS utils
import geopandas
import numpy as np
import pandas as pd


def NCATgetRadius(dist_to_mid):
    """ 
    NCAT Curves are in sprial-constant-sprial config.
    The constant radius (Rc) is 476'
    The inital radius (max radius on spiral) is about 1265 ft
    """
    # Total Sprial Length (get from distance between C.S.and S.T. stations)
    CS = 543.7
    ST = 951.7
    l_s = ST-CS
    Rc = 476
    # offest from centerline to center of the lane
    offset = -6
    dist_to_mid = np.abs(dist_to_mid)
    if  dist_to_mid < CS:
        R = Rc + offset
    elif dist_to_mid < ST:
        R = min(1000000, l_s/(ST-dist_to_mid)*Rc + offset)
    else:
        R = 1000000 # Assume a large number for radius on tengent
    return float(R)

def GYROgetRadius(speed, gyro, flip_sign = False):
    """
    Return radius in ft
    """
    # Use 10^-9 to aviod zero gyro or radius results
    if abs(gyro) < 10**(-9):
        gyro = 10**(-9)
    R = speed/gyro
    if R > 0:
        R = max(10**(-9), min(1000000, R))
    else:
        R = min(-10**(-9), max(-1000000, R))
    
    if flip_sign:
        R = -R

    return float(R*3.28084)



def NCATgetReferencePoints(track_center_line,mid_point_E,mid_point_W):
    # Get linear reference distance (LRD) of two curve mid-points
    # Get the half-way point between two curve mid-points (south tangent mid-point)
    # Subtract half track distance to get north tangent mid-point
    track_length = track_center_line.length
    mid_point_E_dist = track_center_line.project(mid_point_E)
    mid_point_W_dist = track_center_line.project(mid_point_W)
    mid_ST_dist = (mid_point_E_dist + mid_point_W_dist)/2
    mid_NT_dist = mid_ST_dist - track_length/2
    referencePoints = np.array(
        [mid_point_E_dist, mid_point_W_dist, mid_ST_dist, mid_NT_dist])
    return referencePoints


def getDistToMid(track_dist, track_length, referencePoints):
    # Get distance to mid-point base which half (W or E) part of the track the point is on
    [mid_point_E_dist, mid_point_W_dist, mid_ST_dist, mid_NT_dist] = referencePoints
    if mid_NT_dist < mid_ST_dist: # if track start point is before North Tangent Mid Point
        if track_dist >= mid_NT_dist and track_dist <= mid_ST_dist:
            dist = track_dist-mid_point_W_dist
        elif track_dist >= 0 and track_dist < mid_NT_dist:
            dist = track_dist+track_length-mid_point_E_dist
        elif track_dist > mid_ST_dist:
            dist = track_dist-mid_point_E_dist
    else:  # if track start point is after North Tangent Mid Point
        if track_dist >= mid_ST_dist and track_dist <= mid_NT_dist:
            dist = track_dist-mid_point_E_dist
        elif track_dist >= 0 and track_dist < mid_ST_dist:
            dist = track_dist+track_length-mid_point_W_dist
        elif track_dist > mid_NT_dist:
            dist = track_dist-mid_point_W_dist

    return dist


def getReferenceCurve(track_dist, referencePoints):
    # Get distance to mid-point base which half (W or E) part of the track the point is on
    [mid_point_E_dist, mid_point_W_dist, mid_ST_dist, mid_NT_dist] = referencePoints
    if mid_NT_dist < mid_ST_dist:  # if track start point is before North Tangent Mid Point
        if track_dist >= mid_NT_dist and track_dist <= mid_ST_dist:
            reference_curve = 'West Curve'
        elif track_dist >= 0 and track_dist < mid_NT_dist:
            reference_curve = 'East Curve'
        elif track_dist > mid_ST_dist:
            reference_curve = 'East Curve'
    else:  # if track start point is after North Tangent Mid Point
        if track_dist >= mid_ST_dist and track_dist <= mid_NT_dist:
            reference_curve = 'East Curve'
        elif track_dist >= 0 and track_dist < mid_ST_dist:
            reference_curve = 'West Curve'
        elif track_dist > mid_NT_dist:
            reference_curve = 'West Curve'

    return reference_curve

    
def getRangeIndex(df_in, dist_label, reference_point):
    """ 
    return the row index of the beginning and ending of each lap
    df_in: data frame contains the liner referecning of the data series. 
    reference_point: the linear referecing of the referece point which is used to deteremine whether or not a lap was completed.     
    """
    idx_list = []

    # Determine if the beginning of the data is before or after the reference point
    if df_in.iloc[0, :][dist_label] < reference_point:
        before_flag = True
    else:
        before_flag = False
    
    # Find idx where the vehicle just past the reference point
    prev_point = df_in.iloc[0, :][dist_label]
    for idx in range(len(df_in)):
        curr_point = df_in.iloc[idx, :][dist_label]
        if prev_point < reference_point and curr_point > reference_point:
            idx_list.append(idx)
        prev_point = curr_point
    if before_flag:
        idx_list = idx_list[1:]

    # Index range of each lap
    idx_range = []
    for lap in range(len(idx_list)):
        if lap == 0:
            idx_range.append([0,idx_list[lap]])
        else:
            idx_range.append([idx_list[lap-1], idx_list[lap]])
    idx_range.append([idx_list[-1], len(df_in)-1])

    return idx_range

