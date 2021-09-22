# Callable function for merging and resampling data series with common time stamps
  
import pandas as pd
import numpy as np
from scipy import optimize
from functools import reduce
from scipy.stats import pearsonr
import re
from datetime import datetime, timedelta

def checkColumnExist(df, keywords):
    for column_name in df.columns:
        if all([keyword in column_name for keyword in keywords]):
            return column_name
    return None

# Old AllGather Version Insert Orientation on the first row, this function is used to handel both cases
def smartLoadCSVFiles(filename):

    with open(filename, "r") as file:
        first_line = file.readline()
    return pd.read_csv(filename, skiprows=1) if "orientation" in first_line else pd.read_csv(filename)


def SmartPhoneFormatter(df):
    # Remove bbi columns
    for column_name in df.columns:
        if column_name.__contains__('bbi'):
            df = df.drop(columns=column_name)
    # This line is for Tianqi's fat fingers for putting a space before header
    df = df.rename(columns={' speed_ms': 'speed_ms'})
    return df

def GoProRename(df):
    # Remove cts columns
    for column_name in df.columns:
        if column_name.__contains__('cts'):
            df = df.drop(columns=column_name)
    # Rename acceleration columns
    for column_idx in range(len(df.columns)):
        column_name = df.columns[column_idx]
        if column_name.__contains__('Accelerometer'):
            df = df.rename(columns={
                           df.columns[column_idx]: 'accel_z_mps2', 
                           df.columns[column_idx+1]: 'accel_x_mps2', 
                           df.columns[column_idx+2]: 'accel_y_mps2',
                           })
            break
    # Rename gyro columns
    for column_idx in range(len(df.columns)):
        column_name = df.columns[column_idx]
        if column_name.__contains__('Gyroscope'):
            df = df.rename(columns={
                           df.columns[column_idx]: 'gyro_z_radps',
                           df.columns[column_idx+1]: 'gyro_x_radps',
                           df.columns[column_idx+2]: 'gyro_y_radps',
                           })
            break
    # Rename GPS columns
    for column_idx in range(len(df.columns)):
        column_name = df.columns[column_idx]
        if column_name.__contains__('GPS (Lat.)'):
            df = df.rename(columns={
                           df.columns[column_idx]: 'latitude_dd',
                           df.columns[column_idx+1]: 'longitude_dd',
                           df.columns[column_idx+2]: 'altitude_m',
                           df.columns[column_idx+3]: 'speed_ms',
                           df.columns[column_idx+4]: 'speed_3D_ms',
                           })
            break
    return df


def mergeData(inFiles, dt=0.25, col_keywords=["local", "timestamp"], new_col_name="local_timestamp_utc", GoPro = False):

    keywords = col_keywords # keyword for looking up timestamp column
    column_rename_to = new_col_name # output timestamp column name

    allDataFrames = [smartLoadCSVFiles(filename) for filename in inFiles]

    for i, df in enumerate(allDataFrames):
        col_name = checkColumnExist(df, keywords)
        if col_name is None:
            print("One of the DataFrame does not have local_timestamp column")
            exit(0)
        allDataFrames[i] = df.rename(columns={col_name: column_rename_to})

    df_final = reduce(lambda left, right: pd.merge(left, right, on=column_rename_to, how='outer'), allDataFrames)

    # Resample to time interval
    delta = pd.Timedelta(float(dt), unit='s')
    if type(df_final[column_rename_to][1]) == str:
        df_final.index = pd.to_datetime(df_final[column_rename_to])
    else:
        df_final.index = pd.to_datetime(df_final[column_rename_to], unit='ms')
    # df_final = df_final.resample(delta).mean().fillna(method='ffill')
    df_final = df_final.resample(delta).mean().interpolate(method='linear')

    # Replace time format in the final df
    df_final["local_time"] = df_final.index

    # Rename Header for GoPro data
    # Note that SmartPhone accel_z perpendicular to screen, while GoPro is accel_y
    if GoPro:
        df_final = GoProRename(df_final)
    else:
        df_final = SmartPhoneFormatter(df_final)

    if len(df_final) >= sum([len(df) for df in allDataFrames]):
        print("WARNING: NO COMMON TIMESTAMP!!!")
        return

    return df_final


def calculateBBI(row, zero_vector, x_acc_label, y_acc_label):
	return -np.degrees(np.arctan2(np.linalg.det([zero_vector, [row[x_acc_label], row[y_acc_label]]]),
                               np.dot(zero_vector, [row[x_acc_label], row[y_acc_label]])))


def backCalculateBBI(V, R, e, mps=True):
    """
    V: Speed in ft/s, if mps flag is true, speed is in m/s
    R: Curve radius
    e: Superelevation
    BBI: Ball bank indicator value (deg)
    k: vehicle roll rate 
    """
    # Convert meters/s to ft/sec
    if mps:
        V = 3.28084*V
    g = 32.1740
    
    BBI = np.rad2deg(np.arctan((V)**2/(g*R))-np.arctan(e/100))
    return BBI


def simpleLowpassFilter(input, alpha):
	output = [0.] * len(input)
	output[0] = input[0]
	for i in range(1, len(input)):
		output[i] = output[i-1] + float(alpha) * (input[i] - output[i-1])
	return output


def getZeroVector(data, nZero, x_acc_label, y_acc_label):
	#Get the first nZero record
	data = data[int(nZero[0]):min(int(nZero[1]), len(data))]
	return [data[x_acc_label].mean(), data[y_acc_label].mean()]


def computeBBI(inDF, zero_range = [0,10], lowpass_alpha = 1, x_acc_label='accel_x_mps2', y_acc_label='accel_y_mps2'):

    # Sign convention reminder
    # R positive: Left hand turn
    # e positive: counter close wise
    # BBI positive: counter close wise

    # Input DataFrame and rename acceleration column names
    data = inDF
    data = data.dropna(subset=[x_acc_label, y_acc_label])
    # Zeroing using first K measurements
    zero_vector = getZeroVector(data, zero_range, x_acc_label, y_acc_label)
    
    # Compute Unfiltered BBI
    data['BBI_computed'] = data.apply(lambda row: calculateBBI(
	    row, zero_vector, x_acc_label, y_acc_label), axis=1, result_type='expand')
    
    data = data.dropna(subset=['BBI_computed'])
    
    # Compute Filtered BBI
    data['accel_x_filtered'] = simpleLowpassFilter(
	    data[x_acc_label].tolist(), alpha=float(lowpass_alpha))
    data['accel_y_filtered'] = simpleLowpassFilter(
	    data[y_acc_label].tolist(), alpha=float(lowpass_alpha))
    data['BBI_computed_filtered'] = data.apply(lambda row: calculateBBI(
	    row, zero_vector, 'accel_x_filtered', 'accel_y_filtered'), axis=1, result_type='expand')

	# if'BBI' in data.columns:
	# 	corr = np.correlate(data['BBI_computed_filtered'], data['BBI'], 'full')
	# 	maxcorr = max(corr)
	# 	offset = np.argmax(corr)
	# 	print("max corr = ", maxcorr)
	# 	print("offset = ", offset)

	# if args.outFile:
	# 	data.to_csv(args.outFile[0], index=False)
	# else:
	# 	data.to_csv(args.inFile[0], index=False)
    
    return data


def getSuperelevation(V, R, BBI, k = 0, mps = True):
    """
    V: Speed in ft/s, if mps flag is true, speed is in m/s
    R: Curve radius
    BBI: Ball bank indicator value (deg)
    k: vehicle roll rate 
    """
    # Sign convention reminder
    # R positive: Left hand turn
    # e positive: counter close wise
    # BBI positive: counter close wise

    # Convert meters/s to ft/sec
    if mps:
        V = 3.28084*V
    g = 32.1740

    # Compute Superelevation
    superelevation = 100*np.tan(np.arctan(V**2/(g*R))-np.deg2rad(BBI/(1+k)))

    return superelevation

def getAdvisorySpeed(superelevation,R,BBI_max=12):

    # Sign convention reminder
    # R positive: Left hand turn
    # e positive: counter close wise rotation from level surface
    # BBI positive: counter close wise rotation from "zero" degree mark

    f_max = np.tan(np.deg2rad(BBI_max))

    if R < 0:
        f_max = -f_max

    adv_speed = np.sqrt(15*R*(f_max+superelevation/100))

    # return np.floor(adv_speed/5)*5
    return adv_speed


def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def leastsq_circle(x, y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    # if type(R) is not float:
    #     R = float(1000000)
    return R

def parseRiekerBBI(file, sample_dt = 1):
    """
    Read log file from Rieker BBI device
    """
    regex = r"\[([\d\-:. ]+)\] ([+\-]) ([\d.]+)"
    res = []
    df = None
    with open(file, 'r') as inFile:
        lines = inFile.readlines()
        for line in lines:
            matches = re.finditer(regex, line, re.MULTILINE)
            for matchNum, match in enumerate(matches, start=1):
                for groupNum in range(0, len(match.groups())):
                    groupNum = groupNum + 1
                    dt = match.group(1)
                    bbi = float(match.group(3)) if match.group(2) == '+' else -float(match.group(3))
                    naive = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
                    dt = '{:%Y-%m-%d %H:%M:%S.%f}'.format(naive)
                    d = {"local_timestamp_milliseconds": int(naive.timestamp() * 1000), "BBI": bbi}
                    res.append(d)
        df = pd.DataFrame(res)
        df.index = pd.to_datetime(df["local_timestamp_milliseconds"], unit='ms')
        delta = pd.Timedelta(float(sample_dt), unit='s')
        df_final = df.resample(delta).mean().interpolate(method='linear')
        return df_final


def alignData(target_df, moving_series, target_column='BBI_computed_filtered', new_column='BBI'):
    """
    Align time series with the same sampling frequency
    target_df: data frame with signal that the moving series will be aligned to 
    moving_series, data series with signal to be algned to the singal in the target df
    target_column: column name in the arget_df
    new_column: column name for the moving signal after alignment
    """
    out_df = target_df.copy()
    fixed = out_df[target_column].reset_index(drop=True)
    moving = moving_series.reset_index(drop=True)
    corr = np.correlate(fixed, moving, "full")
    offset = np.argmax(corr)-len(moving)+1
    if any(ele in new_column for ele in out_df.columns):
        col_idx = out_df.columns.get_loc(new_column)
    else:
        col_idx = -1
    # print('max_corr:', np.argmax(corr))
    # print('offset:', offset)
    # print("length_fixed:", len(fixed))
    # print("length_moving:", len(moving))
    out_df[new_column] = 0
    if offset >= 0:
        overlap_len = min(len(fixed), offset+len(moving))-offset
        out_df.iloc[offset:offset+overlap_len, col_idx] = moving.tolist()[0:overlap_len]
    else:
        overlap_len = min(len(moving), -offset+len(fixed))+offset
        out_df.iloc[0:overlap_len, col_idx] = moving.tolist()[-offset:-offset+overlap_len]
    
    # Check if registration resulted in resonalbe correlation results
    corr = pearsonr(out_df[target_column], out_df[new_column])[0]
    if corr > 0.5:
        print('Max Correlation found! corr = %5.4f ... ' %(corr), end ='')
        return out_df
    else:
        print('Warning: Could not get reasonable registration, registration skipped, please make sure correct file is used or change sign of the BBI values if positive BBI is defined differently.')
        return target_df


