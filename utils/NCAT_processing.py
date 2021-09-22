# NCAT_processing
import geopandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.stats import pearsonr
from seaborn import palettes

from .BBI_utils import mergeData, computeBBI, simpleLowpassFilter, getSuperelevation, getAdvisorySpeed, leastsq_circle, parseRiekerBBI, alignData, backCalculateBBI, getZeroVector
from .LRS_utils import NCATgetReferencePoints, getDistToMid, getReferenceCurve, NCATgetRadius, getRangeIndex, GYROgetRadius

class NCAT_processing:
    def __init__(self, inFiles, GoPro = True, dt = 1, suspension_para = 0):
        
        self.inFiles = inFiles
        self.dt = dt
        self.suspension_para = suspension_para
        self.lowpassAlpha = 0.9

        print('Initializing and Loading NCAT Track Geometry ... ', end = '')

        # Enable driver for reading KML files
        geopandas.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

        # Reading the KML files and Reprojection
        center_line = geopandas.read_file(
            '../NCAT Geometry/NCAT Center Line.kml', driver='KML')
        self.center_line = center_line.to_crs("EPSG:32166")

        left_line = geopandas.read_file(
            '../NCAT Geometry/NCAT Left Edge Line.kml', driver='KML')
        self.left_line = left_line.to_crs("EPSG:32166")

        right_line = geopandas.read_file(
            '../NCAT Geometry/NCAT Right Edge Line.kml', driver='KML')
        self.right_line = right_line.to_crs("EPSG:32166")

        mid_point_E = geopandas.read_file(
            '../NCAT Geometry/Curve Mid-Point East.kml', driver='KML')
        mid_point_E = mid_point_E.to_crs("EPSG:32166")

        mid_point_W = geopandas.read_file(
            '../NCAT Geometry/Curve Mid-Point West.kml', driver='KML')
        mid_point_W = mid_point_W.to_crs("EPSG:32166")

        self.track_length = self.center_line.geometry[0].length
        self.referencePoints = NCATgetReferencePoints(
            self.center_line.geometry[0], mid_point_E.geometry[0], mid_point_W.geometry[0])
        [self.mid_point_E_dist, self.mid_point_W_dist,
            self.mid_ST_dist, self.mid_NT_dist] = self.referencePoints
        
        superelevation_data = {
            'DTM': [-1000, -951.7, -900, -800, -700, -600, -543.7, -400, -200, 0, 200, 400, 543.7, 600, 700, 800, 900, 951.7, 1000],
            'SUP_W': [2.0, 2.7, 3.6, 6.2, 8.7, 12.8, 13.9, 14.1, 15.3, 14.3, 16.0, 16.0, 14.7, 12.6, 7.9, 5.1, 3.9, 2.6, 2.1],
            'SUP_E': [0.0, 2.1, 2.9, 5.1, 8.7, 13.0, 14.9, 15.7, 15.8, 14.2, 14.3, 15.2, 13.8, 12.2, 8.6, 5.6, 1.6, 2.7, 2.7],
        }
        self.df_super = pd.DataFrame(data=superelevation_data)
        self.df_super['Radius'] = self.df_super.apply(lambda row: NCATgetRadius(row['DTM']), axis=1)
        self.df_super['adv_speed_W'] = self.df_super.apply(lambda row: getAdvisorySpeed(row['SUP_W'], row['Radius']), axis=1)
        self.df_super['adv_speed_E'] = self.df_super.apply(lambda row: getAdvisorySpeed(row['SUP_E'], row['Radius']), axis=1)

        print('Done!')

        if GoPro:
            self.GoProProcessing()
        else:
            self.SmartPhoneProcessing()


    def GoProProcessing(self):
        """
        Cominbe tables
        Create GeoDataFrame
        Set crs and project to crs with unit of feet
        find reference points
        compute track dist of each point
        find range idx of each lap
        """
        print('Processing GoPro Data ... ', end = '')
        # merge acceleration, gps and gyro data (dt = 0.25 or 1)
        df = mergeData(self.inFiles, dt=self.dt, col_keywords=['date'], new_col_name="local_timestamp_utc", GoPro=True)
        # compute BBI
        df = computeBBI(df, lowpass_alpha=self.lowpassAlpha, y_acc_label='accel_z_mps2')
        # Check if radius calculation need to have the sign flipped
        zero_vector = getZeroVector(df, [0, 10], x_acc_label='accel_x_mps2', y_acc_label='accel_z_mps2')
        if zero_vector[1] > 0:
            flip_sign_flag = False
        else:
            flip_sign_flag = True
        # filter GYRO
        df['gyro_z_radps'] = simpleLowpassFilter(df['gyro_z_radps'].tolist(), alpha=float(self.lowpassAlpha))
        # make GeoDataFrame
        self.gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df['longitude_dd'], df['latitude_dd']))
        self.gdf = self.gdf.set_crs("epsg:4326")
        self.gdf = self.gdf.to_crs("EPSG:32166")
        # process data
        self.gdf['Track_dist'] = self.gdf.apply(
            lambda row: self.center_line.geometry[0].project(row['geometry']), axis=1)

        self.gdf['Dist_to_mid'] = self.gdf.apply(lambda row: getDistToMid(
            row['Track_dist'], self.track_length, self.referencePoints), axis=1)

        self.gdf['Reference_curve'] = self.gdf.apply(lambda row: getReferenceCurve(
            row['Track_dist'], self.referencePoints), axis=1)

        self.interpolateMeasuredSuper()
        
        self.gdf['Radius'] = self.gdf.apply(lambda row: NCATgetRadius(row['Dist_to_mid']), axis=1)

        self.gdf['adv_speed_measured'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Measured_Superelevation'], row['Radius']), axis=1)

        self.gdf['Superelevation'] = self.gdf.apply(lambda row: getSuperelevation(
            row['speed_ms'], row['Radius'], row['BBI_computed_filtered'], k=self.suspension_para), axis=1)

        self.gdf['adv_speed'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Superelevation'], row['Radius']), axis=1)
        
        self.gdf['Radius_GYRO'] = self.gdf.apply(
            lambda row: GYROgetRadius(row['speed_ms'], row['gyro_z_radps'], flip_sign=flip_sign_flag), axis=1)

        self.gdf['Superelevation_GYRO'] = self.gdf.apply(lambda row: getSuperelevation(
            row['speed_ms'], row['Radius_GYRO'], row['BBI_computed_filtered'], k=self.suspension_para), axis=1)

        self.gdf['adv_speed_GYRO'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Superelevation_GYRO'], row['Radius']), axis=1)

        self.gdf['Side_friction_angle'] = self.gdf.apply(
            lambda row: backCalculateBBI(row['speed_ms'], row['Radius_GYRO'], row['Measured_Superelevation']), axis=1)
        
        self.idx_range = getRangeIndex(self.gdf, 'Track_dist', self.mid_NT_dist)

        self.getLapNumber()

        print('Done!')

    def SmartPhoneProcessing(self):
        """
        Cominbe tables
        Create GeoDataFrame
        Set crs and project to crs with unit of feet
        find reference points
        compute track dist of each point
        find range idx of each lap
        """
        print('Processing Smartphone Data ... ', end='')
        # merge acceleration, gps and gyro data
        df = mergeData(self.inFiles, dt=self.dt, col_keywords=[
                       "local", "timestamp"], new_col_name="local_timestamp_utc", GoPro=False)
        # compute BBI
        df = computeBBI(df, zero_range=[0, 10], lowpass_alpha=self.lowpassAlpha, x_acc_label='accel_y_mps2', y_acc_label='accel_x_mps2')
        # Check if radius calculation need to have the sign flipped
        zero_vector = getZeroVector(df, [0, 10], x_acc_label='accel_y_mps2', y_acc_label='accel_x_mps2')
        if zero_vector[1] > 0:
            flip_sign_flag = False
        else:
            flip_sign_flag = True
        # filter GYRO
        df['angvelocity_x_radps'] = simpleLowpassFilter(df['angvelocity_x_radps'].tolist(), alpha=float(self.lowpassAlpha))
        # make GeoDataFrame
        self.gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df['longitude_dd'], df['latitude_dd']))
        self.gdf = self.gdf.set_crs("epsg:4326")
        self.gdf = self.gdf.to_crs("EPSG:32166")
        # process data
        self.gdf['Track_dist'] = self.gdf.apply(
            lambda row: self.center_line.geometry[0].project(row['geometry']), axis=1)

        self.gdf['Dist_to_mid'] = self.gdf.apply(lambda row: getDistToMid(
            row['Track_dist'], self.track_length, self.referencePoints), axis=1)

        self.gdf['Reference_curve'] = self.gdf.apply(lambda row: getReferenceCurve(
            row['Track_dist'], self.referencePoints), axis=1)

        self.interpolateMeasuredSuper()

        self.gdf['Radius'] = self.gdf.apply(
            lambda row: NCATgetRadius(row['Dist_to_mid']), axis=1)

        self.gdf['adv_speed_measured'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Measured_Superelevation'], row['Radius']), axis=1)

        self.gdf['Superelevation'] = self.gdf.apply(lambda row: getSuperelevation(
            row['speed_ms'], row['Radius'], row['BBI_computed_filtered'], k=self.suspension_para), axis=1)

        self.gdf['adv_speed'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Superelevation'], row['Radius']), axis=1)
        
        self.gdf['Radius_GYRO'] = self.gdf.apply(lambda row: GYROgetRadius(
            row['speed_ms'], row['angvelocity_x_radps'], flip_sign=flip_sign_flag), axis=1)
        
        self.gdf['Superelevation_GYRO'] = self.gdf.apply(lambda row: getSuperelevation(
            row['speed_ms'], row['Radius_GYRO'], row['BBI_computed_filtered'], k=self.suspension_para), axis=1)

        self.gdf['adv_speed_GYRO'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Superelevation_GYRO'], row['Radius']), axis=1)
        
        self.gdf['Side_friction_angle'] = self.gdf.apply(
            lambda row: backCalculateBBI(row['speed_ms'], row['Radius_GYRO'], row['Measured_Superelevation']), axis=1)
        
        self.idx_range = getRangeIndex(self.gdf, 'Track_dist', self.mid_NT_dist)

        self.getLapNumber()

        print('Done!')

    def getLapNumber(self):
        self.gdf['lap_number'] = 0
        lap_number = 1
        for range_idx in self.idx_range:
            self.gdf.iloc[range_idx[0]:range_idx[1], -1] = lap_number
            lap_number += 1

    def getRadiusFromGPS(self, window_length = 9):
        """
        Compute Radius from GPS points
        For each given point, a set of points around it will be used to estimate the radius

        """
        self.gdf['Radius_GPS'] = ''
        column_idx = self.gdf.columns.get_loc('Radius_GPS')
        for idx in range(len(self.gdf)):
            start_idx = max(0,idx-int(np.floor(window_length/2)))
            end_idx = start_idx + window_length
            x = self.gdf.geometry[start_idx:end_idx].x.to_numpy()
            y = self.gdf.geometry[start_idx:end_idx].y.to_numpy()
            self.gdf.iloc[idx, column_idx] = min(1000000, leastsq_circle(x, y))
        
        self.gdf['Superelevation_GPS'] = self.gdf.apply(lambda row: getSuperelevation(
            row['speed_ms'], row['Radius_GPS'], row['BBI_computed_filtered'], k=self.suspension_para), axis=1)

        self.gdf['adv_speed_GPS'] = self.gdf.apply(lambda row: getAdvisorySpeed(
            row['Superelevation_GPS'], row['Radius']), axis=1)

    def plot_lap(self, column = 'BBI_computed_filtered', lap=1):
        """
        First lap will be plotted with lap = 1 instead of lap = 0
        """ 
        sub_gdf = self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :]
        ax = sub_gdf.plot(column=column, figsize=[30, 30], markersize=40, legend=False)
        self.center_line.plot(ax = ax, color='red',linewidth = 0.5)
        self.left_line.plot(ax=ax, color='blue', linewidth=0.5)
        self.right_line.plot(ax=ax, color='blue', linewidth=0.5)
        plt.show()
    
    def plot_superelevation(self, column='Superelevation', lap_list=[1, 2, 3, 4]):
        """ plot superelevation results of a lap """
        # sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
        # sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
        # sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
        # sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])

        # ax = self.df_super.plot(x='DTM', y='SUP_W', figsize=[10, 5])
        # sub_set_W.plot(ax=ax, x='Dist_to_mid', y=column)
        # plt.show()

        # ax = self.df_super.plot(x='DTM', y='SUP_E', figsize=[10, 5])
        # sub_set_E.plot(ax=ax, x='Dist_to_mid', y=column)
        # plt.show()

        sub_df_super = self.df_super[np.abs(self.df_super['DTM']) <= 1000]
        ax_E = sub_df_super.plot(x='DTM', y='SUP_E', figsize=[10, 5])
        ax_W = sub_df_super.plot(x='DTM', y='SUP_W', figsize=[10, 5])
        for idx in range(len(lap_list)):
            lap = lap_list[idx]
            sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
            sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
            sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
            sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])
            ax_E = sub_set_E.plot(ax=ax_E, x='Dist_to_mid', y=column)
            ax_W = sub_set_W.plot(ax=ax_W, x='Dist_to_mid', y=column)
        plt.show(ax_E)
        plt.show(ax_W)

    def superelevation_accuracy(self, column='Superelevation', lap_list=[1, 2, 3, 4], method='RMSE'):
        """ compare superelevation results of each lap """
       
        results = np.zeros((len(lap_list),2))

        for idx in range(len(lap_list)):
            lap = lap_list[idx]
            sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
            sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
            sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
            sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])

            GT_series_E = sub_set_E['Measured_Superelevation'].to_numpy()
            Test_series_E = sub_set_E[column].to_numpy()

            GT_series_W = sub_set_W['Measured_Superelevation'].to_numpy()
            Test_series_W = sub_set_W[column].to_numpy()

            if method == 'RMSE':
                results[idx][0] = np.sqrt(np.mean((GT_series_E-Test_series_E)**2))
                results[idx][1] = np.sqrt(np.mean((GT_series_W-Test_series_W)**2))
            elif method == 'CORR':
                results[idx][0] = pearsonr(GT_series_E, Test_series_E)[0]
                results[idx][1] = pearsonr(GT_series_W, Test_series_W)[0]
        return results # East in 1st column, West in 2nd column
    
    def getAdvisorySpeedResults(self, column='adv_speed', lap_list=[1, 2, 3, 4]):
        results = np.zeros((len(lap_list), 2))
        for idx in range(len(lap_list)):
            lap = lap_list[idx]
            sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
            sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
            sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
            sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])
            results[idx][0] = np.min(sub_set_E[column])
            results[idx][1] = np.min(sub_set_W[column])
        return results  # East in 1st column, West in 2nd column
    
    def plotAdvisorySpeed(self, column = 'adv_speed', lap_list = [1,2,3,4]):
        """
        Plot the advisory speed
        """
        sub_df_super = self.df_super[np.abs(self.df_super['DTM']) <= 700]
        ax_E = sub_df_super.plot(x='DTM', y='adv_speed_E', figsize=[10, 5])
        ax_W = sub_df_super.plot(x='DTM', y='adv_speed_W', figsize=[10, 5])
        for idx in range(len(lap_list)):
            lap = lap_list[idx]
            sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
            sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 700]
            sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
            sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])
            ax_E = sub_set_E.plot(ax=ax_E, x='Dist_to_mid', y=column)
            ax_W = sub_set_W.plot(ax=ax_W, x='Dist_to_mid', y=column)
        plt.show(ax_E)
        plt.show(ax_W)

    def registerRiekerBBI(self, BBIfile):
        print('Registering RiekerBBI data ... ', end='')
        BBI_df = parseRiekerBBI(BBIfile, sample_dt=self.dt)
        BBI_df['BBI'] = -BBI_df['BBI'] # Rieker BBI will have a different sign than the computed BBI
        self.gdf = alignData(self.gdf, BBI_df['BBI'], target_column='BBI_computed_filtered', new_column='BBI')

        # Check if BBI registration is successful
        if any(ele in 'BBI' for ele in self.gdf.columns):
            self.gdf['BBI_filtered'] = simpleLowpassFilter(self.gdf['BBI'].tolist(), alpha=float(self.lowpassAlpha))

            self.gdf['Superelevation_Rieker'] = self.gdf.apply(lambda row: getSuperelevation(
                row['speed_ms'], row['Radius'], row['BBI_filtered'], k=self.suspension_para), axis=1)

            self.gdf['adv_speed_Rieker'] = self.gdf.apply(lambda row: getAdvisorySpeed(
                row['Superelevation_Rieker'], row['Radius']), axis=1)
            
            self.gdf['Superelevation_GYRO_Rieker'] = self.gdf.apply(lambda row: getSuperelevation(
                row['speed_ms'], row['Radius_GYRO'], row['BBI_filtered'], k=self.suspension_para), axis=1)

            self.gdf['adv_speed_GYRO_Rieker'] = self.gdf.apply(lambda row: getAdvisorySpeed(
                row['Superelevation_GYRO_Rieker'], row['Radius']), axis=1)
            
            print('Done!')    

    def BBI_accuracy(self, ref_column='BBI_filtered', test_column='BBI_computed_filtered', lap_list=[1, 2, 3, 4, 5], method='RMSE'):
        """ compare BBI results of each lap """
        
        results = np.zeros((len(lap_list),2))

        for idx in range(len(lap_list)):
            lap = lap_list[idx]
            sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
            sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
            sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
            sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])
            # series for east curve
            GT_series_E = sub_set_E[ref_column].to_numpy()
            Test_series_E = sub_set_E[test_column].to_numpy()
            # series for west curve
            GT_series_W = sub_set_W[ref_column].to_numpy()
            Test_series_W = sub_set_W[test_column].to_numpy()

            if method == 'RMSE':
                results[idx][0] = np.sqrt(np.mean((GT_series_E-Test_series_E)**2))
                results[idx][1] = np.sqrt(np.mean((GT_series_W-Test_series_W)**2))
            elif method == 'CORR':
                results[idx][0] = pearsonr(GT_series_E, Test_series_E)[0]
                results[idx][1] = pearsonr(GT_series_W, Test_series_W)[0]
        return results # East in 1st column, West in 2nd column

    def plotRegression(self, data='BBI', lap_list=[1, 2, 3, 4]):
                
        if 'Super' in data:
            if data == 'Super':
                column = 'Superelevation'
            if data == 'Super_Rieker':
                column = 'Superelevation_Rieker'
            elif data == 'Super_GYRO':
                column = 'Superelevation_GYRO'
            elif data == 'Super_GYRO_Rieker':
                column = 'Superelevation_GYRO_Rieker'


            DTM = self.df_super.DTM.to_numpy()
            SUP_E = self.df_super.SUP_E.to_numpy()
            SUP_W = self.df_super.SUP_W.to_numpy()
            f_E = interpolate.interp1d(DTM, SUP_E)
            f_W = interpolate.interp1d(DTM, SUP_W)

            for idx in range(len(lap_list)):
                lap = lap_list[idx]
                sub_set = pd.DataFrame(
                    self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
                sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
                sub_set_W = sub_set[sub_set['Reference_curve'] == 'West Curve'].sort_values(by=['Dist_to_mid'])
                sub_set_E = sub_set[sub_set['Reference_curve'] == 'East Curve'].sort_values(by=['Dist_to_mid'])
                # Interpolate East superelevation
                xnew = sub_set_E.Dist_to_mid.to_numpy()
                # use interpolation function returned by `interp1d`
                Series_E = sub_set_E['Measured_Superelevation'].reset_index(
                    drop=True).to_frame('Measured_Superelevation')
                Series_E['Computed_Superelevation'] = sub_set_E[column].reset_index(drop=True)
                # Interpolate West superelevation
                xnew = sub_set_W.Dist_to_mid.to_numpy()
                # use interpolation function returned by `interp1d`
                Series_W = sub_set_W['Measured_Superelevation'].reset_index(
                    drop=True).to_frame('Measured_Superelevation')
                Series_W['Computed_Superelevation'] = sub_set_W[column].reset_index(drop=True)
                if idx == 0:
                    df = Series_E
                    df = df.append(Series_W)
                else:
                    df = df.append(Series_E)
                    df = df.append(Series_W)
            ax = sns.lmplot(x="Measured_Superelevation",
                            y="Computed_Superelevation", data=df, truncate=False, line_kws={"linestyle": (0, (3, 3)), 'color': [0.1, 0.1, 0.8]})
            ax = plt.axline((0, 0), slope=1, color="black", lw=2)
            plt.show()

        elif data == 'BBI':
            column_GT = 'BBI'
            column_Test = 'BBI_computed_filtered'

            for idx in range(len(lap_list)):
                lap = lap_list[idx]
                sub_set = pd.DataFrame(self.gdf.iloc[self.idx_range[lap-1][0]:self.idx_range[lap-1][1], :])
                # sub_set = sub_set[np.abs(sub_set['Dist_to_mid']) < 1000]
                Series = pd.DataFrame(data=sub_set[column_GT].to_numpy(), columns=['Rieker BBI'])
                Series['Computed BBI'] = sub_set[column_Test].reset_index(drop=True)
                if idx == 0:
                    df = Series
                else:
                    df = df.append(Series)
            ax = sns.lmplot(x="Rieker BBI", y="Computed BBI",
                            data=df, truncate=False, line_kws = {"linestyle":(0, (3, 3)),'color':[0.1,0.1,0.8]})
            ax = plt.axline((0, 0), slope=1, color="black", lw=2)
            plt.show()

        else:
            print('Data not found!')

    def interpolateMeasuredSuper(self):
        DTM = self.df_super.DTM.to_numpy()
        SUP_E = self.df_super.SUP_E.to_numpy()
        SUP_W = self.df_super.SUP_W.to_numpy()
        f_E = interpolate.interp1d(DTM, SUP_E)
        f_W = interpolate.interp1d(DTM, SUP_W)
        DTM_idx = self.gdf.columns.get_loc('Dist_to_mid')
        RefCurve_idx = self.gdf.columns.get_loc('Reference_curve')
        self.gdf['Measured_Superelevation'] = np.nan
        
        for idx in range(len(self.gdf)):
            if np.abs(self.gdf.iloc[idx, DTM_idx]) < 1000:
                if self.gdf.iloc[idx, RefCurve_idx] == 'East Curve':
                    self.gdf.iloc[idx, -1] = f_E(self.gdf.iloc[idx, DTM_idx])
                else:
                    self.gdf.iloc[idx, -1] = f_W(self.gdf.iloc[idx, DTM_idx])

    def TestChengboMethod(self, lambda_value = 0.95, dt = 1, GoPro = False):
        """
        Test Code to evaluate Chengbo's method for superelevation
        """
        if GoPro:
            Yacc_label = 'accel_x_mps2'
            Yacc_sign = -1
            Zacc_label = 'accel_z_mps2'
            Zacc_sign = +1
            Xang_label = 'gyro_y_radps'
            Xang_sign = -1
        else:
            Yacc_label = 'accel_y_mps2'
            Yacc_sign = +1
            Zacc_label = 'accel_x_mps2'
            Zacc_sign = +1
            Xang_label = 'angvelocity_z_radps'
            Xang_sign = -1
        
        theta = 0
        # g = 32.1740
        g = 9.80665
        self.gdf['Superelevation_chengbo'] = ''
        self.gdf['debug_yacc'] = ''
        self.gdf['debug_V'] = ''
        self.gdf['debug_R'] = ''

        for idx in range(len(self.gdf)):
            
            V = self.gdf['speed_ms'][idx]
            R = self.gdf['Radius_GYRO'][idx]/3.28084
            Xang = self.gdf[Xang_label][idx]*Xang_sign
            Yacc = self.gdf[Yacc_label][idx]*Yacc_sign
            Zacc = self.gdf[Zacc_label][idx]*Zacc_sign

            theta = lambda_value*(theta+Xang*dt)+(1-lambda_value)*np.arctan(Yacc/Zacc)

            # theta = (0.1/(1.1))*np.arctan(Yacc/Zacc)

            term1 = np.arccos(Yacc/np.sqrt(((V)**2/R)**2+g**2))
            term2 = np.arccos((V**2/R)/np.sqrt(((V)**2/R)**2+g**2))
            self.gdf['Superelevation_chengbo'][idx] = np.tan(term1-term2+theta)*100

            self.gdf['debug_yacc'][idx] = Yacc
            self.gdf['debug_V'][idx] = V
            self.gdf['debug_R'][idx] = R





