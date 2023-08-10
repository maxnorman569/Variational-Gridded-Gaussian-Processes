# numeric imports
import numpy as np
# data imports
import xarray as xr
# integration imports
from scipy.integrate import simpson
# misc imports
import os
from typing import List, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class MissionData:

    def __init__(self, 
                 root_folder : str, 
                 mission_name : str, 
                 years : List[str], 
                 months : List[str], 
                 latitude_range : Tuple[float, float] = (-90., 90.),
                 longitude_range : Tuple[float, float] = (-180., 180.)
                 ) -> 'MissionData':
        """
        Class to load and store mission data.

        Arguments:
            root_folder (str)       : The folder path where the mission data is located.
            mission_name (str)      : The name of the mission. Should be one of the acceptable mission names. (i.e in ['e1', 'e1g', 'e2', 'tp', 'tpn', 'g2', 'j1', 'j1n', 'j1g', 'j2', 'j2n', 'j2g', 'j3', 'j3n', 'en', 'enn', 'c2', 'c2n', 'al', 'alg', 'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 's3b', 's6a-hr', 's6a-lr'])
            year (str)              : The year of the mission as a string. (i.e 'YYYY')
            month (str)             : The month of the mission as a string. (i.e 'MM', '01' for January)
            latitude_range (tuple)  : The latitude range of the mission data. (min, max) from -90 to 90.
            longitude_range (tuple) : The longitude range of the mission data. (min, max) from -180 to 180.

        Returns:
            None

        Raises:
            ValueError              : If the provided mission name is not in the list of acceptable mission names.
        """
        # check that provided mission name is valid
        available_missions = ['e1', 'e1g', 'e2', 'tp', 'tpn', 
                              'g2', 'j1', 'j1n', 'j1g', 'j2', 
                              'j2n', 'j2g', 'j3', 'j3n', 'en', 
                              'enn', 'c2', 'c2n', 'al', 'alg', 
                              'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 
                              's3b', 's6a-hr', 's6a-lr']
        # raise error if mission name not in list of available missions
        if mission_name not in available_missions:
            raise ValueError("Invalid mission name provided. Mission name must be one of the following: {}".format(available_missions))
        # set class attributes
        self.mission_name = mission_name
        self.mission_folder = os.path.join(root_folder, f'cmems_obs-sl_eur_phy-ssh_my_{self.mission_name}-l3-duacs_PT1S')
        self.years = years
        self.months = months
        self.min_latitude = latitude_range[0]
        self.max_latitude = latitude_range[1]
        self.min_longitude = longitude_range[0]
        self.max_longitude = longitude_range[1]
        self.mission_data = self.load_data()

    def load_data(self, ):
        """
        Loades the mission data from the provided arguments.

        Arguments:
            None

        Returns:
            month_dataset (xarray.Dataset)  : The mission data for the provided arguments.

        Raises:
            ValueError                      : If no data files are found in the provided directory.
        """
        
        # print mission name
        print("\nLoading data for mission: '{}'".format(self.mission_name))

        # load mission data
        datasets = []
        for year in self.years:
            for month in self.months:
                # get data directory
                data_dir = os.path.join(self.mission_folder, year, month)
                # check data directory exists
                if os.path.isdir(data_dir):
                    pass
                else:
                    print('> {}-{} | Directory does not exist: {}'.format(year, month, data_dir))
                    continue
                # get data files
                files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
                # check data files exist
                if len(files) == 0:
                    print('> {}-{} | Data files do not exist: {}'.format(year, month, data_dir))
                    continue
                # load data files as xarray datasets
                for file in files:
                    try:
                        data = xr.open_dataset(file)
                        # convert longitude from 0-360 to -180-180
                        data['longitude'] = xr.where(data['longitude'] > 180., data['longitude'] - 360., data['longitude'])
                        # filter data by latitude and longitude
                        longitude_mask = xr.where((data['longitude'] > self.min_longitude) & (data['longitude'] < self.max_longitude), True, False)
                        latitude_mask = xr.where((data['latitude'] > self.min_latitude) & (data['latitude'] < self.max_latitude), True, False)
                        data = data.where(longitude_mask & latitude_mask, drop = True)
                        # append data to list of datasets
                        datasets.append(data)
                    except:
                        pass
                # print completion message
                print('> {}-{} | completed'.format(year, month))
        # concatenate the datasets along the time dimension
        if len(datasets) == 0:
            return xr.Dataset(
                coords=dict(
                        time=None, 
                        longitude=None, 
                        latitude=None),
                data_vars=dict(
                        cycle = None, 
                        track = None, 
                        sla_unfiltered = None, 
                        sla_filtered = None,
                        dac = None, 
                        ocean_tide = None, 
                        internal_tide = None, 
                        lwe = None, 
                        mdt = None, 
                        tpa_correction = None,),
                attrs=dict(
                        description="Empty dataset, no data found."),
            )
        elif len(datasets) == 1:
            mission_data = datasets[0]
        else:
            mission_data = xr.concat(datasets, dim = 'time')
        return mission_data
    

class MissionAgnosticData:
    
    def __init__(self, 
                root_folder : str,
                mission_names : List[str], 
                years : List[str], 
                months : List[str],
                latitude_range : Tuple[float, float] = (-90., 90.),
                longitude_range : Tuple[float, float] = (-180., 180.)
                 ):
        """
        Class to load and store mission agnostic data. (i.e data that is not specific to a single mission.)

        Arguments:
            root_folder (str)    : The folder path where the mission data is located.
            mission_names (str)      : The name of the mission. Should be one of the acceptable mission names.
            years (str)              : The year of the mission.
            months (str)             : The month of the mission.
            latitude_range (tuple)  : The latitude range of the mission data. (min, max) from -90 to 90.
            longitude_range (tuple) : The longitude range of the mission data. (min, max) from -180 to 180.

        Returns:
            None

        Raises:
            ValueError              : If the provided mission names are not in the list of acceptable mission names.
        """
        # check that provided mission name is valid
        available_missions = ['e1', 'e1g', 'e2', 'tp', 'tpn', 
                              'g2', 'j1', 'j1n', 'j1g', 'j2', 
                              'j2n', 'j2g', 'j3', 'j3n', 'en', 
                              'enn', 'c2', 'c2n', 'al', 'alg', 
                              'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 
                              's3b', 's6a-hr', 's6a-lr']
        if set(mission_names).issubset(set(available_missions)):
            pass
        else:
            raise ValueError("Invalid mission name provided. Mission names must be in: {}".format(available_missions))
        # set class attributes
        self.data = xr.concat([MissionData(root_folder, mission_name, years, months, latitude_range, longitude_range).mission_data for mission_name in mission_names], dim = 'time')


class SimulationData(ABC):

    def __init__(self, 
                 root_folder : str, 
                 year : str, 
                 month : str, 
                 day : str,
                 ):
        """ 
        Class to lead and store simulation data in the Gulf Stream

        Arguments:
            root_folder (str)       : The folder path where the mission data is located.
            year (str)              : The year of the simulation. (i.e 'YYYY')
            month (str)             : The months of the simulation. (i.e 'MM')
            day (str)               : The day of the simulation. (i.e 'DD')

        Returns:
            None

        Raises:
            ValueError              : If the provided arguments do not correspond to a simulation.
        """
        # check that file exists
        sim_file_name = f'NATL60-CJM165_GULFSTREAM_y{year}m{month}d{day}.1h_SSH.nc'
        sim_file_path = os.path.join(root_folder, sim_file_name)
        if not os.path.exists(sim_file_path):
            raise ValueError(f'File {sim_file_path} does not exist.')
        # set class attributes
        self.simulation_year = year
        self.simulation_month = month
        self.simulation_day = day
        self.simulation_name = f"NATL60-CJM165_GULFSTREAM_y{year}m{month}d{day}"
        self.simulation_file_path = sim_file_path
        self.simulation_tracked_observations = None

        @abstractmethod
        def load_data(self,):
            pass


class SimulationDataDay(SimulationData):

    """ Class inherits from SimulationData to lead simulation data for a whole day. (24h) """

    def __init__(self, 
                 root_folder : str, 
                 year : str, 
                 month : str, 
                 day : str,):
        super().__init__(root_folder, year, month, day) 
        self.data = self.load_data()

    def load_data(self, ) -> xr.Dataset:
        """ 
        Load the all day simulation data into an xarray dataset.

        Arguments:
            None

        Returns:
            None

        Raises:
            None
        """
        ds = xr.open_dataset(self.simulation_file_path)

        return ds
    

class SimulationDataHour(SimulationData):

    """ Class inherits from SimulationData to lead simulation data for a specific hour. """

    def __init__(self, 
                 root_folder : str, 
                 year : str, 
                 month : str, 
                 day : str,
                 hour : int):
        super().__init__(root_folder, year, month, day,)  # Call the parent class constructor
        self.hour = hour
        self.data = self.load_data()

    def load_data(self,) -> xr.Dataset:
        """ 
        Load a specific hour from an all day simulation into an xarray dataset.

        Arguments:
            hour (int)              : The hour of the simulation to load. (i.e 0 - 23)

        Returns:
            None

        Raises:
            ValueError             : If the provided hour is not between 0 and 23.
        """
        # check that hour is valid
        if self.hour < 0 or self.hour > 23:
            raise ValueError(f'Hour must be between 0 and 23. Provided hour: {self.hour}')
        # load data set
        ds = xr.open_dataset(self.simulation_file_path)
        # select the hour
        ds = ds.isel(time = self.hour)
        return ds
    
    def generate_track(
        self,
        trajectory_gradient : int,
        track_sparsity : float,
        observation_sparisty : int,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates synthetic satellite track observations for a SimulationDataHour object.  

        # Arguments:
            data (SimulationDataHour)                   : The SimulationDataHour object to generate tracks for.
            trajectory_gradient (int)                   : The gradient of the satelite trajectory (i.e 0 for horizontal/vertical lines).
            track_sparsity (float)                      : The spacing between tracks in degrees measured on the longitude. (i.e 1 for 1 track per degree, 0.5 for 2 tracks per degree, etc)
            observation_sparisty (int)                  : How many observations to "skip" on the track (i.e 5 selects every 5th simulation point along the track). Choose 0 for no sparsity.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]   : A tuple containing of the track (longitude, latitude, SSH) data.
        """
        # check that track sparsity is valid
        if (track_sparsity <= 0) or (track_sparsity > 10):
            raise ValueError(f'Track sparsity must be between 0 and 10. Provided track sparsity: {track_sparsity}')
        # meta
        DEGREE_RANGE = 10
        LON_DIM = 600
        LAT_DIM = 600
        # data
        track_lon_idxs = np.empty(0)
        track_lat_idxs = np.empty(0)
        ############################
        # TRACKS FROM LONGITUDE
        ############################
        # lon index range
        max_lon_idx = int(LON_DIM / trajectory_gradient)
        # compute track spacing
        n_lon_tracks = int((DEGREE_RANGE / track_sparsity))
        lon_index_shift_size =  track_sparsity * (LON_DIM / DEGREE_RANGE) 
        # generate tracks
        for i in range(n_lon_tracks):
            # compute the start and end indices for the longitude
            lon_index_shift = int(i * lon_index_shift_size)
            lon_idx_start = int(lon_index_shift)
            lon_idx_end = int(max_lon_idx + lon_index_shift) if int(max_lon_idx + lon_index_shift) <= 600 else 600
            # get lon indxs
            forward_lon_idxs = np.repeat(np.arange(lon_idx_start, lon_idx_end), trajectory_gradient)
            backward_lon_idxs = np.repeat(np.arange(lon_idx_start, lon_idx_end), trajectory_gradient)
            lon_idxs = np.append(forward_lon_idxs, backward_lon_idxs)
            track_lon_idxs = np.append(track_lon_idxs, lon_idxs)
            # get lat indxs
            forward_lat_idxs = np.arange(len(forward_lon_idxs))
            backward_lat_idx = np.arange(-1, -len(backward_lon_idxs)-1, -1)
            lat_idxs = np.append(forward_lat_idxs, backward_lat_idx)
            track_lat_idxs = np.append(track_lat_idxs, lat_idxs)
        ############################
        # TRACKS FROM LATITUDE
        ############################
        # lon index range
        max_lon_idx = int(LON_DIM / trajectory_gradient)
        # adjust the sparisty
        lat_track_sparsity = track_sparsity * trajectory_gradient
        # compute the track spacing
        n_lat_tracks = int((DEGREE_RANGE / lat_track_sparsity))
        lat_index_shift_size =  lat_track_sparsity * (LAT_DIM / DEGREE_RANGE) 
        for j in range(n_lat_tracks):
            # compute the start and end indices for the longitude
            lat_index_shift = int(j * lat_index_shift_size)
            lat_idx_start = int(lat_index_shift)
            lat_idx_end = LAT_DIM
            # get lat indxs
            forward_lat_idxs = np.arange(lat_idx_start , lat_idx_end)
            backward_lat_idx = np.arange(-lat_idx_start, -LAT_DIM, -1)
            lat_idxs = np.append(forward_lat_idxs, backward_lat_idx)
            track_lat_idxs = np.append(track_lat_idxs, lat_idxs)
            # get lon indxs
            forward_lon_idxs = np.repeat(np.arange(0, max_lon_idx), trajectory_gradient)[:len(forward_lat_idxs)]
            backward_lon_idxs = np.repeat(np.arange(0, max_lon_idx), trajectory_gradient)[:len(backward_lat_idx)]
            lon_idxs = np.append(forward_lon_idxs, backward_lon_idxs)
            track_lon_idxs = np.append(track_lon_idxs, lon_idxs)
        ############################
        # UNPACK DATA
        ############################
        track_lon = self.data.lon.values[track_lon_idxs.astype(int)]
        track_lat = self.data.lat.values[track_lat_idxs.astype(int)]
        track_ssh = self.data.sossheig.values[track_lat_idxs.astype(int), track_lon_idxs.astype(int)]
        # return track_lon, track_lat, track_ssh based on specified sparsity
        if observation_sparisty == 0:
            return track_lon, track_lat, track_ssh
        else:
            return track_lon[::observation_sparisty], track_lat[::observation_sparisty], track_ssh[::observation_sparisty]
        


class GulfStream: 

    """ 
    This class is used to load the observation simulation data from a specified satallite 
    
    Available Satellites: ['envisat', 'geosat2', 'jason1', 'karin_swot', 'nadir_swot', 'topex-poseidon_interleaved']
    """ 

    def __init__(self,
                obs_root_folder : str,
                ref_root_folder : str,
                satellite_name : str,
                year_frame : Tuple[str, str],
                month_frame : Tuple[str, str],
                day_frame : Tuple[str, str],
                hour_frame : Tuple[str, str],):
        """ 
        Arguments:
            obs_root_folder (str)           : The root folder ('dc_obs') where the observation data is stored
            ref_root_folder (str)           : The root folder ('dc_ref') where the reference data is stored
            statellite_names (List[str])    : The name of the satellites to load the data frame
            year_frame (Tuple[str, str])    : The year range to load data from (min 'YYYY', max 'YYYY')
            month_frame (Tuple[str, str])   : The month range to load data from (min 'MM', max 'MM')
            day_frame (Tuple[str, str])     : The day range to load data from (min 'DD', max 'DD')
            hour_frame (Tuple[str, str])    : The hour range to load data from (min 'HH:MM:SS', max 'HH:MM:SS')
        """
        # SATELLITE NAMES
        self.satellite_names = satellite_name
        available_satellites = ['envisat', 'geosat2', 'jason1', 'karin_swot', 'nadir_swot', 'topex-poseidon_interleaved']
        if not set([satellite_name]).issubset(set(available_satellites)):
            raise ValueError("Invalid satellite name provided. Satellite names must be in: {}".format(available_satellites))
        # FILE PATHS
        self.ref_root_folder = ref_root_folder
        assert os.path.exists(self.ref_root_folder), "The reference root folder does not exist: {}".format(self.ref_root_folder)
        self.obs_root_folder = obs_root_folder
        assert os.path.exists(self.obs_root_folder), "The observation root folder does not exist: {}".format(self.obs_root_folder)
        self.obs_file_path = os.path.join(obs_root_folder, f'2020a_SSH_mapping_NATL60_{satellite_name}.nc')
        assert os.path.exists(self.obs_file_path), "The observation file path does not exist: {}".format(self.obs_file_path)
        # META
        self.year_frame = year_frame
        self.month_frame = month_frame
        self.day_frame = day_frame
        self.hour_frame = hour_frame
        # DATA
        self.obs_data = self._load_obs_data()
        self.ref_data = self._load_ref_data()

    def _load_obs_data(self,) -> xr.Dataset:
        """ loads the observation data for the specified satellite between the given time frame into an xarray dataset """
        # get the start and end data in 'YYYY-MM-DD HH:MM:SS' format
        iso_start_date = '-'.join([self.year_frame[0], self.month_frame[0], self.day_frame[0]]) + ' ' + self.hour_frame[0]
        iso_end_date = '-'.join([self.year_frame[1], self.month_frame[1], self.day_frame[1]]) + ' ' + self.hour_frame[1]
        # load the data sliced by the time frame
        ds = xr.open_dataset(self.obs_file_path).sel(time = slice(iso_start_date, iso_end_date))
        return ds
    
    def _load_ref_data(self,) -> xr.Dataset:
        """ loads the referece data for the specified satellite between the given time frame into an xarray dataset """
        # get the satart and end data in 'YYYY-MM-DD' format
        iso_start_date_str = '-'.join([self.year_frame[0], self.month_frame[0], self.day_frame[0]])
        iso_end_date_str = '-'.join([self.year_frame[1], self.month_frame[1], self.day_frame[1]])
        # Convert the start and end date strings to datetime objects
        iso_start_date = datetime.strptime(iso_start_date_str, '%Y-%m-%d')
        iso_end_date = datetime.strptime(iso_end_date_str, '%Y-%m-%d')
        # Calculate the range of dates between the start and end dates
        date_range = [iso_start_date + timedelta(days=x) for x in range((iso_end_date - iso_start_date).days + 1)]
        # Convert the datetime objects back to 'YYYY-MM-DD' formatted strings
        iso_dates_str = [date.strftime('%Y-%m-%d') for date in date_range]
        # get the file paths for the reference data
        ref_file_names = ['NATL60-CJM165_GULFSTREAM_y{year}m{month}d{day}.1h_SSH.nc'.format(year = date[0:4], month = date[5:7], day = date[8:10]) for date in iso_dates_str]
        ref_file_paths = [os.path.join(self.ref_root_folder, file_name) for file_name in ref_file_names]
        # mask the files that exist
        file_mask = [os.path.exists(file_path) for file_path in ref_file_paths]
        # remove the file paths that do not exist
        ref_file_paths = [file_path for file_path, mask in zip(ref_file_paths, file_mask) if mask]
        # load the data
        ds = xr.open_mfdataset(ref_file_paths, combine = 'by_coords')
        # average the data over time
        return ds
    
    def grid_ref_data_average(self, 
                              n_grids : int) -> np.ndarray:
        """ 
        grids the reference data (averaged across time) by averaging the data in each grid cell  
        
        Arguments:
            n_grids (int)   : The number of grid cells to split the data into (e.g. 1 grid cell = 600x600, 4 grid cells = 300x300)

        Returns:
            np.ndarray      : The gridded reference data (n_grids x n_grids)
        """
        # no. of point / grid cell
        N_SPLINES = n_grids
        N_POINTS = int(600 / n_grids)
        # average dataset over time
        ds_mean = self.ref_data.mean(dim='time')
        # compute the averages
        averages = []
        for i in range(N_SPLINES):
            for j in range(N_SPLINES):
                avg = ds_mean.sossheig.values[i*N_POINTS:(i+1)*N_POINTS, j*N_POINTS:(j+1)*N_POINTS].flatten().mean()
                averages.append(avg)
        return np.array(averages).reshape(N_SPLINES, N_SPLINES)
    
    def grid_ref_data_trapz(self, 
                              n_grids : int) -> np.ndarray:
        """ 
        grids the reference data (averaged across time) by numerically integrating the data in each grid cell by the trapezoidal rule
        
        Arguments:
            n_grids (int)   : The number of grid cells to split the data into (e.g. 1 grid cell = 600x600, 4 grid cells = 300x300)

        Returns:
            np.ndarray      : The gridded reference data (n_grids x n_grids)
        """
        # no. of point / grid cell
        N_SPLINES = n_grids
        N_POINTS = int(600 / n_grids)
        # average dataset over time
        ds_mean = self.ref_data.mean(dim='time')
        # compute the integrals
        integrals = []
        for i in range(N_SPLINES):
            for j in range(N_SPLINES):
                # get values
                lon_vals = ds_mean.lon.values[i*N_POINTS:(i+1)*N_POINTS]
                lat_vals = ds_mean.lat.values[j*N_POINTS:(j+1)*N_POINTS]
                ssh_vals = ds_mean.sossheig.values[i*N_POINTS:(i+1)*N_POINTS, j*N_POINTS:(j+1)*N_POINTS]
                cell_integral = np.trapz(np.trapz(ssh_vals, dx=lon_vals[1] - lon_vals[0], axis=1), dx=lat_vals[1] - lat_vals[0])
                integrals.append(cell_integral)
        return np.array(integrals).reshape(N_SPLINES, N_SPLINES)

    def grid_ref_data_simpson(self, 
                              n_grids : int) -> np.ndarray:
        """ 
        grids the reference data (averaged across time) by numerically integrating the data in each grid cell by Simpson's rule
        
        Arguments:
            n_grids (int)   : The number of grid cells to split the data into (e.g. 1 grid cell = 600x600, 4 grid cells = 300x300)

        Returns:
            np.ndarray      : The gridded reference data (n_grids x n_grids)
        """
        # no. of point / grid cell
        N_SPLINES = n_grids
        N_POINTS = int(600 / n_grids)
        # average dataset over time
        ds_mean = self.ref_data.mean(dim='time')
        # compute the integrals
        integrals = []
        for i in range(N_SPLINES):
            for j in range(0, N_SPLINES):
                # get values
                lon_vals = ds_mean.lon.values[i*N_POINTS:(i+1)*N_POINTS]
                lat_vals = ds_mean.lat.values[j*N_POINTS:(j+1)*N_POINTS]
                ssh_vals = ds_mean.sossheig.values[i*N_POINTS:(i+1)*N_POINTS, j*N_POINTS:(j+1)*N_POINTS]
                cell_integral = simpson(simpson(ssh_vals, dx=lon_vals[1] - lon_vals[0], axis=1), dx=lat_vals[1] - lat_vals[0])
                integrals.append(cell_integral)
        return np.array(integrals).reshape(N_SPLINES, N_SPLINES)
