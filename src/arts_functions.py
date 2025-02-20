import numpy as np
import pyarts
from scipy.optimize import curve_fit
import xarray as xr
import pandas as pd
from scipy.stats import linregress
from pyarts.workspace import arts_agenda


def setup_workspace(verbosity=0):
    """Set up ARTS workspace.

    Returns:
        Workspace: ARTS workspace.
    """

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.iy_main_agendaSet(option="Emission")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Absorption species
    ws.abs_speciesSet(
        species=[
            "H2O-PWR2022",
            "O2-PWR2022",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "O3",
        ]
    )

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")

    return ws


def basic_setup(f_grid, sensor_description=[], version="2.6.8", verbosity=0):
    """
    Sets up a basic ARTS workspace configuration for radiative transfer calculations.
    This function initializes an ARTS workspace with standard settings for atmospheric
    radiative transfer calculations, particularly focused on microwave sensors and
    tropospheric applications.
    Parameters
    ----------
    f_grid : numpy.ndarray
        Frequency grid for calculations. Must be provided if sensor_description is not used.
    sensor_description : list, optional
        AMSU sensor description parameters. Cannot be used simultaneously with f_grid.
        Default is empty list.
    version : str, optional
        Version of ARTS catalogue to be downloaded. Default is "2.6.8".
    verbosity : int, optional
        Level of output verbosity. Default is 0 (minimal output).
    Returns
    -------
    ws : pyarts.workspace.Workspace
        Configured ARTS workspace instance.
    Raises
    ------
    ValueError
        If both f_grid and sensor_description are provided simultaneously,
        or if neither f_grid nor sensor_description is provided.
    Notes
    -----
    The function sets up:
    - Standard emission calculation
    - Cosmic background radiation
    - Surface properties (non-reflecting surface)
    - Absorption species (H2O, O2, N2 using Rosenkranz models)
    - Planck brightness temperature as output unit
    - 1D atmosphere
    - Path calculation without refraction
    For sensor description, the function includes an iterative adjustment mechanism
    that modifies the frequency spacing if initial sensor response generation fails.
    """

    pyarts.cat.download.retrieve(verbose=True, version=version)

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # standard emission agenda
    ws.iy_main_agendaSet(option="Emission")

    # cosmic background radiation
    ws.iy_space_agendaSet(option="CosmicBackground")

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    # sensor-only path
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # no refraction
    ws.ppath_step_agendaSet(option="GeometricPath")

    # Non reflecting surface
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    #########################################################################

    # Definition of absorption species
    # We use predefined models for H2O, O2, and N2 from Rosenkranz
    # as they are fast and accurate for microwave and tropospheric retrieval.
    ws.abs_speciesSet(
        species=[
            "H2O-PWR2022",
            "O2-PWR2022",
            "N2-SelfContPWR2021",
        ]
    )

    ws.abs_lines_per_speciesSetEmpty()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    ws.AtmosphereSet1D()

    it_max = 5

    if np.size(f_grid) == 0 and np.size(sensor_description) > 0:
        iterate = True
        N_it = 0
        while iterate and N_it < it_max:
            N_it += 1
            try:
                ws.sensor_description_amsu = sensor_description
                ws.sensor_responseGenericAMSU(spacing=1e12)
                iterate = False
            except RuntimeError:
                rel_change = 0.9

                print(
                    f"adjusting relative mandatory minimum frequency spacing by factor {rel_change}"
                )

                # adjust relative mandatory minimum frequency spacing
                sensor_description[:, -1] *= rel_change

    elif np.size(f_grid) > 0 and np.size(sensor_description) > 0:
        raise ValueError(
            "f_grid and sensor_description cannot be provided simultaneously"
        )

    elif np.size(f_grid) > 0 and np.size(sensor_description) == 0:
        # Set the frequency grid
        ws.f_grid = f_grid
        ws.sensorOff()
    else:
        raise ValueError("f_grid or sensor_description must be provided")

    # on-the-fly absorption
    ws.propmat_clearsky_agendaAuto()
    ws.abs_lines_per_speciesSetEmpty()

    # switch off jacobian calculation by default
    ws.jacobianOff()

    return ws


def Hamp_channels(band_selection, rel_mandatory_grid_spacing=1.0 / 4.0):
    """
    Returns sensor description and characteristics for HAMP (Humidity And Temperature Profiler) channels.

    This function provides frequency specifications and sensor characteristics for different
    frequency bands (K, V, W, F, G) of the HAMP instrument. Each band contains multiple channels
    with specific center frequencies, offsets, and other parameters.

    Parameters
    ----------
    band_selection : list
        List of strings indicating which frequency bands to include.
        Valid options are 'K', 'V', 'W', 'F', and 'G'.
        If empty list is provided, prints available bands and their specifications.
    rel_mandatory_grid_spacing : float, optional
        Relative mandatory frequency grid spacing for the passbands.
        Default is 0.25 (1/4). This means that the mandatory grid spacing is 1/4 of the
        passbands bandwidth.

    Returns
    -------
    tuple or None
        If band_selection is empty, returns None and prints available bands.
        Otherwise returns tuple of (sensor_description, NeDT, Accuracy, FWHM_Antenna):
            - sensor_description : ndarray
                Array of [frequency, offset1, offset2, bandwidth, df] for each channel
            - NeDT : ndarray
                Noise equivalent differential temperature for each channel
            - Accuracy : ndarray
                Accuracy in Kelvin for each channel
            - FWHM_Antenna : ndarray
                Full Width at Half Maximum of antenna beam pattern in degrees

    Raises
    ------
    ValueError
        If an invalid band is specified in band_selection.

    Notes
    -----
    Frequency bands:
    - K band: 7 channels around 22-31 GHz
    - V band: 7 channels around 50-58 GHz
    - W band: 1 channel at 90 GHz
    - F band: 4 channels around 118.75 GHz
    - G band: 6 channels around 183.31 GHz
    """

    channels = {}
    channels["K"] = {
        "f_center": np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]) * 1e9,
        "Offset1": np.zeros(7),
        "Offset2": np.zeros(7),
        "NeDT": 0.1,
        "Accuracy": 0.5,
        "Bandwidth": 230e6,
        "FWHM_Antenna": 5.0,
        "df": rel_mandatory_grid_spacing,
    }
    channels["V"] = {
        "f_center": np.array([50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]) * 1e9,
        "Offset1": np.zeros(7),
        "Offset2": np.zeros(7),
        "NeDT": 0.2,
        "Accuracy": 0.5,
        "Bandwidth": 230e6,
        "FWHM_Antenna": 3.5,
        "df": rel_mandatory_grid_spacing,
    }
    channels["W"] = {
        "f_center": np.array([90]) * 1e9,
        "Offset1": np.zeros(1),
        "Offset2": np.zeros(1),
        "NeDT": 0.25,
        "Accuracy": 1.5,
        "Bandwidth": 2e9,
        "FWHM_Antenna": 3.3,
        "df": rel_mandatory_grid_spacing,
    }

    channels["F"] = {
        "f_center": np.ones(4) * 118.75e9,
        "Offset1": np.array([1.4, 2.3, 4.2, 8.5]) * 1e9,
        "Offset2": np.zeros(4),
        "NeDT": 0.6,
        "Accuracy": 1.5,
        "Bandwidth": 400e6,
        "FWHM_Antenna": 3.3,
        "df": rel_mandatory_grid_spacing,
    }

    channels["G"] = {
        "f_center": np.ones(6) * 183.31e9,
        "Offset1": np.array([0.6, 1.5, 2.5, 3.5, 5.0, 7.5]) * 1e9,
        "Offset2": np.zeros(6),
        "NeDT": 0.6,
        "Accuracy": 1.5,
        "Bandwidth": np.array([200e6, 200e6, 200e6, 200e6, 200e6, 1000e6]),
        "FWHM_Antenna": 2.7,
        "df": rel_mandatory_grid_spacing,
    }

    if len(band_selection) == 0:
        print("No band selected")
        print("Following bands are available:\n")
        for key in channels.keys():
            print(f"{key} =====================================================")
            print(f'f_grid: {channels[key]["f_center"]} Hz')
            print(f'Offset1: {channels[key]["Offset1"]} Hz')
            print(f'Offset2: {channels[key]["Offset2"]} Hz')
            print(f'NeDT: {channels[key]["NeDT"]} K')
            print(f'Accuracy: {channels[key]["Accuracy"]} K')
            print(f'Bandwidth: {channels[key]["Bandwidth"]} Hz')
            print(f'FWHM_Antenna: {channels[key]["FWHM_Antenna"]} deg')
            print(f'df: {channels[key]["df"]}')
            print("=====================================================\n")
        return

    else:
        sensor_description = []
        NeDT = []
        Accuracy = []
        FWHM_Antenna = []

        for band in band_selection:
            if band in channels.keys():
                for i in range(len(channels[band]["f_center"])):
                    freq = channels[band]["f_center"][i]
                    offset1 = channels[band]["Offset1"][i]
                    offset2 = channels[band]["Offset2"][i]

                    if isinstance(channels[band]["Bandwidth"], float):
                        bandwidth = channels[band]["Bandwidth"]
                    else:
                        bandwidth = channels[band]["Bandwidth"][i]

                    desc_i = [
                        freq,
                        offset1,
                        offset2,
                        bandwidth,
                        bandwidth * channels[band]["df"],
                    ]

                    sensor_description.append(desc_i)
                    NeDT.append(channels[band]["NeDT"])
                    Accuracy.append(channels[band]["Accuracy"])
                    FWHM_Antenna.append(channels[band]["FWHM_Antenna"])

            else:
                raise ValueError(f"Band {band} not available")

        return (
            np.array(sensor_description),
            np.array(NeDT),
            np.array(Accuracy),
            np.array(FWHM_Antenna),
        )


def set_sensor_position_and_view(ws, sensor_pos, sensor_los):
    """Set sensor position and line-of-sight direction in workspace.
    This function sets the sensor position and line-of-sight (viewing direction)
    in the workspace for radiative transfer calculations.
    Parameters
    ----------
    ws : Workspace
        ARTS workspace object where sensor parameters will be set
    sensor_pos : array-like
        Sensor position coordinates (e.g., [x,y,z])
    sensor_los : array-like
        Line-of-sight direction vector (e.g., [dx,dy,dz])
    Returns
    -------
    None
        Modifies workspace in-place by setting sensor_pos and sensor_los variables
    """

    ws.sensor_pos = np.array([[sensor_pos]])
    ws.sensor_los = np.array([[sensor_los]])


def forward_model(
    ws,
    atm_fields_compact,
    surface_windspeed,
    surface_temperature,
    sensor_altitude,
    sensor_los=180,
    retrieval_quantity="",
):
    """
    Performs radiative transfer calculations using ARTS (Atmospheric Radiative Transfer Simulator).
    This function sets up and executes forward model calculations, optionally including Jacobian
    calculations for retrievals of water vapor (H2O) or temperature (T).
    Parameters
    ----------
    ws : arts.Workspace
        ARTS workspace object containing the computational environment.
    atm_fields_compact : arts.AtmFieldsCompact
        Compact representation of atmospheric fields.
    surface_reflectivity : float
        Surface reflectivity value (between 0 and 1).
    surface_temperature : float
        Surface temperature value in Kelvin.
    retrieval_quantity : str, optional
        Specifies the quantity to be retrieved. Must be either 'H2O' or 'T'.
        If empty, no Jacobian is calculated.
    Returns
    -------
    tuple
        - numpy.ndarray: Calculated radiances (ws.y value)
        - arts.Matrix or empty Matrix: Jacobian matrix if retrieval_quantity is specified,
          otherwise an empty matrix
    Notes
    -----
    The function performs the following main steps:
    1. Sets up atmospheric fields including N2 and O2 if not present
    2. Configures surface properties
    3. Sets sensor position and line of sight
    4. Calculates Jacobians if retrieval_quantity is specified
    5. Performs radiative transfer calculations in clear-sky conditions
    The calculation assumes no scattering (cloudbox is turned off).
    Raises
    ------
    ValueError
        If retrieval_quantity is neither 'H2O' nor 'T' when specified.
    """

    #########################################################################

    set_sensor_position_and_view(ws, sensor_altitude, sensor_los)

    # Atmosphere and surface

    ws.atm_fields_compact = atm_fields_compact

    # check if N2 and O2 in atm_fields_compact
    if "abs_species-N2" not in atm_fields_compact.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-N2", 0.7808, 0
        )

    if "abs_species-O2" not in atm_fields_compact.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-O2", 0.2095, 0
        )

    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    ws.Extract(ws.z_surface, ws.z_field, 0)

    # configure surface emissions
    ws.NumericCreate("wspeed")
    ws.VectorCreate("trans")
    ws.NumericCreate("surface_temperature")
    ws.VectorCreate("transmittance")
    ws.IndexCreate("nf")

    # Set surface temperature equal to the lowest atmosphere level
    ws.wspeed = surface_windspeed
    ws.surface_skin_t = surface_temperature
    ws.surface_temperature = surface_temperature
    ws.transmittance = np.ones(ws.f_grid.value.shape)

    # agenda for surface properties
    @arts_agenda
    def surface_rtprop_agenda_fastem(ws):
        ws.Copy(ws.surface_skin_t, ws.surface_temperature)
        ws.specular_losCalc()
        ws.nelemGet(ws.nf, ws.f_grid)
        ws.VectorSetConstant(ws.trans, ws.nf, 1.0)
        ws.surfaceFastem(
            salinity=0.034, wind_speed=ws.wspeed, transmittance=ws.transmittance
        )

    ws.surface_rtprop_agenda = surface_rtprop_agenda_fastem(ws)

    #########################################################################

    # Jacobian calculation
    if len(retrieval_quantity) > 0:
        ws.jacobianInit()
        if retrieval_quantity == "H2O":
            ws.jacobianAddAbsSpecies(
                g1=ws.p_grid,
                g2=ws.lat_grid,
                g3=ws.lon_grid,
                species="H2O-PWR2022",
                unit="vmr",
            )
        elif retrieval_quantity == "T":
            ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
        else:
            raise ValueError("only H2O or T are allowed as retrieval quantity")
        ws.jacobianClose()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Perform RT calculations
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.yCalc()

    if len(retrieval_quantity) > 0:
        jacobian = ws.jacobian.value[:].copy()
    else:
        jacobian = pyarts.arts.Matrix()

    return ws.y.value[:].copy(), jacobian


def run_arts(
    pressure_profile,
    temperature_profile,
    h2o_profile,
    surface_ws,
    surface_temp,
    ws: pyarts.workspace.Workspace,
    N2=0.78,
    O2=0.21,
    O3=1e-6,
    zenith_angle=180,
    height=None,
    surface_altitude=0.0,
    frequencies=None,
):
    """Perform a radiative transfer simulation.

    Parameters:
        species (list[str]): List of species tags.
        zenith_angle (float): Viewing angle [deg].
        height (float): Sensor height [m].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.

    Returns:
        ndarray, ndarray, ndarray:
          Frequency grid [Hz], Brightness temperature [K], Optical depth [1]
    """

    # Set frequencies
    ws.f_grid = np.array(frequencies)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # No sensor properties
    ws.sensorOff()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    # Extract optical depth as auxiliary variables
    ws.ArrayOfStringSet(ws.iy_aux_vars, ["Optical depth"])

    # Atmosphere and surface
    ws.Touch(ws.lat_grid)
    ws.Touch(ws.lon_grid)
    ws.lat_true = np.array([0.0])
    ws.lon_true = np.array([0.0])

    ws.AtmosphereSet1D()
    ws.p_grid = pressure_profile
    ws.t_field = temperature_profile[:, np.newaxis, np.newaxis]

    vmr_field = np.zeros((4, len(pressure_profile), 1, 1))
    vmr_field[0, :, 0, 0] = h2o_profile
    vmr_field[1, :, 0, 0] = O2
    vmr_field[2, :, 0, 0] = N2
    vmr_field[3, :, 0, 0] = O3
    ws.vmr_field = vmr_field

    ws.z_surface = np.array([[surface_altitude]])
    ws.p_hse = 100000
    ws.z_hse_accuracy = 100.0
    ws.z_field = 16e3 * (5 - np.log10(pressure_profile[:, np.newaxis, np.newaxis]))
    ws.atmfields_checkedCalc()
    ws.z_fieldFromHSE()

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, np.array([[height]]))
    ws.MatrixSet(ws.sensor_los, np.array([[zenith_angle]]))

    # configure surface emissions
    ws.IndexCreate("nf")
    ws.VectorCreate("trans")
    ws.NumericCreate("wspeed")
    ws.NumericCreate("surface_temperature")

    # Set surface temperature equal to the lowest atmosphere level
    ws.wspeed = surface_ws
    ws.surface_skin_t = surface_temp
    ws.surface_temperature = surface_temp

    # agenda for surface properties
    @arts_agenda
    def surface_rtprop_agenda_tessem(ws):
        ws.Copy(ws.surface_skin_t, ws.surface_temperature)
        ws.specular_losCalc()

        ws.nelemGet(ws.nf, ws.f_grid)
        ws.VectorSetConstant(ws.trans, ws.nf, 1.0)

        ws.surfaceFastem(
            salinity=0.034, wind_speed=ws.wspeed, transmittance=ws.transmittance
        )

    ws.VectorCreate("transmittance")
    ws.transmittance = np.ones(ws.f_grid.value.shape)
    ws.surface_rtprop_agenda = surface_rtprop_agenda_tessem(ws)

    # Perform RT calculations
    ws.propmat_clearsky_agendaAuto()  # Calculate the absorption coefficient matrix automatically
    ws.lbl_checkedCalc()  # checks if line-by-line parameters are ok
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.yCalc()

    return (
        ws.f_grid.value[:].copy(),
        ws.y.value[:].copy(),
        ws.y_aux.value[0][:].copy(),
    )


def exponential(x, a, b):
    return a * np.exp(b * x)


def fit_exponential(x, y, p0):
    nanmask = np.isnan(y) | np.isnan(x)
    if np.sum(nanmask) > 0:
        popt, _ = curve_fit(exponential, x[~nanmask], y[~nanmask], p0=p0)
        offset = y[~nanmask][-1]
        idx_nan = np.where(nanmask)[0]
        nanmask[idx_nan - 1] = True  # get overlap of one
        filled = np.zeros_like(y)
        filled[~nanmask] = y[~nanmask]
        new_vals = exponential(x[nanmask], *popt)
        filled[nanmask] = new_vals - new_vals[0] + offset
        return filled
    else:
        return y


def fit_linear(x, y, upper_val, height):
    nanmask = np.isnan(y) | np.isnan(
        x
    )  # nan's exist under plane where we want to extrapolate
    last_val = y[~nanmask][-1]
    last_height = x[~nanmask][-1]
    idx_nan = np.where(nanmask)[0]
    nanmask[idx_nan - 1] = True  # get overlap of one
    slope = (upper_val - last_val) / (height - last_height)
    filled = np.zeros_like(y)
    filled[~nanmask] = y[~nanmask]
    new_vals = slope * (x[nanmask] - last_height) + last_val
    filled[nanmask] = new_vals
    return filled


def fit_regression(x, y):
    nanmask = np.isnan(y) | np.isnan(
        x
    )  # nan's exist under plane where we want to extrapolate
    last_val = y[~nanmask][-1]
    last_height = x[~nanmask][-1]
    idx_nan = np.where(nanmask)[0]
    nanmask[idx_nan - 1] = True  # get overlap of one
    slope = linregress(x[~nanmask], y[~nanmask]).slope
    filled = np.zeros_like(y)
    filled[~nanmask] = y[~nanmask]
    new_vals = slope * (x[nanmask] - last_height) + last_val
    filled[nanmask] = new_vals
    return filled


def fit_constant(x, y):
    nanmask = np.isnan(y) | np.isnan(
        x
    )  # nan's exist under plane where we want to extrapolate
    last_val = y[~nanmask][-1]
    idx_nan = np.where(nanmask)[0]
    nanmask[idx_nan - 1] = True  # get overlap of one
    filled = np.zeros_like(y)
    filled[~nanmask] = y[~nanmask]
    filled[nanmask] = last_val
    return filled


def get_profiles(sonde, ds_dropsonde, radiometers):
    ds_dropsonde_loc = ds_dropsonde.sel(sonde=sonde)
    drop_time = ds_dropsonde_loc["sonde_time"].values
    hampdata_loc = radiometers.dropna("time").sel(time=drop_time, method="nearest")
    height = float(hampdata_loc.plane_altitude.values)
    return ds_dropsonde_loc, hampdata_loc, height, drop_time


def extrapolate_dropsonde(ds_dropsonde, height):
    # drop nans
    ds_dropsonde = ds_dropsonde.where(ds_dropsonde["altitude"] < height, drop=True)
    bool = (ds_dropsonde["p"].isnull()) & (
        ds_dropsonde["altitude"] < 100
    )  # drop nans at lower levels
    ds_dropsonde = ds_dropsonde.where(~bool, drop=True)

    p_extrap = fit_exponential(
        ds_dropsonde["altitude"].values,
        ds_dropsonde["p"].interpolate_na("altitude").values,
        p0=[1e5, -0.0001],
    )

    ta_extrap = fit_regression(
        ds_dropsonde["altitude"].values,
        ds_dropsonde["ta"].interpolate_na("altitude").values,
    )

    q_extrap = fit_constant(
        ds_dropsonde["altitude"].values,
        ds_dropsonde["q"].interpolate_na("altitude").values,
    )

    return xr.Dataset(
        {
            "p": (("altitude"), p_extrap),
            "ta": (("altitude"), ta_extrap),
            "q": (("altitude"), q_extrap),
        },
        coords={"altitude": ds_dropsonde["altitude"].values},
    )


def get_surface_temperature(dropsonde):
    """
    Get temperature at lowest level of dropsonde which is not nan.

    Parameters:
        dropsonde (xr.Dataset): Dropsonde data.

    Returns:
        float: Surface temperature.
    """

    T_surf = dropsonde["ta"].where(~dropsonde["ta"].isnull(), drop=True).values[0]
    return T_surf


def get_surface_windspeed(dropsonde):
    """
    Get windspeed at lowest level of dropsonde which is not nan.

    Parameters:
        dropsonde (xr.Dataset): Dropsonde data.

    Returns:
        float: Surface windspeed.
    """
    u = dropsonde["u"].where(~dropsonde["u"].isnull(), drop=True).values[0]
    v = dropsonde["v"].where(~dropsonde["v"].isnull(), drop=True).values[0]
    return np.sqrt(u**2 + v**2)


def is_complete(dropsonde, hamp, drop_time, height, sonde):
    """
    Check if data is complete and valid.

    Parameters:
        dropsonde (xr.Dataset): Dropsonde data.
        bahamas (xr.Dataset): Bahamas data.
        hamp (xr.Dataset): HAMP data.

    Returns:
        bool: True if data is valid, False otherwise.
    """

    if (
        (dropsonde["ta"].isnull().mean() == 1)
        or (dropsonde["p"].isnull().mean() == 1)
        or (dropsonde["q"].isnull().mean() == 1)
    ):
        print(f"Dropsonde {sonde} contains nan only, skipping")
        return False

    if dropsonde["p"].dropna("altitude").diff("altitude").max() > 0:
        print(f"Dropsonde {sonde} pressure is not monotonically decreasing, skipping")
        return False

    if dropsonde["p"].max() < 900e2:
        print(f"Dropsonde {sonde} pressure is below 900 hPa, skipping")
        return

    if np.isnan(height):
        print(f"Bahamas at {drop_time} contains nan, skipping")
        return False

    if hamp["TBs"].isnull().mean().values == 1:
        print(f"HAMP data at {drop_time} contains nan, skipping")
        return False

    if hamp["time"].values - drop_time > pd.Timedelta("2 minutes"):
        print(f"HAMP data at {drop_time} is more than 2 minutes away, skipping")
        return False

    if (
        dropsonde["altitude"].where(~dropsonde["ta"].isnull(), drop=True).values[0] > 20
    ) or (
        dropsonde["altitude"].where(~dropsonde["u"].isnull(), drop=True).values[0] > 20
    ):
        print(f"Dropsonde {sonde} lowest altitude is above 20 m, skipping")
        return False

    return True
