import yaml


def extract_config_params(config_file):
    """Load configuration from YAML file,
    return dict with correctly formated configuration parameters"""

    config = {}
    with open(config_file, "r") as file:
        print(f"Reading config YAML: '{config_file}'")
        config_yaml = yaml.safe_load(file)

    config["path_radar"] = config_yaml["radar"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    config["path_radiometers"] = config_yaml["radiometer"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    config["path_iwv"] = config_yaml["iwv"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    config["path_saveplots"] = config_yaml["path_saveplots"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    config["flightname"] = config_yaml["flightname"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )
    config["date"] = config_yaml["date"]
    config["path_dropsondes"] = config_yaml["path_dropsondes"]
    config["path_position_attitude"] = config_yaml["position_attitude"].format(
        date=config_yaml["date"], flightletter=config_yaml["flightletter"]
    )

    return config
