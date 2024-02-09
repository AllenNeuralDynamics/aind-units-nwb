import json
import warnings
from pathlib import Path
from packaging.version import parse

from pynwb.file import Device


def get_devices_from_metadata(session_folder, segment_index=0):
    """
    Return NWB devices from metadata target locations.

    The schemas used to pupulate the NWBFile and metadata dictionaries are:
    - session.json
    - rig.json

    Parameters
    ----------
    session_folder : str or Path
        The path to the session folder
    segment_index : int
        The segment index to instantiate NWBFile for.
        This is needed to correctly instantiate devices and their locations.

    Returns
    -------
    added_devices: dict (device_name: pynwb.Device) or None
        The instantiated Devices with AIND metadata
    devices_target_location: dict
        Dict with device name to target location
    """
    session_folder = Path(session_folder)
    session_file = session_folder / "session.json"
    rig_file = session_folder / "rig.json"

    # load json files
    session = None
    if session_file.is_file():
        with open(session_file, "r") as f:
            session = json.load(f)

    rig = None
    if rig_file.is_file():
        with open(rig_file, "r") as f:
            rig = json.load(f)

    data_streams = None
    if session is not None:
        if "schema_version" in session:
            session_schema_version = session["schema_version"]
            if parse(session_schema_version) >= parse("0.3.0"):
                data_streams = session["data_streams"]
            else:
                warnings.warn(f"v{session_schema_version} for session schema is not currently supported")

    # Add devices here
    devices = None
    devices_target_location = {}
    if rig is not None and data_streams is not None:
        if "schema_version" in rig:
            rig_schema_version = rig["schema_version"]
            if parse(rig_schema_version) >= parse("0.5.1"):
                if data_streams is not None:
                    probes_in_session = session["data_streams"][segment_index]["probes"]
                    ephys_modules = rig["ephys_modules"]

                    if len(probes_in_session) <= len(ephys_modules):
                        for probe in probes_in_session:
                            probe_name = probe["name"]
                            for ephys_module in ephys_modules:
                                probe_info = ephys_module["probes"][0]
                                if probe_info["name"] == probe_name:
                                    probe_device_name = probe_info["name"]
                                    probe_device_description = f"Model: {probe_info['probe_model']} - Serial number: {probe_info['serial_number']}"
                                    probe_device_manufacturer = f"{probe_info['manufacturer']}"
                                    probe_device = Device(
                                        name=probe_device_name,
                                        description=probe_device_description,
                                        manufacturer=probe_device_manufacturer,
                                    )
                                    if devices is None:
                                        devices = {}
                                    devices[probe_device_name] = probe_device
                                    devices_target_location[probe_device_name] = probe["primary_targeted_structure"]

                                    # Add internal lasers for NP-opto
                                    if "lasers" in probe_info and len(probe_info["lasers"]) > 1:
                                        for laser in probe_info["lasers"]:
                                            laser_device_name = laser["name"]
                                            laser_device_description = f"Type: internal - Wavelength: {laser['wavelength']}nm - Max power: {laser['maximum_power']}uW - Coupling: {laser['coupling']}"
                                            laser_device_manufacturer = laser["manufacturer"]
                                            internal_laser_device = Device(
                                                name=laser_device_name,
                                                description=laser_device_description,
                                                manufacturer=laser_device_manufacturer,
                                            )
                                            if devices is None:
                                                devices = {}
                                            devices[laser_device_name] = internal_laser_device

                    # # TODO: Add external lasers should be added by other capsules
                    # if "laser_modules" in rig:
                    #     for laser in rig["laser_modules"][0]["lasers"]:
                    #         laser_device_name = laser["name"]
                    #         laser_device_description = f"Type: external - Wavelength: {laser['wavelength']}nm"
                    #         if "coupling" in laser:
                    #             laser_device_description += f" - Coupling: {laser['coupling']}"
                    #         laser_device_manufacturer = laser['manufacturer'] # TODO get this info from rig.json
                    #         external_laser_device = Device(name=laser_device_name,
                    #                                        description=laser_device_description,
                    #                                        manufacturer=laser_device_manufacturer)
                    #       if added_devices is None:
                    #           added_devices = {}
                    #       added_devices[laser_device_name] = internal_laser_device

            #                     else:
            #                         warnings.warn(f"Inconsistency between rig and session devices")
            else:
                warnings.warn(f"v{rig_schema_version} for rig schema is not currently supported")

    return devices, devices_target_location
