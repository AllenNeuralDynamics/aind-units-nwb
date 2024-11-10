import json
import warnings
from pathlib import Path
from typing import Union, List, Optional, Literal
from packaging.version import parse
import numpy as np

from spikeinterface import BaseRecording, SortingAnalyzer

import pynwb
from pynwb.file import Device


def get_devices_from_rig_metadata(session_folder: str, segment_index: int = 0):
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
        session_schema_version = session.get("schema_version", None)

        if session_schema_version is None:
            warnings.warn(f"Session file does not have schema_version")
            return devices, devices_target_location
        if parse(session_schema_version) >= parse("0.3.0"):
            data_streams = session.get("data_streams", None)
            if data_streams is None:
                warnings.warn(f"Session file does not have data_streams")
                return None, None
        else:
            warnings.warn(f"v{session_schema_version} for session schema is not currently supported")
            return None, None
    else:
        warnings.warn(f"Session file not found in {session_folder}")
        return None, None

    stimulus_epochs = session.get("stimulus_epochs", None)
    stimulus_device_names = []
    if stimulus_epochs is not None:
        for epoch in stimulus_epochs:
            stimulus_device_names += epoch.get("stimulus_device_names", [])

    devices = None
    devices_target_location = None
    if rig is not None:
        rig_schema_version = rig.get("schema_version", None)
        if rig_schema_version is None:
            warnings.warn(f"Rig file does not have schema_version")
        elif parse(rig_schema_version) >= parse("0.5.1"):
            ephys_modules = session["data_streams"][segment_index]["ephys_modules"]
            ephys_assemblies = rig.get("ephys_assemblies", [])
            laser_assemblies = rig.get("laser_assemblies", [])

            # gather all probes and lasers
            probe_devices = {}
            laser_devices = {}
            for ephys_assembly in ephys_assemblies:
                probes_in_assembly = ephys_assembly["probes"]

                for probe_info in probes_in_assembly:
                    probe_device_name = probe_info["name"]
                    probe_model_name = probe_info.get("probe_model", None)
                    probe_device_manufacturer = probe_info.get("manufacturer", None)
                    probe_serial_number = probe_info.get("serial_number", None)
                    probe_device_description = ""
                    if probe_device_name is None:
                        if probe_model_name is not None:
                            probe_device_name = probe_model_name
                        else:
                            probe_device_name = "Probe"
                    if probe_model_name is not None:
                        probe_device_description += f"Model: {probe_device_description}"
                    if probe_serial_number is not None:
                        if len(probe_device_description) > 0:
                            probe_device_description += " - "
                        probe_device_description += f"Serial number: {probe_serial_number}"
                    probe_device = Device(
                        name=probe_device_name,
                        description=probe_device_description,
                        manufacturer=probe_device_manufacturer,
                    )
                    if probe_device_name not in probe_devices:
                        probe_devices[probe_device_name] = probe_device
                    # Add internal lasers for NP-opto
                    if "lasers" in probe_info and len(probe_info["lasers"]) > 1:
                        for laser in probe_info["lasers"]:
                            laser_device_name = laser["name"]
                            laser_device_description, laser_device_manufacturer = get_laser_description_manufacturer(
                                laser, "internal"
                            )
                            internal_laser_device = Device(
                                name=laser_device_name,
                                description=laser_device_description,
                                manufacturer=laser_device_manufacturer,
                            )
                            if laser_device_name not in laser_devices:
                                laser_devices[laser_device_name] = internal_laser_device

            for laser_assembly in laser_assemblies:
                for laser in laser_assembly["lasers"]:
                    laser_device_name = laser["name"]
                    laser_device_description, laser_device_manufacturer = get_laser_description_manufacturer(
                        laser, "external"
                    )
                    external_laser_device = Device(
                        name=laser_device_name,
                        description=laser_device_description,
                        manufacturer=laser_device_manufacturer,
                    )
                    if laser_device_name not in laser_devices:
                        laser_devices[laser_device_name] = external_laser_device

            # get probes and lasers used in the session
            devices = {}
            devices_target_location = {}
            for ephys_module in ephys_modules:
                assembly_name = ephys_module["assembly_name"]

                for probe_name, probe_device in probe_devices.items():
                    if probe_name in assembly_name and probe_name not in devices:
                        devices[probe_name] = probe_device
                        devices_target_location[probe_name] = ephys_module["primary_targeted_structure"]
            if len(stimulus_device_names) > 0:
                for stimulus_device_name in stimulus_device_names:
                    if stimulus_device_name in laser_devices and stimulus_device_name not in devices:
                        devices[stimulus_device_name] = laser_devices[stimulus_device_name]
        else:
            warnings.warn(f"v{rig_schema_version} for rig schema is not currently supported")
    else:
        warnings.warn(f"Rig file not found in {session_folder}")

    return devices, devices_target_location


def get_laser_description_manufacturer(laser, type):
    laser_device_description = f"Type: {type} "
    wavelength = laser.get("wavelength", None)
    if wavelength is not None:
        laser_device_description += f" - Wavelength: {wavelength} {laser.get('wavelength_unit', 'nanometer')}"
    max_power = laser.get("maximum_power", None)
    if max_power is not None:
        laser_device_description += f" - Max power: {max_power} {laser.get('power_unit', 'milliwatt')}"
    coupling = laser.get("coupling", None)
    if coupling is not None:
        laser_device_description += f" - Coupling: {coupling}"
    laser_device_manufacturer = laser.get("manufacturer", None)
    if isinstance(laser_device_manufacturer, dict):
        laser_device_manufacturer = laser_device_manufacturer.get("name", None)
    return laser_device_description, laser_device_manufacturer


def add_waveforms_with_uneven_channels(
    sorting_analyzer: SortingAnalyzer,
    recording: BaseRecording = None,
    nwbfile: Optional[pynwb.NWBFile] = None,
    metadata: Optional[dict] = None,
    unit_ids: Optional[List[Union[str, int]]] = None,
    skip_properties: Optional[List[str]] = None,
    property_descriptions: Optional[dict] = None,
    write_as: Literal["units", "processing"] = "units",
    units_name: str = "units",
    units_description: str = "Autogenerated by neuroconv.",
):
    """
    Modified version of the neurconv.tools.spikeinterface.add_waveforms function
    to deal with multiple streams with an uneven number of channels (due to bad channel removel).
    The strategy is to add electrodes using the "full" recording and pad missing waveform channels
    with zeros.
    """
    from neuroconv.tools.spikeinterface.spikeinterface import (
        add_electrodes_info_to_nwbfile,
        add_units_table_to_nwbfile,
        _get_electrode_group_indices,
    )

    # TODO: move into add_units
    assert write_as in [
        "units",
        "processing",
    ], f"Argument write_as ({write_as}) should be one of 'units' or 'processing'!"
    if write_as == "units":
        assert units_name == "units", "When writing to the nwbfile.units table, the name of the table must be 'units'!"
    write_in_processing_module = False if write_as == "units" else True

    num_units = len(sorting_analyzer.unit_ids)
    # pad with zeros if needed
    template_ext = sorting_analyzer.get_extension("templates")
    if recording.get_num_channels() >= sorting_analyzer.get_num_channels():
        template_means_partial = template_ext.get_templates()
        template_stds_partial = template_ext.get_templates(operator="std")
        num_samples = template_means_partial.shape[1]
        template_means = np.zeros((num_units, num_samples, recording.get_num_channels()))
        template_stds = np.zeros((num_units, num_samples, recording.get_num_channels()))
        channel_mask = np.isin(recording.channel_ids, sorting_analyzer.channel_ids)
        template_means[:, :, channel_mask] = template_means_partial
        template_stds[:, :, channel_mask] = template_stds_partial
    else:
        template_means = template_ext.get_templates()
        template_stds = template_ext.get_templates(operator="std")

    sorting = sorting_analyzer.sorting
    if unit_ids is not None:
        unit_indices = sorting.ids_to_indices(unit_ids)
        template_means = template_means[unit_indices]
        template_stds = template_stds[unit_indices]

    # metrics properties (quality, template) are added as properties to the sorting copy
    sorting_copy = sorting.select_units(unit_ids=sorting.unit_ids)
    if sorting_analyzer.has_extension("quality_metrics"):
        qm = sorting_analyzer.get_extension("quality_metrics").get_data()
        for prop in qm.columns:
            if prop not in sorting_copy.get_property_keys():
                sorting_copy.set_property(prop, qm[prop])
    if sorting_analyzer.has_extension("template_metrics"):
        tm = sorting_analyzer.get_extension("template_metrics").get_data()
        for prop in tm.columns:
            if prop not in sorting_copy.get_property_keys():
                sorting_copy.set_property(prop, tm[prop])

    add_electrodes_info_to_nwbfile(recording, nwbfile=nwbfile, metadata=metadata)
    electrode_group_indices = _get_electrode_group_indices(recording, nwbfile=nwbfile)
    unit_electrode_indices = [electrode_group_indices] * num_units

    add_units_table_to_nwbfile(
        sorting=sorting_copy,
        nwbfile=nwbfile,
        unit_ids=unit_ids,
        property_descriptions=property_descriptions,
        skip_properties=skip_properties,
        write_in_processing_module=write_in_processing_module,
        units_table_name=units_name,
        unit_table_description=units_description,
        waveform_means=template_means,
        waveform_sds=template_stds,
        unit_electrode_indices=unit_electrode_indices,
    )
