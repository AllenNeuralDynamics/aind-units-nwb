import json
import warnings
from pathlib import Path
from typing import Union, List, Optional, Literal
from packaging.version import parse
import numpy as np

from spikeinterface import BaseRecording, SortingAnalyzer

import pynwb
from pynwb.file import Device


def get_devices_from_metadata(session_folder: str, segment_index: int=0):
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
        add_electrodes_info,
        add_units_table,
        get_electrode_group_indices
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

    add_electrodes_info(recording, nwbfile=nwbfile, metadata=metadata)
    electrode_group_indices = get_electrode_group_indices(recording, nwbfile=nwbfile)
    unit_electrode_indices = [electrode_group_indices] * num_units

    add_units_table(
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
