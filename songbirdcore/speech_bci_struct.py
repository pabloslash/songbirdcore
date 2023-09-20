import os
import socket

locations_dict = dict()

locations_dict['jupyter-pablotostado--phy-2ddesktop'] = {'data': os.path.abspath('/net2/expData/speech_bci/')}


def get_experiment_struct(bird, date, sess, ephys_software='sglx', sort='', location_dict: dict = dict()):
    """
    Create dictionary of important directories and files of interest.

    Arguments:
        location {dict} -- data_dir
        sess_par {dict} -- session parameters dictionary. Example:
            sess_par = {'bird': 'z_w12m7_20',
                        'date': '2020-11-04',
                        'ephys_software': 'sglx',
                        'sess': '2500r250a_3500_dir_g0',
                        'probe': 'probe_0',
                        'sort': 'ksort3_pt',
                        }
    """
    # Get the configuration of the experiment (data_dir). If no location dict is entered, try to get it from the hostname (from locations_dict).
    if location_dict:
        pass
    else:
        location_dict = get_location_dict()

    # make the exp struct dict.
    sess_par_dict = {'bird': bird,
                     'date': date,
                     'sess': sess,
                     'sort': sort,
                     'ephys_software': ephys_software}
    exp_struct = get_file_structure(location_dict, sess_par_dict)

    return exp_struct


def get_location_dict() -> dict:
    hostname = socket.gethostname()
    return locations_dict[hostname]


def get_file_structure(location: dict, sess_par: dict) -> dict:
    """
    Arguments:
        location {dict} -- data_dir
        sess_par {dict} -- session parameters dictionary. Example:
            sess_par = {'bird': 'z_w12m7_20',
                        'date': '2020-11-04',
                        'ephys_software': 'sglx',
                        'sess': '2500r250a_3500_dir_g0',
                        'probe': 'probe_0',
                        'sort': 'ksort3_pt',
                        }
            - bird, date and sess are self-explanatory and refer to the folder containing the raw files.
            - probe describes the probe that was used to do the sorting, which in turn determines
              neural port (via the rig.json file) and probe mapping (via rig.json file and
              pipefinch.pipeline.filestructure.probes)
            - sort determines the version of sorting, in case multiple sorts were made on the same
              session (e.g different time spans, different configurations of the sorting algorithm)
              if the field is not present or is None: ?

    Returns:
        dict -- ditcionary containing paths of folders and files of interest within data_dir.
    """

    try:
        ephys_folder = sess_par['ephys_software']
    except KeyError:
        logger.info('ephys folder defaults to sglx')
        ephys_folder = 'sglx'

    exp_struct = {}
    bird, date, sess = sess_par['bird'], sess_par['date'], sess_par['sess']

    exp_struct['folders'] = {}
    exp_struct['files'] = {}

    # The RAW DATA folders and files of interest (meta, binary etc.):
    exp_struct['folders']['bird'] = os.path.join(location['data'], 'raw_data', bird)
    exp_struct['folders']['raw'] = os.path.join(location['data'], 'raw_data', bird, date)
    exp_struct['folders'][ephys_folder] = os.path.join(exp_struct['folders']['raw'], ephys_folder)
    for f, n in zip(['rig'], ['rig.json']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders'][ephys_folder], n)

    # The DERIVED DATA folders and files of interest (wav, sort, etc.)
    exp_struct['folders']['derived'] = os.path.join(location['data'], 'derived_data', bird, date, ephys_folder, sess)
    for f, n in zip(['wav_mic'], ['wav_mic.wav']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['derived'], n)

    # KILOSORT FILE STRUCTURE
    try:
        sort_version = str(sess_par['sort'])
        if sort_version is None:
            sort_version = 'sort'
    except KeyError:
        sort_version = 'sort'

    exp_struct['folders']['sort'] = os.path.join(
        exp_struct['folders']['derived'], sort_version)

    for f, n in zip(['sort_params'], ['params.py']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['sort'], n)

    return exp_struct