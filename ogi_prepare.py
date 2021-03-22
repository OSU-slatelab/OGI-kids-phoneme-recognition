"""
Data preparation for OGI kids' speech corpus.

Authors
* Peter Plantinga 2020
"""

import os
import re
import json
import logging
from glob import glob
from g2p_en import G2p
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

G2P = G2p()
NOT_PHONEMES = [" ", ".", "!", "?"]
logger = logging.getLogger(__name__)
SAMPLERATE = 16000
GRADE_LIST = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def prepare_ogi(
    data_folder, train_file, valid_file, test_file, valid_frac=0.03, test_frac=0.03,
):
    """
    Prepares the json files for the CSLU dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original CSLU dataset is stored.
    train_file : str
        Path for storing the train data manifest file.
    valid_file : str
        Path for storing the validation data manifest file.
    test_file : str
        Path for storing the test data manifest file.
    valid_frac : float
        Approximate fraction of speakers to assign to valid data in each grade.
    test_frac : float
        Approximate fraction of speakers to assign to test data in each grade.
    """
    # Setting file extension.
    extension = [".wav"]

    # Check if this phase is already done (if so, skip it)
    if skip(train_file, valid_file, test_file):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    logger.debug("Creating json files for the OGI Dataset...")

    speech_dir = os.path.join(data_folder, "speech", "scripted")

    # Creating json file for training data
    wav_lst_train = []
    wav_lst_valid = []
    wav_lst_test = []
    for grade in GRADE_LIST:
        grade_dir = os.path.join(speech_dir, grade)

        # Collect list of folders for distinct speakers
        spk_list = sorted(glob(os.path.join(grade_dir, "*", "ks*")))

        # Divide into sublists
        valid_len = int(valid_frac * len(spk_list))
        test_len = int(test_frac * len(spk_list))
        valid_spk_list = spk_list[:valid_len]
        test_spk_list = spk_list[valid_len : valid_len + test_len]
        train_spk_list = spk_list[valid_len + test_len :]

        # Find files and add em
        for folder in train_spk_list:
            wav_lst_train.extend(get_all_files(folder, match_and=extension))
        for folder in valid_spk_list:
            wav_lst_valid.extend(get_all_files(folder, match_and=extension))
        for folder in test_spk_list:
            wav_lst_test.extend(get_all_files(folder, match_and=extension))

    # Create data maps
    id2word = load_id2word_map(data_folder)
    id2verify = load_id2verify_map(data_folder)

    # Create json with all files
    create_json(wav_lst_train, train_file, id2word, id2verify)
    create_json(wav_lst_valid, valid_file, id2word, id2verify)
    create_json(wav_lst_test, test_file, id2word, id2verify)


def skip(*filenames):
    """
    Detects if the data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    for json in filenames:
        if not os.path.isfile(json):
            return False

    return True


def load_id2word_map(data_folder):
    """Load map from word id to characters"""
    word_file = os.path.join(data_folder, "docs", "all.map")
    id2word = {}
    for line in open(word_file):
        line = line.split()
        if len(line) == 0:
            continue
        word_id = line[0]

        # Normalize and store
        words = " ".join(line[1:]).lower()
        words = re.sub(r"[^a-z ]", "", words)
        id2word[word_id] = words

    return id2word


def load_id2verify_map(data_folder):
    id2verify = {}
    for grade in GRADE_LIST:
        verify_file = os.path.join(data_folder, "docs", grade + "-verified.txt")
        for line in open(verify_file):
            filename, verify = line.split()
            utterance_id = filename[-12:-4]
            id2verify[utterance_id] = int(verify)

    return id2verify


def words2phonemes(word):
    "Convert all words to phonemes"
    return ".".join(p.strip("012") for p in G2P(word) if p not in NOT_PHONEMES)


def create_json(wav_lst, json_file, id2word, id2verify, alignments=None):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
        The path of the output json file
    id2word : dict
        A mapping from id to word
    """

    logger.debug(f"Creating json manifest list: {json_file}")
    json_dict = {}

    # Processing all the wav files in the list
    for wav_file in wav_lst:

        path = os.path.normpath(wav_file).split(os.path.sep)
        relative_filename = os.path.join("{root}", *path[-6:])

        # Compute phoneme targets
        snt_id = wav_file[-12:-4]
        word_id = wav_file[-7:-5].upper()
        words = id2word[word_id]
        phonemes = " ".join(words2phonemes(word) for word in words.split())

        # Don't use bad quality recordings
        if snt_id not in id2verify or id2verify[snt_id] == 3:
            continue

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        json_dict[snt_id] = {
            "wav": relative_filename,
            "length": duration,
            "words": words,
            "phonemes": phonemes,
        }

        if alignments is not None:
            json_dict[snt_id].update(alignments[snt_id])

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.debug(f"{json_file} successfully created!")


def prepare_aligned_ogi(
    data_folder, train_file, valid_file, train_align_file, valid_align_file,
):
    """
    Prepares the json files for the CSLU dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original CSLU dataset is stored.
    train_file : str
        Path for storing the train data manifest file.
    valid_file : str
        Path for storing the validation data manifest file.
    train_align_file : str
        Path for file containing the alignments for training data.
    valid_align_file : str
        Path for file containing the alignments for validation data.
    """
    # Setting file extension.
    extension = [".wav"]

    # Check if this phase is already done (if so, skip it)
    if skip(train_file, valid_file):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    logger.debug("Creating json files for the OGI Dataset...")

    # Load train and validation alignments
    with open(train_align_file) as f:
        train_alignments = json.load(f)
    with open(valid_align_file) as f:
        valid_alignments = json.load(f)

    speech_dir = os.path.join(data_folder, "speech", "scripted")

    wav_lst_train = []
    wav_lst_valid = []
    full_wav_lst = get_all_files(speech_dir, match_and=extension)
    for wav_file in full_wav_lst:
        snt_id = wav_file[-12:-4]
        if snt_id in train_alignments:
            wav_lst_train.append(wav_file)
        elif snt_id in valid_alignments:
            wav_lst_valid.append(wav_file)


    # Create data maps
    id2word = load_id2word_map(data_folder)
    id2verify = load_id2verify_map(data_folder)

    # Create json with all files
    create_json(wav_lst_train, train_file, id2word, id2verify, train_alignments)
    create_json(wav_lst_valid, valid_file, id2word, id2verify, valid_alignments)
