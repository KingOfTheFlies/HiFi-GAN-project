import json
import os
import random
import re
import shutil
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tarfile
import urllib.request

import torchaudio
# from speechbrain.utils.data_utils import download_file
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from .base_dataset import BaseDataset
# from src.utils.mel_utils import MelSpectrogram


class LJSpeechDataset(BaseDataset):
    URL_LINK = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'

    SAMPLE_RATE = 22050
    VAL_NUM_SAMPLES = 100

    def __init__(self, dataset_path, dataset_length, gen_mel_spec, override=True, *args, **kwargs):
        # Ensure dataset is downloaded and extracted
        dataset_dir = ROOT_PATH / dataset_path
        self._ensure_dataset(dataset_dir)

        # Prepare index
        index_path = dataset_dir / "index" / "ljspeech_index.json"
        if index_path.exists() and not override:
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_dir, dataset_length)

        super().__init__(index, gen_mel_spec, *args, **kwargs)

    def _ensure_dataset(self, dataset_dir):
        """
        Ensure the dataset is downloaded and extracted.
        """
        
        if dataset_dir.exists():
            print("Dataset dir:", dataset_dir)
            return
        archive_path = dataset_dir / "LJSpeech-1.1.tar.bz2"
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download the dataset if archive doesn't exist
        if not archive_path.exists():
            print(f"Downloading LJSpeech dataset from {self.URL_LINK}...")
            self._download_file(self.URL_LINK, archive_path)

        # Extract the dataset if not already extracted
        extracted_dir = dataset_dir / "LJSpeech-1.1"
        if not extracted_dir.exists():
            print("Extracting LJSpeech dataset...")
            self._extract_archive(archive_path, dataset_dir)

    def _download_file(self, url, dest_path):
        """
        Download a file from a URL with a progress bar.
        """
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get("Content-Length", 0))
            with open(dest_path, "wb") as out_file, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                while True:
                    buffer = response.read(1024)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    pbar.update(len(buffer))

    def _extract_archive(self, archive_path, extract_to):
        """
        Extract a tar.bz2 archive.
        """
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=extract_to)

    def _create_index(self, dataset_path, dataset_length):
        index_path = ROOT_PATH / dataset_path / "index"
        audio_path = ROOT_PATH / dataset_path / "wavs"

        data_names = [path.split('.')[0] for path in os.listdir(audio_path)]
        index_path.mkdir(exist_ok=True, parents=True)
        print("Creating Index for dataset")
        index = []
        dataset_length = dataset_length if dataset_length is not None else len(data_names)
        for i in tqdm(range(dataset_length)):
            data_name = data_names[i]
            data_sample = {
                "wav_name": str(audio_path / f"{data_name}.wav"),
            }

            index.append(data_sample)
        write_json(index, str(index_path / "ljspeech_index.json"))
        return index

    # def _get_index(self, train: bool) -> List[str]:
    #     index_filepath = self._indices_dir / 'ljspeech_index.json'
    #     if not index_filepath.exists():
    #         self._build_index(index_filepath)
    #     index = json.load(open(index_filepath, 'r'))
    #     random.seed(42)
    #     random.shuffle(index)
    #     if train:
    #         return index[:-self.VAL_NUM_SAMPLES]
    #     else:
    #         return index[-self.VAL_NUM_SAMPLES:]

    # def _build_index(self, index_filepath: Path):
    #     dataset_dirpath = self._data_dir / 'LJSpeech-1.1'
    #     if not dataset_dirpath.exists():
    #         self._download_dataset()
    #     wavs_dirpath = dataset_dirpath / 'wavs'

    #     print('Building audio index...')
    #     audio_pattern = re.compile(r'^LJ\d{3}-\d{4}\.wav$')
    #     audio_filenames = sorted(
    #         filename for filename in os.listdir(wavs_dirpath)
    #         if audio_pattern.match(filename))

    #     full_index = [str(wavs_dirpath / filename) for filename in audio_filenames]
    #     index_filepath.parent.mkdir(parents=True, exist_ok=True)
    #     json.dump(full_index, open(index_filepath, 'w'))

    # def _download_audios(self):
    #     arch_filepath = self._data_dir / 'LJSpeech-1.1.tar.bz2'
    #     extracted_dataset_dirpath = self._data_dir / 'LJSpeech-1.1'
    #     self._download_archive(self.URL_LINK, arch_filepath, extracted_dataset_dirpath,
    #                            desc='audios')

    # @staticmethod
    # def _download_archive(link: str, arch_filepath: Path, extracted_dirpath: Path, desc: str):
    #     if not extracted_dirpath.exists():
    #         if not arch_filepath.exists():
    #             print(f'Downloading LJSpeech {desc}...')
    #             download_file(link, arch_filepath)
    #         print(f'Extracting LJSpeech {desc}...')
    #         shutil.unpack_archive(arch_filepath, extracted_dirpath.parent)
    #         # os.remove(arch_filepath)