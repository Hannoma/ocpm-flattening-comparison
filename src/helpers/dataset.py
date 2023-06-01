import os


def log_path_for_dataset(dataset: dict, override_format: str = None) -> str:
    from definitions import ROOT_DIR

    return os.path.join(ROOT_DIR, 'data', 'processed',
                        dataset['filename'] + '.' +
                        (override_format if override_format is not None else dataset['format']))


def append_suffix_to_filename(filename: str, suffix: str) -> str:
    parts = filename.split('.')
    return '.'.join(parts[:-1]) + suffix + '.' + parts[-1]


def get_selected_dataset() -> dict:
    from dotenv import load_dotenv, find_dotenv
    from definitions import DATASETS

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    return DATASETS[os.environ["DATASET"]]
