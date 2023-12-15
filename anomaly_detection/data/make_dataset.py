import os
import click
from zipfile import ZipFile, BadZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_competition_data(competition_name: str, save_path: str = "./") -> None:
    """
    Download data for a Kaggle competition using the Kaggle API if it does not already exist.

    Args:
        competition_name (str): The name of the Kaggle competition.
        save_path (str): The directory where the data will be downloaded. Defaults to the current directory.
    """
    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()  # Make sure your Kaggle API credentials are configured

        # Download data for the competition
        api.competition_download_files(competition_name, path=save_path, quiet=False)

        print(f"Data for '{competition_name}' downloaded successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def unzip_file(zip_path: str, extract_path: str):
    """
    Unzip the file at 'zip_path' to the 'extract_path'.
    """
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Unzipped {zip_path} to {extract_path}")
    except BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred while unzipping {zip_path}: {str(e)}")

@click.command()
@click.argument("competition_name", type=click.Path())
@click.argument("raw_path", type=click.Path())

def main(competition_name: str, raw_path: str) -> None:
    """
    Runs data processing scripts to download the data from source, extract the data from zip files, and process the data, so it is ready for analysis.
    """
    # Execute the `download_kaggle_competition_data()` function
    download_kaggle_competition_data(competition_name, raw_path)

    # Define the path to the .zip file
    zip_path = os.path.join(raw_path, competition_name + ".zip")

    # Unzip the downloaded file and extract its contents
    unzip_file(zip_path, raw_path)

if __name__ == "__main__":
    main()




    