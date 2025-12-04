import urllib.request
from tqdm import tqdm
import os
import yaml


def load_config(config_path):
    """
    Loads the configuration yaml file

    :param config_path: The path to the configuration file.
    :return: The configuration as a dictionary.
    """

    # Load the full configuration directly from params.yaml
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"### Successfully loaded configuration from {config_path} ###")
    except Exception as e:
        print(f"### ERROR loading configuration from {config_path}: {e} ###")
        print(f"{e}")
        config = {}

    return config

def download_weights(weight_filename, weight_url):
    """
    Downloads model weights if they do not exist locally yet

    : param weight_filename: The filename that the model weights should be saved as
    : param weight_url: The URL that the mdoel weight can be downloaded from
    """

    # Define a callback function to update the progress bar
    def reporthook(block_num, block_size, total_size):
        """
        Downloads model weights if they do not exist locally yet

        : param weight_filename: The filename that the model weights should be saved as
        : param weight_url: The URL that the mdoel weight can be downloaded from
        """


        if pbar.total is None:
            pbar.total = total_size
        pbar.update(block_size)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = os.path.join(base_dir, 'image_analyzer', 'weights')
    weights_file = os.path.join(weights_dir, weight_filename)

    # Create the directory if it doesn't exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Check and download the config file if it doesn't exist
    if not os.path.isfile(weights_file):
        print(f"Weights {weights_file} are not stored locally. Downloading...")
        # Create a request with User-Agent header to avoid 403 errors
        req = urllib.request.Request(weight_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        
        with urllib.request.urlopen(req) as response:
            # Get file size from headers if available
            total_size = int(response.headers.get('Content-Length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, miniters=1, desc=weights_file) as pbar:
                with open(weights_file, 'wb') as f:
                    while True:
                        chunk = response.read(8192)  # Read in 8KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
