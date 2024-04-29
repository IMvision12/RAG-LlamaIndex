import os
import argparse
import requests
from typing import List
from tqdm import tqdm


def download_paper(link: str, save_dir: str, index: int) -> str:
    """
    Download a single paper from a given link.

    Args:
        link (str): The URL of the paper to download.
        save_dir (str): The directory where the paper will be saved.
        index (int): The index of the paper in the list of links.

    Returns:
        str: The filename of the downloaded paper.
    """
    filename = f"{index + 1}_paper.pdf"
    filepath = os.path.join(save_dir, filename)
    try:
        response = requests.get(link, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        total_size = int(response.headers.get("content-length", 0))
        with open(filepath, "wb") as file, tqdm(
            desc=f"Paper {index + 1}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Paper {index + 1} downloaded and saved as '{filename}'.")
    except requests.RequestException as e:
        print(f"Failed to download paper {index + 1} from '{link}': {e}")
    return filename


def download_papers_from_links(links: List[str], save_dir: str = "data") -> None:
    """
    Download papers from a list of links sequentially and save them to a specified directory.

    Args:
        links (List[str]): List of URLs pointing to the papers.
        save_dir (str, optional): Directory to save the downloaded papers. Defaults to "data".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, link in enumerate(links):
        download_paper(link, save_dir, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download papers from given URLs.")
    parser.add_argument(
        "--links",
        metavar="URL",
        type=str,
        nargs="+",
        help="URLs of the papers to download",
    )
    parser.add_argument(
        "--dir",
        dest="save_dir",
        type=str,
        default="data",
        help="Directory to save the downloaded papers",
    )
    args = parser.parse_args()

    download_papers_from_links(args.links, args.save_dir)
