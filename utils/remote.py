import zipfile
import requests
from tqdm import tqdm

from utils.fs import AnyFile, File


def download_remote_file(url: str, out_path: AnyFile, progress_bar: bool = True):

    out_path = File(out_path)

    with requests.get(url, stream=progress_bar) as req:
        with open(out_path, 'wb') as f:
            if progress_bar:

                chunk_size, unit = 1024, 'KB'

                total_length = int(req.headers.get('content-length'))
                n_chunks, remainder = divmod(total_length, chunk_size)

                for chunk in tqdm(req.iter_content(chunk_size), total=n_chunks, unit=unit):
                    if chunk:
                        f.write(chunk)

            else:
                f.write(req.content)

        f.flush()

    return out_path
