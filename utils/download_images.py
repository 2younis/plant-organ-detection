from PIL import Image
from urllib import request
from socket import timeout
from glob import glob
import os
from utils.parameters import *


def download(id_url):
    gbif_id = id_url[0]
    url = id_url[1]

    paths = [f for f in glob(HRBParis_imgDir + '*.jpg')]

    download_ids = []
    for i in range(len(paths)):
        download_ids.append(os.path.splitext(os.path.basename(paths[i]))[0])

    if gbif_id not in download_ids:

        path = HRBParis_imgDir + gbif_id + '.jpg'

        try:
            response = request.urlopen(url, timeout=20)

            with open(path, 'wb') as img:
                img.write(response.read())
            img = Image.open(path)
            img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            img.save(path)

            progress = ((len(paths) + 1) / TOTAL_HRBParis_urls) * 100
            print(' Progress:', f'{progress:.1f}%', end="\r", flush=True)

        except OSError:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        except (timeout, ValueError):
            pass
