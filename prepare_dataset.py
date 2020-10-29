from pathlib import Path
from glob import glob
from zipfile import ZipFile
from multiprocessing import Pool
import os
import time
import csv
from utils.parameters import *
from utils.download_images import download


def move_HS():
    if not Path(extracted_Dir).exists():
        print('The extracted Dataset directory does not exists!')
        print('Please download and extract the Herbarium Senckenbergianum '
              '(FR) dataset in the dataset drectory.')

    else:
        Path(HRBFR_annoDir).mkdir(parents=True, exist_ok=True)
        Path(HRBFR_imgDir).mkdir(parents=True, exist_ok=True)

        print('Moving Herbarium Senckenbergianum (FR) scans and annotations...',
              flush=True)

        if not Path(HRBFR_detectionsDir).exists():
            Path(extracted_detections).rename(HRBFR_detectionsDir)

        anno_paths = [f for f in glob(extracted_hrbDir + '*.xml')]
        img_paths = [f for f in glob(extracted_hrbDir + '*.jpg')]

        fr_ids = []

        for anno_path, img_path in zip(anno_paths, img_paths):

            fr_id = os.path.splitext(os.path.basename(anno_path))[0]
            fr_ids.append(fr_id)

            Path(anno_path).rename(HRBFR_annoDir + fr_id + '.xml')
            Path(img_path).rename(HRBFR_imgDir + fr_id + '.jpg')

        if not Path(HRBFR_Dir + 'list.txt').exists():
            with open(HRBFR_Dir + 'list.txt', 'w') as f:
                for t in fr_ids:
                    f.write(t + '\n')


def extract_annotations():
    print('Extracting MNHN Paris Herbarium annotations...', flush=True)
    with ZipFile(HRBParis_Dir + 'annotations.zip', 'r') as zip_ref:
        zip_ref.extractall(HRBParis_Dir)


def download_imgs():
    Path(HRBParis_imgDir).mkdir(parents=True, exist_ok=True)
    print('Downloading MNHN Paris Herbarium scans...', flush=True)
    with open(HRBParis_urls, 'r', encoding="utf8", errors='ignore')as file:
        reader = csv.reader(file)
        header = next(reader)
        urls = {rows[0]: rows[1] for rows in reader}

    for i in range(5):
        paths = [f for f in glob(HRBParis_imgDir + '*.jpg')]

        if len(paths) == TOTAL_HRBParis_urls:
            break

        pool = Pool(4)
        pool.map(download, urls.items())
        pool.close()
        pool.join()

        time.sleep(20)

    print()

    download_ids = []
    for i in range(len(paths)):
        download_ids.append(os.path.splitext(os.path.basename(paths[i]))[0])

    not_downloaded = [
        gbif_id for gbif_id in urls if gbif_id not in download_ids]

    if not_downloaded:
        print('The following images could not be downloaded, please try later')
        for nd in not_downloaded:
            print(nd + '.jpg')


def copy_test_imgs():
    Path(HRBParis_testDir).mkdir(parents=True, exist_ok=True)
    print('Copying MNHN Paris Herbarium test scans...', flush=True)

    with open(HRBParis_Dir + 'test.txt', 'r') as f:
        test_ids = f.read().splitlines()

        paths = [f for f in glob(HRBParis_imgDir + '*.jpg')]

        for path in paths:
            gbif_id = os.path.splitext(os.path.basename(path))[0]

            if gbif_id in test_ids:
                dest = Path(HRBParis_testDir + gbif_id + '.jpg')
                dest.write_bytes(Path(path).read_bytes())


if __name__ == '__main__':

    move_HS()
    extract_annotations()
    download_imgs()
    copy_test_imgs()
