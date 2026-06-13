import argparse
import locale
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import pydicom
import tomlkit
from PIL import Image
from minio import Minio
from tqdm import tqdm

from kernel import fast_drr

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


def main(cfg_path: str, done: str, nii_name: str):
    done = Path(done) / nii_name
    if done.exists() and done.is_file():
        return

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    with tempfile.TemporaryDirectory() as tdir:
        f = Path(tdir) / 'image.nii.gz'

        client.fget_object('nii', nii_name, f.as_posix())

        dcm = nii_name.removesuffix('.nii.gz') + '.dcm'
        dcm = client.get_object('dcm', dcm).data
        dcm = pydicom.dcmread(BytesIO(dcm))

        image = itk.imread(f)

    info = itk.dict_from_image(image)
    info.update(info['imageType'])
    del info['name'], info['bufferedRegion'], info['data'], info['imageType']

    image = itk.array_from_image(image)
    info['range'] = np.array([np.min(image), np.max(image)]).tolist()
    info['origin'] = np.array(info['origin']).tolist()
    info['spacing'] = np.array(info['spacing']).tolist()
    info['size'] = np.array(info['size']).tolist()
    info['direction'] = np.array(info['direction']).tolist()
    info['dicom'] = {
        'ImageType': str(dcm.get('ImageType')),
        'PatientID': str(dcm.get('PatientID')),
        'PatientName': str(dcm.get('PatientName')),
        'StudyDate': str(dcm.get('StudyDate')),
        'StudyTime': str(dcm.get('StudyTime')),
        'PatientPosition': str(dcm.get('PatientPosition')),
    }

    with tempfile.TemporaryDirectory() as tdir:
        name = nii_name + '/info.toml'
        file = Path(tdir) / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(tomlkit.dumps(info), 'utf-8')
        client.fput_object('drr', name, file.as_posix())

        l = np.array(info['spacing']) * np.array(info['size'])
        l = tuple(max(round(_) * 2, 1) for _ in l)

        try:
            for _ in range(2):
                x = fast_drr(image, _)
                x = Image.fromarray(x).resize([(l[0], l[1]), (l[0], l[2])][_])

                name = nii_name + '/' + ['axial', 'coronal'][_] + '.png'
                file = Path(tdir) / name
                file.parent.mkdir(parents=True, exist_ok=True)
                x.save(file.as_posix())
                client.fput_object('drr', name, file.as_posix())
            no_drr = False
        except (ValueError, Exception):
            no_drr = True

        if any((no_drr, info['dimension'] != 3, info['componentType'] in ('uint8',), info['components'] != 1)):
            client.put_object('drr', nii_name + '/invalid', BytesIO(b''), 0)

    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()


def get_nii(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    for _ in client.list_objects('nii', recursive=True):
        if _.is_dir:
            continue

        yield _.object_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--done', required=True)
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(main, args.config, args.done, _): _ for _ in get_nii(args.config)}

        try:
            for fu in (bar := tqdm(as_completed(futures), total=len(futures))):
                try:
                    fu.result()
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
