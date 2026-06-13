# https://github.com/minio/minio?tab=readme-ov-file#install-from-source
# minio server /Volumes/3726/minio/sh9_tha
# mc alias set local http://localhost:9000 minioadmin minioadmin
# mc mb local/tha --ignore-existing

# compression (optional)
# mc admin config set local compression enable=true extensions=".dcm" mime_types="application/dicom"
# mc admin service restart local

import argparse
import shutil
import tempfile
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itk
import numpy as np
import pydicom
import tomlkit
from minio import Minio
from tqdm import tqdm

from kernel import itk_monkey_patch

itk_monkey_patch()

class FatalError(Exception):
    pass


def main(zip_file: str, cfg_path: str):
    zip_file = Path(zip_file)
    done = zip_file.parent / f'{zip_file.name}.done'
    if done.exists():
        return

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    client = Minio(**cfg['minio']['client'])

    tmpdir: str | None = cfg.get('local', {}).get('tmpdir')
    Path(tmpdir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmpdir) as tdir:
        tdir = Path(tdir)

        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                with zf.open(member) as stream:
                    from pydicom.errors import InvalidDicomError
                    try:
                        it = pydicom.dcmread(stream)
                    except (InvalidDicomError, ValueError):
                        continue

                # 校验模态
                if it.get('Modality') != 'CT':
                    continue

                f = tdir / it.StudyInstanceUID / it.SeriesInstanceUID / f'{it.SOPInstanceUID}.dcm'
                f.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src, f.open('wb') as dst:
                    shutil.copyfileobj(src, dst)

        for study in tdir.iterdir():
            if not study.is_dir():
                continue

            for series in study.iterdir():
                if not series.is_dir():
                    continue

                # 校验层数
                slices = list(series.iterdir())
                if not len(slices) > 9:
                    continue

                # 读取图像体积
                try:
                    image = itk.imread(series)
                except (RuntimeError, Exception):
                    continue

                # 强行抹除原始 Direction，将体素坐标系升格为全局笛卡尔坐标系
                direction = itk.matrix_from_array(np.identity(3, dtype=np.float64))
                image.SetDirection(direction)

                # 校验分辨率和层厚
                spacing = np.array(itk.spacing(image))
                if not np.all((0 < np.array(spacing)) & (np.array(spacing) < 3.5)):
                    continue

                # 写入临时文件
                f = tdir / f'{series.name}.nii.gz'
                itk.imwrite(image, f, compression=True)

                # 上传归档
                try:
                    it = pydicom.dcmread(slices[0])

                    _ = f'{it.PatientID}/{it.SeriesInstanceUID}.nii.gz'
                    client.fput_object('nii', _, f.as_posix())

                    _ = f'{it.PatientID}/{it.SeriesInstanceUID}.dcm'
                    client.fput_object('dcm', _, slices[0].as_posix())
                except Exception as _:
                    raise FatalError(str(_)) from None

    done.touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).absolute()
    zip_files = [_ for _ in zip_dir.rglob('*.zip')]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(
            main, _.as_posix(), args.config,
        ): (_.as_posix(), args.config,) for _ in zip_files}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except FatalError as _:
                    raise _
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}')

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
