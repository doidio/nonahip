import argparse
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import tomlkit
from minio import Minio, S3Error
from tqdm import tqdm


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    object_names = []
    for _ in client.list_objects('pair', recursive=True):
        if not _.object_name.endswith('.nii.gz'):
            continue

        pid, rl, op, nii = _.object_name.split('/')

        if op not in ('pre', 'post'):
            continue

        try:
            client.stat_object('nii', _ := f'{pid}/{nii}')
        except S3Error:
            continue

        object_names.append(_)

    for object_name in tqdm(object_names):
        for bucket in ('total', 'appendicular-bones'):
            try:
                if client.stat_object(bucket, object_name).size > 0:
                    continue
            except S3Error:
                pass

            print(object_name, bucket)

            with tempfile.TemporaryDirectory() as tdir:
                image = Path(tdir) / 'image.nii.gz'
                label = Path(tdir) / 'label.nii.gz'

                client.fget_object('nii', object_name, image.as_posix())

                with ProcessPoolExecutor(max_workers=1) as executor:
                    task = bucket.replace('-', '_')
                    future = executor.submit(seg, image.as_posix(), label.as_posix(), task)
                    try:
                        future.result()
                    except Exception as e:
                        traceback.print_exception(e)
                        warnings.warn(str(e), stacklevel=3)

                if label.exists():
                    print(object_name, bucket, 'done')
                    client.fput_object(bucket, object_name, label.as_posix())
                else:
                    warnings.warn(' '.join([object_name, bucket, 'error']))


def seg(image_path: str, label_path: str, task: str):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(image_path, label_path, True, task=task, quiet=True)


# 数据超过 (512,512,1000) Windows 平台报错 OSError: [WinError 87]
# 修改源码 https://github.com/wasserth/TotalSegmentator/issues/533
# def predict_from_data_iterator(self,
#                                data_iterator,
#                                save_probabilities: bool = False,
#                                num_processes_segmentation_export: int = default_num_processes):
#     """
#     each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
#     If 'ofile' is None, the result will be returned instead of written to a file
#     """
#     r = []
#     for preprocessed in data_iterator:
#         data = preprocessed['data']
#         if isinstance(data, str):
#             delfile = data
#             data = torch.from_numpy(np.load(data))
#             os.remove(delfile)
#
#         ofile = preprocessed['ofile']
#
#         properties = preprocessed['data_properties']
#
#         prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
#
#         if ofile is not None:
#             result = export_prediction_from_logits(
#                 prediction, properties, self.configuration_manager, self.plans_manager,
#                 self.dataset_json, ofile, save_probabilities
#             )
#         else:
#             result = convert_predicted_logits_to_segmentation_with_correct_shape(
#                 prediction, self.plans_manager, self.configuration_manager, self.label_manager,
#                 properties, save_probabilities
#             )
#
#         r.append(result)
#
#     ret = [out for out in r]
#
#     if isinstance(data_iterator, MultiThreadedAugmenter):
#         data_iterator._finish()
#
#     # clear lru cache
#     compute_gaussian.cache_clear()
#     # clear device cache
#     empty_cache(self.device)
#     return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    try:
        while True:
            main(args.config)

            for _ in range(60, 0, -1):
                print(f'\rRetry in {_}s...', end='', flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
