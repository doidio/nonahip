from pathlib import Path

import tomlkit
from tqdm import tqdm


def load_pairs(cfg: dict, categories: list[str]):
    root = Path(cfg['dataset']['root'])
    pair = root / 'pair'
    excluded = cfg['pairs']['excluded']

    pairs = {}
    for pid in tqdm(list(pair.iterdir())):
        for rl in 'RL':
            prl = pid / rl

            if not (prl / 'pair.toml').exists():
                continue

            it = {}

            for category in set(['pair'] + categories):
                f = prl / f'{category}.toml'
                if f.exists():
                    data = tomlkit.loads(f.read_text('utf-8')).unwrap()
                    it[category] = data
                else:
                    it[category] = {}

            prl = '_'.join([pid.name, rl])
            if prl in excluded:
                it['excluded'] = True

            it['prl'] = prl
            pairs[prl] = it

    return cfg, pairs


if __name__ == '__main__':
    print(tomlkit.dumps(list(load_pairs('config.toml', ['roi', 'context', 'align'])[1].values())[0]))
