import argparse
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import tomlkit
from tqdm import tqdm

"""
b0_prothesis.py - 髋关节假体库 definition 与自动提取工具
本模块集成了主流全髋人工关节（THA）产品的规格定义，并提供了从原始手术记录（Excel）中自动提取型号、规格、偏距等参数的工具函数。
"""

# 股骨柄型号规格
FEMORAL = {
    '': [''],
    'DePuy Corail': [
        '',
        '8 (标准无领)', '9 (标准无领)', '10 (标准无领)', '11 (标准无领)', '12 (标准无领)', '13 (标准无领)',
        '14 (标准无领)', '15 (标准无领)', '16 (标准无领)', '18 (标准无领)', '20 (标准无领)',
        '8 (标准带领)', '9 (标准带领)', '10 (标准带领)', '11 (标准带领)', '12 (标准带领)', '13 (标准带领)',
        '14 (标准带领)', '15 (标准带领)', '16 (标准带领)', '18 (标准带领)', '20 (标准带领)',
        '9 (高偏心无领)', '10 (高偏心无领)', '11 (高偏心无领)', '12 (高偏心无领)', '13 (高偏心无领)', '14 (高偏心无领)',
        '15 (高偏心无领)', '16 (高偏心无领)',
        '9 (内翻带领)', '10 (内翻带领)', '11 (内翻带领)', '12 (内翻带领)', '13 (内翻带领)', '14 (内翻带领)',
        '15 (内翻带领)', '16 (内翻带领)', '18 (内翻带领)', '20 (内翻带领)',
        '6 (DDH)',
        '10 (翻修标准)', '11 (翻修标准)', '12 (翻修标准)', '13 (翻修标准)', '14 (翻修标准)', '15 (翻修标准)',
        '16 (翻修标准)', '18 (翻修标准)', '20 (翻修标准)',
        '10 (翻修高偏心)', '11 (翻修高偏心)', '12 (翻修高偏心)', '13 (翻修高偏心)', '14 (翻修高偏心)',
        '15 (翻修高偏心)', '16 (翻修高偏心)', '18 (翻修高偏心)', '20 (翻修高偏心)'
    ],
    'DePuy Tri-Lock': ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    'DePuy SUMMIT': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'DePuy S-ROM': ['', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
    'Stryker Accolade TMZF': [
        '',
        '0 (127°)', '1 (127°)', '2 (127°)', '2.5 (127°)', '3 (127°)', '3.5 (127°)', '4 (127°)', '4.5 (127°)',
        '5 (127°)', '5.5 (127°)', '6 (127°)', '7 (127°)', '8 (127°)',
        '0 (132°)', '1 (132°)', '2 (132°)', '2.5 (132°)', '3 (132°)', '3.5 (132°)', '4 (132°)', '4.5 (132°)',
        '5 (132°)', '5.5 (132°)', '6 (132°)', '7 (132°)', '8 (132°)'
    ],
    'Stryker Accolade II': [
        '',
        '0 (127°)', '1 (127°)', '2 (127°)', '3 (127°)', '4 (127°)', '5 (127°)', '6 (127°)', '7 (127°)', '8 (127°)',
        '9 (127°)', '10 (127°)', '11 (127°)',
        '0 (132°)', '1 (132°)', '2 (132°)', '3 (132°)', '4 (132°)', '5 (132°)', '6 (132°)', '7 (132°)', '8 (132°)',
        '9 (132°)', '10 (132°)', '11 (132°)'
    ],
    'Stryker Secur-Fit': [
        '',
        '6 (127°)', '7 (127°)', '8 (127°)', '9 (127°)', '10 (127°)', '11 (127°)', '12 (127°)', '13 (127°)',
        '4 (132°)', '5 (132°)', '6 (132°)', '7 (132°)', '8 (132°)', '9 (132°)', '10 (132°)', '11 (132°)', '12 (132°)',
        '13 (132°)', '14 (132°)'
    ],
    'Wright Profemur': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'Smith & Nephew Synergy': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
    'Smith & Nephew Anthology': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    'Smith & Nephew Plus-TS': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Zimmer M/L Taper': ['', '4', '5', '6', '7.5', '9', '10', '11', '12.5', '13.5', '15', '16.25', '17.5', '20', '22.5',
                         'ML'],
    'Zimmer CLS Spotorno': ['', '5', '6', '7', '8', '9', '10', '11.25', '12.5', '13.75', '15', '16.25'],
    'AK Medical ML-TP': ['', '1', '2', '2.5', '3', '3.5', '4', '5', '6'],
    'Waldemar Link LCU': ['', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                          '22', '23', '24', '25', '26', '27', '28'],
    'Keyi Bangen SQKA': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Zimmer CPT': ['', '0', '1', '2', '3', '4', '5', 'Long Size 2', 'Long Size 3', 'Long Size 4'],
    'Zimmer Wagner SL': ['', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                         '29', '30'],
    'Wagner Cone': [
        '',
        '13 (125°)', '14 (125°)', '15 (125°)', '16 (125°)', '17 (125°)', '18 (125°)', '19 (125°)', '20 (125°)',
        '21 (125°)', '22 (125°)',
        '13 (135°)', '14 (135°)', '15 (135°)', '16 (135°)', '17 (135°)', '18 (135°)', '19 (135°)', '20 (135°)',
        '21 (135°)', '22 (135°)'
    ],
}

HEAD_OFFSET = ['', -5.0, -4.0, -3.5, -3.0, -2.7, -2.5, -2.0, 0.0, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 7.5, 8.0, 8.5,
               9.0, 10.0, 10.5, 11.5, 12.0, 16.0]


def parse_stem(name, spec):
    """从假体名称和规格描述中识别具体型号和带角度的尺寸"""
    text = (str(name) + ' ' + str(spec)).upper()
    model, size = '', ''
    # 1. 优先根据货号前缀识别型号
    if '6020-' in text or '6021-' in text:
        model = 'Stryker Accolade TMZF'
    elif '6720-' in text or '6721-' in text:
        model = 'Stryker Accolade II'
    elif '6051-' in text or '6052-' in text:
        model = 'Stryker Secur-Fit'
    elif '3L92' in text or '925' in text or '924' in text:
        model = 'DePuy Corail'
    elif '7711-' in text:
        model = 'Zimmer M/L Taper'
    elif '01.00561' in text:
        model = 'Wagner Cone'
    elif '01.0010' in text:
        model = 'Zimmer Wagner SL'
    elif '1570-' in text or '2570-' in text:
        model = 'DePuy SUMMIT'
    elif '5232' in text:
        model = 'DePuy S-ROM'
    elif 'PLUS-TS' in text or 'A2308' in text:
        model = 'Smith & Nephew Plus-TS'
    elif '1012-' in text:
        model = 'DePuy Tri-Lock'
    elif '1100-28' in text:
        model = 'AK Medical ML-TP'
    elif '165-0' in text:
        model = 'Waldemar Link LCU'

    # 2. 识别大类型号（文本关键字）
    if not model:
        if 'CORAIL' in text or '无领双锥度' in text:
            model = 'DePuy Corail'
        elif 'TRI-LOCK' in text or 'TRILOCK' in text:
            model = 'DePuy Tri-Lock'
        elif 'SUMMIT' in text:
            model = 'DePuy SUMMIT'
        elif 'ACCOLADE' in text:
            model = 'Stryker Accolade TMZF' if 'II' not in text else 'Stryker Accolade II'
        elif 'SECUR-FIT' in text or 'SECURFIT' in text or 'SUCUR-FIT' in text:
            model = 'Stryker Secur-Fit'
        elif 'S-ROM' in text or 'SROM' in text:
            model = 'DePuy S-ROM'
        elif 'PROFEMUR' in text or 'PHA00' in text or 'PRTL' in text:
            model = 'Wright Profemur'
        elif 'SYNERGY' in text or re.search(r'7130-?\d{4}', text):
            model = 'Smith & Nephew Synergy'
        elif 'ANTHOLOGY' in text or 'SZ5' in text or 'SZ' in text:
            model = 'Smith & Nephew Anthology'
        elif 'TAPERED' in text or '股骨柄ML' in text or 'ML TAPER' in text:
            model = 'Zimmer M/L Taper'
        elif 'SQKA' in text or 'BDS/C-1' in text:
            model = 'Keyi Bangen SQKA'
        elif 'WAGNER SL' in text or ('WAGNERSL' in text.replace(' ', '')):
            model = 'Zimmer Wagner SL'
        elif 'WAGNER CONE' in text:
            model = 'Wagner Cone'
        elif 'AK-ML' in text or 'AK MEDICAL' in text:
            model = 'AK Medical ML-TP'
        elif 'LINK' in text or 'LCU' in text:
            model = 'Waldemar Link LCU'

    # 识别结果修正：如果只识别出型号但没有识别出规格，且型号识别依据较弱时，应置空
    # 特别是针对 generic 的 "股骨柄" 名称，不要默认归类为某个品牌

    if model:
        sizes = [s for s in FEMORAL[model] if s != '']
        sizes.sort(key=lambda x: len(str(x)), reverse=True)

        if model == 'Stryker Accolade TMZF':
            if m := re.search(r'602([01])-0?([0-9]{1,2})[0-9]{2}', text):
                angle = '132°' if m.group(1) == '0' else '127°'
                val = int(m.group(2))
                sz = f'{val // 10}.{val % 10}' if val >= 10 else str(val)
                if sz.endswith('.0'): sz = sz[:-2]
                full = f'{sz} ({angle})'
                size = full
            elif m := re.search(r'(?<!\d)([0-9]{1,2})(#|号|STD)', text):
                size = m.group(1)
        elif model == 'Stryker Accolade II':
            if m := re.search(r'672([01])-([0-9]{2})([0-9]{2})', text):
                angle = '132°' if m.group(1) == '0' else '127°'
                size = f'{int(m.group(2))} ({angle})'
        elif model == 'Stryker Secur-Fit':
            if m := re.search(r'605([12])-([0-9]{2})[0-9]{2}', text):
                angle = '132°' if m.group(1) == '1' else '127°'
                size = f'{int(m.group(2))} ({angle})'
            elif m := re.search(r'(?<!\d)([0-9]{1,2})(#|号|STD)', text):
                size = m.group(1)
        elif model == 'DePuy Tri-Lock':
            if m := re.search(r'1012-(?:04|14)-([0-9]{3})', text): size = str(int(m.group(1)) // 10)
        elif model == 'DePuy SUMMIT':
            mapping = {'070': '1', '080': '2', '090': '3', '100': '4', '110': '5', '120': '6', '135': '7', '150': '8',
                       '165': '9', '180': '10'}
            if m := re.search(r'(?:1570|2570)-?01-?(070|080|090|100|110|120|135|150|165|180)', text):
                size = mapping.get(m.group(1), '')
        elif model == 'DePuy S-ROM':
            srom_map = {'3206': '6', '3207': '7', '3208': '8', '3209': '9', '3291': '9', '3292': '11', '3293': '13',
                        '3294': '15', '3295': '17', '3296': '19', '3191': '9', '3192': '11', '3193': '13', '3391': '9',
                        '3392': '11', '3393': '13'}
            if m := re.search(r'52-?(3[123][0-9]{2})', text):
                size = srom_map.get(m.group(1), '')
            elif m := re.search(r'5232([0-9]{2})', text):
                size = str(int(m.group(1)))
        elif model == 'Wright Profemur':
            if m := re.search(r'PHA00(232|234|236|238|240|242|244|246|248)', text):
                size = str((int(m.group(1)) - 232) // 2 + 1)
            elif m := re.search(r'PRTL[S0]{1,2}([0-9]{2})', text):
                size = str(int(m.group(1)))
        elif model == 'Keyi Bangen SQKA':
            if 'SQKA-Ⅱ' in text or 'SQKA-II' in text:
                size = '2'
            elif m := re.search(r'([0-9]{1,2})(#|号)', text):
                size = m.group(1)
        elif model == 'Zimmer M/L Taper':
            mapping = {'04': '4', '05': '5', '06': '6', '07': '7.5', '09': '9', '10': '10', '11': '11', '12': '12.5',
                       '13': '13.5', '15': '15', '16': '16.25', '17': '17.5', '20': '20', '22': '22.5'}
            if m := re.search(r'7711-0?([0-9]{2})-', text):
                size = mapping.get(m.group(1), '')
            elif 'ML' in spec.upper() or 'ML' in name.upper():
                size = 'ML'
        elif model == 'AK Medical ML-TP':
            if m := re.search(r'1100-28(0[1-9]|[1-5][0-9])', text):
                val = int(m.group(1))
                sz = f'{val // 10}.{val % 10}' if val >= 10 else str(val)
                size = sz[:-2] if sz.endswith('.0') else sz
        elif model == 'Waldemar Link LCU':
            if m := re.search(r'165-0([0-2][0-9])', text): size = str(int(m.group(1)))
        elif model == 'DePuy Corail':
            corail_map = {
                '92507': '8 (标准无领)', '92509': '9 (标准无领)', '92510': '10 (标准无领)', '92511': '11 (标准无领)',
                '92512': '12 (标准无领)',
                '92513': '13 (标准无领)', '92514': '14 (标准无领)', '92515': '15 (标准无领)', '92516': '16 (标准无领)',
                '92518': '18 (标准无领)', '92520': '20 (标准无领)',
                '92498': '8 (标准带领)', '92499': '9 (标准带领)', '92500': '10 (标准带领)', '92501': '11 (标准带领)',
                '92502': '12 (标准带领)',
                '92503': '13 (标准带领)', '92504': '14 (标准带领)', '92505': '15 (标准带领)', '92506': '16 (标准带领)',
                '92508': '18 (标准带领)', '92521': '20 (标准带领)',
                '20309': '9 (高偏心无领)', '20310': '10 (高偏心无领)', '20311': '11 (高偏心无领)',
                '20312': '12 (高偏心无领)', '20313': '13 (高偏心无领)', '20314': '14 (高偏心无领)',
                '20315': '15 (高偏心无领)', '20316': '16 (高偏心无领)',
                '93709': '9 (内翻带领)', '93710': '10 (内翻带领)', '93711': '11 (内翻带领)', '93712': '12 (内翻带领)',
                '93713': '13 (内翻带领)', '93714': '14 (内翻带领)', '93715': '15 (内翻带领)', '93716': '16 (内翻带领)',
                '93718': '18 (内翻带领)', '93720': '20 (内翻带领)',
                '20106': '6 (DDH)', '98010': '10 (翻修标准)', '98011': '11 (翻修标准)', '98012': '12 (翻修标准)',
                '98013': '13 (翻修标准)', '98014': '14 (翻修标准)', '98015': '15 (翻修标准)', '98016': '16 (翻修标准)',
                '98018': '18 (翻修标准)', '98020': '20 (翻修标准)',
                '98110': '10 (翻修高偏心)', '98111': '11 (翻修高偏心)', '98112': '12 (翻修高偏心)',
                '98113': '13 (翻修高偏心)', '98114': '14 (翻修高偏心)', '98115': '15 (翻修高偏心)',
                '98116': '16 (翻修高偏心)', '98118': '18 (翻修高偏心)', '98120': '20 (翻修高偏心)',
            }
            if m := re.search(r'(?:3L|L)?(92[45][0-9]{2}|203[0-1][0-9]|937[0-2][0-9]|20106|98[01][0-9]{2})', text):
                size = corail_map.get(m.group(1), '')
            elif m := re.search(r'(?<!\d)([0-9]{1,2})(号|#|STD)', text):
                size = m.group(1) + (' (标准带领)' if '带领' in text else ' (标准无领)')
        elif model == 'Wagner Cone':
            if m := re.search(r'01\.00561\.([23])(1[3-9]|2[0-2])', text):
                angle_str = '125°' if m.group(1) == '2' else '135°'
                full = f'{m.group(2)} ({angle_str})'
                size = full
            elif m := re.search(r'(125|135)°([0-9]{2})', text):
                size = f'{m.group(2)} ({m.group(1)}°)'
            elif m := re.search(r'01\.00561\.3([0-9]{2})', text):
                size = f'{m.group(1)} (135°)'
        elif model == 'Zimmer Wagner SL':
            if m := re.search(r'01\.0010[123]\.[0-9](1[4-9]|2[0-5])', text): size = m.group(1)
        elif model == 'Smith & Nephew Plus-TS':
            if m := re.search(r'A23([0-9]{2})', text): size = str(int(m.group(1)))

        if not size:
            for s in sizes:
                base = str(s).split(' (')[0]
                if re.search(r'(?<!\d)' + re.escape(base) + r'(号|#|SIZE|\b)', text):
                    if '(' in s:
                        angle = re.search(r'\((12[57]|13[25])°\)', s)
                        if angle and angle.group(1) in text: size = s; break
                    else:
                        size = s; break
            if not size:
                for s in sizes:
                    if re.search(r'(?<!\d)' + re.escape(str(s).split(' (')[0]) + r'(号|#|SIZE|\b)',
                                 text): size = s; break
    return model, size


def parse_head_offset(name, spec):
    """识别股骨头偏距，处理各种厂家特有的数值编码和带正负号的文本"""
    text = (str(name) + ' ' + str(spec)).upper()
    if 'CH-12/14' in text and not re.search(r'\+[0-9]', text): return 0.0

    if m := re.search(r'1365-?(\d{2})-?(\d{3})', text):
        d_c, o_c = m.group(1), m.group(2)
        if o_c.startswith('2'): return {'210': 0.0, '220': 4.0, '230': 8.5, '240': 12.0}.get(o_c, '')
        if o_c.startswith('3'): return {'310': 1.5, '320': 5.0, '330': 8.5, '340': 12.0}.get(o_c, '')
        if o_c in ['000', '500', '150']:
            if o_c == '150': return 1.5
            if d_c.startswith('1'): return {1: 1.5, 2: 5.0, 3: 8.5, 4: 12.0, 5: 15.5}.get(int(d_c[1]), '')
            if d_c.startswith('2'): return {1: 1.0, 2: 5.0, 3: 9.0, 4: 13.0, 5: 17.0}.get(int(d_c[1]), '')
            if d_c.startswith('5'): return {1: 1.5, 2: 5.0, 3: 8.5, 4: 12.0, 5: 15.5}.get(int(d_c[1]), '')
            if d_c in ['04', '05', '06', '07', '08', '09']: return {'04': -2.0, '05': 1.5, '06': 5.0, '07': 8.5,
                                                                    '08': 12.0, '09': 15.5}.get(d_c, '')

    if m := re.search(r'(?:17|18|6001)-(\d{2})(\d{2})', text):
        return {'00': 0.0, '05': 5.0, '10': 10.0, '25': 2.5}.get(m.group(2), '')
    if m := re.search(r'18-(?:28|32|36|40)(-[35]|00|05|25)', text):
        return {'-3': -2.5, '-5': -5.0, '00': 0.0, '05': 5.0, '25': 2.5}.get(m.group(1), '')
    if m := re.search(r'6570-?0-?([0-5][23][26]|00)', text):
        return {'032': -4.0, '132': 0.0, '232': 4.0, '036': -5.0, '436': -2.5, '136': 0.0, '536': 2.5, '236': 5.0,
                '00': 0.0}.get(m.group(1), '')
    elif m := re.search(r'65700([0-5][23][26])', text):
        return {'032': -4.0, '132': 0.0, '232': 4.0, '036': -5.0, '436': -2.5, '136': 0.0, '536': 2.5, '236': 5.0}.get(
            m.group(1), '')
    elif m := re.search(r'6570-0-32([84])', text):
        return {'8': -2.7, '4': 0.0}.get(m.group(1), '')

    if m := re.search(r'8018-?(\d{2})-?(\d{2})', text):
        return {'01': -3.5, '02': 0.0, '03': 3.5, '04': 7.0, '05': 10.5, '20': -2.0}.get(m.group(2), '')
    if m := re.search(r'8018(\d{2})(\d{2})', text):
        return {'01': -3.5, '02': 0.0, '03': 3.5, '04': 7.0, '05': 10.5}.get(m.group(2), '')
    if m := re.search(r'877[57]-0?(\d{2})-0?(\d)', text):
        return {1: -3.5, 2: 0.0, 3: 3.5, 4: 7.0}.get(int(m.group(2)), '')
    if m := re.search(r'877[57]0\d{2}0(\d)', text):
        return {1: -3.5, 2: 0.0, 3: 3.5, 4: 7.0}.get(int(m.group(1)), '')
    if m := re.search(r'87750\d{3}([1-4])', text):
        return {1: -3.5, 2: 0.0, 3: 3.5, 4: 7.0}.get(int(m.group(1)), '')

    if m := re.search(r'PHA0440([2468])', text): return {2: -3.5, 4: 0.0, 6: 3.5, 8: 7.0}.get(int(m.group(1)), '')
    if m := re.search(r'2601280([246])', text): return {2: 0.0, 4: 3.5, 6: 7.0}.get(int(m.group(1)), '')
    if m := re.search(r'2600000([456])', text): return {4: -3.5, 5: 0.0, 6: 3.5}.get(int(m.group(1)), '')

    if m := re.search(r'2202-0([012])', text): return {0: 0.0, 1: 0.0, 2: 4.0}[int(m.group(1))]
    if m := re.search(r'128-792/0([1-4])', text):
        return {1: -4.0, 2: 0.0, 3: 4.0, 4: 7.0}.get(int(m.group(1)), '')
    if m := re.search(r'128-793/0([1-4])', text):
        return {1: -4.0, 2: 0.0, 3: 4.0, 4: 8.0}.get(int(m.group(1)), '')
    if m := re.search(r'17\.\d{2}\.0([567])', text): return {5: -3.5, 6: 0.0, 7: 3.5}[int(m.group(1))]
    if 'ZTCT01' in text or 'SHXDZ301' in text or 'LHXDZ301' in text: return 0.0
    if 'CH-12/14' in text and not re.search(r'[+-][0-9]', text): return 0.0

    if any(x in text for x in ['MP', 'MID', '中偏', '32M', '28M', '/0）']): return 0.0
    if '36-12/14 L' in text: return 5.0
    if m := re.search(r'([23][268])\+([0-9](?:\.\d)?)', text.replace(' ', '')): return float(m.group(2))
    clean_text = re.sub(r'(18-|6570-|1365-?|7711-|8775-|8777-|7134-)\d+', '', text)
    if m := re.search(r'([+-](?:0|1[0-6]|[1-9])(?:\.\d)?)(?!\d)', clean_text.replace(' ', '')):
        try:
            val = float(m.group(1))
            if any(abs(o - val) < 0.01 for o in [x for x in HEAD_OFFSET if x != '']): return val
        except:
            pass
    return ''


def extract_prothesis_info(config_file, input_file='THATable.xlsx', output_file='THATableExtracted.xlsx'):
    """主提取函数"""
    cfg_path = Path(config_file)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    if not Path(input_file).exists():
        print(f'错误: 输入文件 {input_file} 不存在。')
        return

    df = pd.read_excel(input_file)
    result = []
    for _, row in tqdm(df.iterrows()):
        prl = row.get('prl')
        if pd.isna(prl): continue
        vals = [str(row.get(c, '')) for c in
                ['股骨柄名称', '股骨柄规格', '股骨头名称', '股骨头规格', '髋臼杯名称', '髋臼杯规格', '内衬名称',
                 '内衬规格']]
        s_n, s_s, h_n, h_s, c_n, c_s, l_n, l_s = [v if v != 'nan' else '' for v in vals]

        model, size = parse_stem(s_n, s_s)
        h_offset = parse_head_offset(h_n, h_s)

        # --- 髋臼杯外径 ---
        c_text = (c_n + ' ' + c_s).upper()
        c_diam = ''
        if 'IT/S' in c_text:
            if m := re.search(r'(?<!\d)([4-7][0-9])(?=[A-Z\s\)\/\）]|$)', c_text):
                c_diam = m.group(1)
            if not c_diam:
                it_s_map = {'A': '42', 'B': '44', 'C': '46', 'D': '48', 'E': '50', 'F': '52', 'G': '54', 'H': '56',
                            'J': '58', 'K': '60'}
                for k, v in it_s_map.items():
                    if re.search(r'\b' + k + r'\b', c_text): c_diam = v; break
        if not c_diam:
            if m := re.search(r'6202([4-6][0-9])', c_text):
                c_diam = m.group(1)
            elif m := re.search(r'364300([4-6][0-9])', c_text):
                c_diam = m.group(1)
            elif m := re.search(r'A2653-([4-6][0-9])', c_text):
                c_diam = m.group(1)
            elif m := re.search(r'87570([4-6][0-9])', c_text):
                c_diam = m.group(1)
            elif m := re.search(r'8753-0([4-6][0-9])', c_text):
                c_diam = m.group(1)
            elif m := re.search(r'1217[0-9]{2,3}([4-6][0-9])', c_text):
                c_diam = m.group(1)
            if not c_diam:
                c_clean = re.sub(r'(?:1217|542|623|00-7000|00-875[37]|00-620[02])-?\d{2,}-?', '', c_text)
                if m := re.search(r'([4-7][0-9])\s*MM', c_clean):
                    c_diam = m.group(1)
                elif m := re.search(r'(?<=0)([4-7][0-9])(?=[A-Z\s\)\）]|$)', c_clean):
                    c_diam = m.group(1)
                elif m := re.search(r'(?<!\d)([4-7][0-9])(?=[A-Z\s\)\）]|$)', c_clean):
                    c_diam = m.group(1)

            # 从内衬规格中辅助提取杯外径
            if not c_diam:
                l_text_temp = (l_n + ' ' + l_s).upper()
                if m := re.search(r'6310-?0([4-7][0-9])-?(22|28|32|36|40)', l_text_temp): c_diam = m.group(1)

        # --- 股骨头外径 ---
        h_text, l_text = (h_n + ' ' + h_s).upper(), (l_n + ' ' + l_s).upper()
        h_diam = ''
        if not ('2600' in h_text and '球头' not in h_text and '股骨头' not in h_text):
            if m := re.search(r'(?<!\d)(22|28|32|36|40|44)\s*MM', h_text):
                h_diam = m.group(1)
            elif m := re.search(r'^(22|28|32|36|40|44)\b', h_s):
                h_diam = m.group(1)
            if not h_diam:
                if m := re.search(r'1365-?([23][268]|4[04])\b', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'1365-?([0125])[0-9]-?[05]00', h_text):
                    h_diam = {'1': '28', '2': '32', '5': '36', '0': '40'}[m.group(1)]
                elif m := re.search(r'136511[45]00', h_text):
                    h_diam = '28'
                elif m := re.search(r'136511600', h_text):
                    h_diam = '32'
                elif m := re.search(r'1365(\d{2})[123]\d{2}', h_text):
                    if m.group(1) in ['28', '32', '36', '40', '44']: h_diam = m.group(1)
                elif '522022' in h_text:
                    h_diam = '22'
                elif m := re.search(r'(?:17|18)-(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'6570-?0-?(?:[0-9])?(22|28|32|36|40|44)', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'8018-?(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'87750([23][286]|40)', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'PHA0440([2468])', h_text):
                    h_diam = '28'
                elif m := re.search(r'6260-5-0(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'6260-4-1(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'6001-(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'2601(\d{2})', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'2600000([456])', h_text):
                    h_diam = '28'
                elif m := re.search(r'2202-\d{2}(\d{2})', h_text):
                    h_diam = m.group(1)
                elif '128-792' in h_text:
                    h_diam = '32'
                elif '128-793' in h_text:
                    h_diam = '36'
                elif m := re.search(r'([23][268])[SL]HXDZ301', h_text):
                    h_diam = m.group(1)
                elif 'CH-12/14' in h_text:
                    if m := re.search(r'(22|28|32|36|40|44)', h_text): h_diam = m.group(1)
            # 内衬辅助及纠正
            if not h_diam or len(h_s) > 50:
                if m := re.search(r'(22|28|32|36|40|44)\s*MM', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'/([23][268])HXDZ302', l_text):
                    h_diam = m.group(1)

            # 货号纠正：只有当股骨头记录无法识别直径时，才参考内衬货号（防止内衬记录为系统默认占位符导致的错误覆盖）
            if not h_diam:
                if m := re.search(r'2047-([234][268])[0-9]{2}[A-Z]?', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'623-10-([234][268])[A-Z]?', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'6310-?0[4-6][0-9]-?([234][268])\b', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'63100[4-6][0-9]([23][268])\b', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'1221-?([234][268])', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'1219-?([234][268])', l_text):
                    h_diam = m.group(1)
                elif m := re.search(r'182-150/0([1-5])', l_text):
                    h_diam = {'1': '28', '2': '32', '3': '36', '4': '36', '5': '40'}[m.group(1)]
                elif m := re.search(r'1218-?8([123])', l_text):
                    h_diam = {'1': '36', '2': '32', '3': '28'}[m.group(1)]

            if not h_diam:
                if m := re.search(r'(22|28|32|36|40|44)M', h_text):
                    h_diam = m.group(1)
                elif m := re.search(r'(22|28|32|36|40|44)(?=-12/14)', h_text):
                    h_diam = m.group(1)

        # --- 内衬偏心距 ---
        l_off_text, l_off = (l_n + ' ' + l_s).upper(), ''

        # 1. 明确的偏距关键词识别
        if any(x in l_off_text for x in
               ['中立', 'NEUTRAL', '标准', 'STANDARD', '防脱位', '32E', '36E', '32D', '40E', '32MM ID']):
            l_off = '0'
        elif '+4' in l_off_text or 'LATERALIZED' in l_off_text or '偏心' in l_off_text:
            if '+4' in l_off_text:
                l_off = '4'
            elif '+6' in l_off_text:
                l_off = '6'
            elif '+7' in l_off_text:
                l_off = '7'
            elif '4MM' in l_off_text:
                l_off = '4'
            elif '6MM' in l_off_text:
                l_off = '6'
            elif '7MM' in l_off_text:
                l_off = '7'

        # 2. 货号特定编码识别 (仅在无法通过关键词判断时)
        if not l_off:
            # DePuy 1219/1221/1218 系列: 第 8 位数字 (倒数第 2 位) 常代表类型
            # 1219-XX-1XX: 10度聚乙烯内衬，通常带有 +4 偏距
            if m := re.search(r'1219-\d{2}-1\d{2}', l_off_text):
                l_off = '4'
            # 1221-XX-1XX: ALTRX 高交联聚乙烯内衬，1XX 通常代表带有偏距或角度
            elif m := re.search(r'1221-\d{2}-1\d{2}', l_off_text):
                if 'ALTRX' in l_off_text and any(x in l_off_text for x in ['+4', 'LATERAL']):
                    l_off = '4'
                else:
                    l_off = '0'  # 如果没有明确偏距关键字，ALTRX 1XX 系列可能是中立带角度，偏距为 0
            # 1218 系列: 陶瓷内衬通常为 0 偏距
            elif '1218' in l_off_text:
                l_off = '0'
            # Stryker 6310 系列: 04/05 代表偏心
            elif m := re.search(r'6310-(04|05)-', l_off_text):
                l_off = '4'
            elif m := re.search(r'6310-?07-?', l_off_text):
                l_off = '7'
            # 其他明确 0 偏距的货号
            elif any(x in l_off_text for x in
                     ['631003', '631004', '631005', '631006', 'HXDZ302', '3200-8752-008', '2047-', '623-10-']):
                l_off = '0'
            elif '877500' in l_off_text or 'PHA04504' in l_off_text:
                l_off = '0'

        # 3. 后备提取
        if not l_off:
            if m := re.search(r'\+([467])\b', l_off_text): l_off = m.group(1)

        result.append({
            'prl': prl, '股骨柄型号': model, '股骨柄规格': size, '股骨头偏距': h_offset,
            '髋臼杯外径': c_diam, '股骨头外径': h_diam, '内衬偏心距': l_off,
            '原始股骨柄名称': s_n, '原始股骨柄规格': s_s, '原始股骨头名称': h_n, '原始股骨头规格': h_s,
            '原始髋臼杯名称': c_n, '原始髋臼杯规格': c_s, '原始内衬名称': l_n, '原始内衬规格': l_s
        })

        data = {}
        try:
            data['femoral_spec'] = [str(model), str(size)]
        except (Exception,):
            pass
        try:
            data['head_offset'] = float(h_offset)
        except (Exception,):
            pass
        try:
            data['cup_outer'] = int(c_diam)
        except (Exception,):
            pass
        try:
            data['head_outer'] = int(h_diam)
        except (Exception,):
            pass
        try:
            data['liner_offset'] = float(l_off)
        except (Exception,):
            pass

        pid, rl = prl.split('_')
        data = tomlkit.dumps(data).encode('utf-8')
        client.put_object('pair', '/'.join([pid, rl, 'context.toml']), BytesIO(data), len(data))

    pd.DataFrame(result).to_excel(output_file, index=False)
    print(f'提取完成: {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('-i', '--input', default='THATable.xlsx')
    parser.add_argument('-o', '--output', default='THATableExtracted.xlsx')
    args = parser.parse_args()
    extract_prothesis_info(args.config, args.input, args.output)
