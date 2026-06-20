# -*- coding: utf-8 -*-
"""
生成合成 CD-SEM 数据（用于演示和冒烟测试）。

使用 forward_model.compute_bossung_cd 以一组"真实"参数生成 CD 曲线，
并添加模拟的 CD-SEM 量测噪声。
"""
import sys
import csv
import numpy as np
from pathlib import Path

# 允许直接运行
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from calibration.forward_model import compute_bossung_cd
from calibration.schemas import PatternType


# "真实"参数：FAB 工艺下被模拟的实际值
TRUE_PARAMS = {
    'resist_threshold': 0.285,
    'diffusion_length': 12.5,
    'na_effective': 1.335,
    'wavelength_effective': 193.2,
    'sigma_effective': 0.78,
    'dose_to_clear': 0.48,
    'resist_contrast': 3.2,
}

# Focus-Dose 采样网格
FOCUS_VALUES_NM = [-150, -100, -60, -30, 0, 30, 60, 100, 150]
DOSE_VALUES = [0.85, 0.92, 1.0, 1.08, 1.15]

# 测试的图形种类（target_cd, pitch, pattern_type）
TEST_STRUCTURES = [
    (45,  90,  PatternType.LINE_SPACE),
    (60,  120, PatternType.LINE_SPACE),
    (80,  160, PatternType.LINE_SPACE),
    (100, 200, PatternType.LINE_SPACE),
    (55,  None, PatternType.ISOLATED_LINE),
    (70,  140, PatternType.CONTACT_HOLE),
]

# CD-SEM 量测不确定度 1σ (nm)
MEASUREMENT_UNCERTAINTY = 1.2


def generate_dataset(output_csv: Path, seed: int = 20260120) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    mid = 0

    for (target_cd, pitch, pt) in TEST_STRUCTURES:
        for focus in FOCUS_VALUES_NM:
            for dose in DOSE_VALUES:
                # 多个 site（模拟晶圆多点量测）
                for site_idx, site in enumerate(['C', 'TL', 'BR']):
                    mid += 1
                    clean_cd = compute_bossung_cd(
                        focus, dose, target_cd, pitch,
                        TRUE_PARAMS, pt, complexity='standard',
                    )
                    # 站点间的系统偏移（±0.8 nm）
                    site_offset = rng.normal(0, 0.8)
                    noise = rng.normal(0, MEASUREMENT_UNCERTAINTY)
                    measured = float(clean_cd + site_offset + noise)

                    rows.append({
                        'measurement_id': f"P{mid:05d}",
                        'site_name': site,
                        'target_cd': target_cd,
                        'measured_cd': round(measured, 3),
                        'focus': focus,
                        'dose': dose,
                        'pattern_type': pt.value,
                        'pitch': '' if pitch is None else pitch,
                        'measurement_uncertainty': MEASUREMENT_UNCERTAINTY,
                        'layer': 'M1',
                    })

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成 {len(rows)} 个合成量测点 → {output_csv}")
    print(f"  True params: {TRUE_PARAMS}")


if __name__ == '__main__':
    out = Path(__file__).parent / 'sample_cd_sem_data.csv'
    generate_dataset(out)
