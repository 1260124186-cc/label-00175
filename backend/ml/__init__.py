# -*- coding: utf-8 -*-
"""
机器学习模块

子模块:
    hotspot_predictor: 版图热点 CNN 预测器，快速筛出高风险区域
"""

from .hotspot_predictor import (
    HotspotPredictorConfig,
    HotspotPatchDataset,
    HotspotDatasetConfig,
    HotspotCNN,
    HotspotTrainingConfig,
    HotspotScanResult,
    build_hotspot_cnn,
    generate_hotspot_dataset,
    train_hotspot_predictor,
    load_hotspot_predictor,
    scan_layout_for_hotspots,
    export_hotspot_predictor,
)

__all__ = [
    'HotspotPredictorConfig',
    'HotspotPatchDataset',
    'HotspotDatasetConfig',
    'HotspotCNN',
    'HotspotTrainingConfig',
    'HotspotScanResult',
    'build_hotspot_cnn',
    'generate_hotspot_dataset',
    'train_hotspot_predictor',
    'load_hotspot_predictor',
    'scan_layout_for_hotspots',
    'export_hotspot_predictor',
]
