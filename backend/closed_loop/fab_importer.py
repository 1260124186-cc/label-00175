# -*- coding: utf-8 -*-
"""
Fab 数据导入模块

负责从指定目录定期扫描并导入 Fab 最新的 CD-SEM 量测 CSV 文件。
功能：
1. 监控目录下的新 CSV 文件（glob 模式匹配）
2. 基于文件哈希的增量导入（避免重复处理）
3. 导入历史记录持久化（JSON）
4. 合并多文件为统一 CDSEMDataset
5. 导入成功后自动归档原文件
"""

import hashlib
import json
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import numpy as np

from backend.calibration.schemas import (
    CDSEMDataset,
    CDSEMDataPoint,
)
from backend.calibration.data_loader import load_cd_sem_from_csv

from .schemas import (
    FabImportConfig,
    FabImportResult,
    ImportedFileRecord,
)

logger = logging.getLogger(__name__)


def _file_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """计算文件 MD5 哈希"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class FabDataImporter:
    """
    Fab CD-SEM 数据导入器

    典型用法::

        importer = FabDataImporter(config)
        result = importer.import_new_data()
        if result.merged_dataset is not None:
            # 使用 result.merged_dataset 进行后续分析
    """

    def __init__(self, config: FabImportConfig):
        self.config = config
        self._history: List[Dict[str, Any]] = []
        self._load_history()

    # ------------------------------------------------------------------
    # 历史记录管理
    # ------------------------------------------------------------------
    def _load_history(self) -> None:
        """从 JSON 文件加载导入历史"""
        path = Path(self.config.history_file)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
                logger.info(f"已加载导入历史: {len(self._history)} 条记录")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"导入历史文件损坏，将重建: {e}")
                self._history = []
        else:
            self._history = []
            path.parent.mkdir(parents=True, exist_ok=True)

    def _save_history(self) -> None:
        """保存导入历史到 JSON 文件"""
        path = Path(self.config.history_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def _is_already_imported(self, file_path: Path, file_hash: str) -> bool:
        """检查文件是否已导入过（按哈希或完整路径判断）"""
        for rec in self._history:
            if rec.get('file_hash') == file_hash:
                return True
            if rec.get('file_path') == str(file_path):
                return True
        return False

    # ------------------------------------------------------------------
    # 文件扫描
    # ------------------------------------------------------------------
    def _scan_candidate_files(self) -> List[Path]:
        """扫描监控目录下匹配模式的候选文件"""
        watch_dir = Path(self.config.watch_dir)
        if not watch_dir.exists():
            logger.warning(f"监控目录不存在: {watch_dir}")
            return []

        candidates = list(watch_dir.glob(self.config.file_pattern))

        if self.config.lookback_days > 0:
            cutoff = datetime.now() - timedelta(days=self.config.lookback_days)
            cutoff_ts = cutoff.timestamp()
            candidates = [
                p for p in candidates
                if p.stat().st_mtime >= cutoff_ts
            ]

        candidates.sort(key=lambda p: p.stat().st_mtime)
        logger.info(
            f"扫描到 {len(candidates)} 个候选文件 "
            f"(pattern={self.config.file_pattern})"
        )
        return candidates

    # ------------------------------------------------------------------
    # 单文件导入
    # ------------------------------------------------------------------
    def _import_single_file(self, filepath: Path) -> tuple[ImportedFileRecord, Optional[CDSEMDataset]]:
        """导入单个 CSV 文件，返回 (记录, 数据集)"""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            fhash = _file_md5(filepath)
            fsize = filepath.stat().st_size

            dataset = load_cd_sem_from_csv(
                filepath,
                encoding=self.config.encoding,
                delimiter=self.config.delimiter,
            )

            rec = ImportedFileRecord(
                file_path=str(filepath),
                file_hash=fhash,
                file_size=fsize,
                import_timestamp=ts,
                n_points=len(dataset),
                fab_name=dataset.fab_name,
                lot_id=dataset.lot_id,
                wafer_id=dataset.wafer_id,
                mask_set_id=dataset.mask_set_id,
                process_node=dataset.process_node,
                success=True,
            )
            logger.info(
                f"  ✓ 导入成功: {filepath.name}, {len(dataset)} 点, "
                f"lot={dataset.lot_id}, wafer={dataset.wafer_id}"
            )
            return rec, dataset

        except Exception as e:
            logger.error(f"  ✗ 导入失败: {filepath.name}: {e}")
            rec = ImportedFileRecord(
                file_path=str(filepath),
                file_hash=(
                    _file_md5(filepath) if filepath.exists() else ""
                ),
                file_size=filepath.stat().st_size if filepath.exists() else 0,
                import_timestamp=ts,
                n_points=0,
                fab_name="",
                lot_id="",
                wafer_id="",
                mask_set_id="",
                process_node="",
                success=False,
                error_message=str(e),
            )
            return rec, None

    # ------------------------------------------------------------------
    # 归档
    # ------------------------------------------------------------------
    def _archive_file(self, filepath: Path) -> Optional[Path]:
        """将已处理文件移动到归档目录"""
        if not self.config.auto_archive or self.config.archive_dir is None:
            return None

        archive_dir = Path(self.config.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        dst = archive_dir / f"{ts}_{filepath.name}"
        try:
            shutil.move(str(filepath), str(dst))
            logger.info(f"  已归档: {filepath.name} → {dst}")
            return dst
        except OSError as e:
            logger.warning(f"  归档失败: {filepath.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # 数据集合并
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_datasets(datasets: List[CDSEMDataset]) -> Optional[CDSEMDataset]:
        """合并多个 CDSEMDataset"""
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]

        all_points: List[CDSEMDataPoint] = []
        fab_names = set()
        process_nodes = set()
        mask_set_ids = set()
        magnifications = []

        for ds in datasets:
            all_points.extend(ds.points)
            if ds.fab_name:
                fab_names.add(ds.fab_name)
            if ds.process_node:
                process_nodes.add(ds.process_node)
            if ds.mask_set_id:
                mask_set_ids.add(ds.mask_set_id)
            magnifications.append(ds.magnification)

        return CDSEMDataset(
            points=all_points,
            magnification=float(np.mean(magnifications)) if magnifications else 4.0,
            fab_name=",".join(sorted(fab_names)),
            process_node=",".join(sorted(process_nodes)),
            mask_set_id=",".join(sorted(mask_set_ids)),
            wafer_id="",
            lot_id="",
            comments=(
                f"Merged {len(datasets)} datasets, "
                f"total {len(all_points)} points"
            ),
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def import_new_data(self,
                        force: bool = False,
                        ) -> FabImportResult:
        """
        导入所有尚未处理的新文件

        Args:
            force: 强制重新导入，即使文件已处理过

        Returns:
            FabImportResult 对象
        """
        logger.info("开始 Fab 数据导入...")
        candidates = self._scan_candidate_files()

        imported_records: List[ImportedFileRecord] = []
        datasets: List[CDSEMDataset] = []
        n_new = 0
        n_skipped = 0
        n_failed = 0

        for fp in candidates:
            try:
                fhash = _file_md5(fp)
            except OSError:
                fhash = ""

            if not force and self._is_already_imported(fp, fhash):
                n_skipped += 1
                logger.debug(f"  跳过(已导入): {fp.name}")
                continue

            rec, dataset = self._import_single_file(fp)
            imported_records.append(rec)
            self._history.append(rec.to_dict())

            if rec.success and dataset is not None:
                datasets.append(dataset)
                n_new += 1
                self._archive_file(fp)
            else:
                n_failed += 1

        self._save_history()

        merged = self._merge_datasets(datasets) if datasets else None
        result = FabImportResult(
            imported_files=imported_records,
            merged_dataset=merged,
            new_files_count=n_new,
            skipped_files_count=n_skipped,
            failed_files_count=n_failed,
        )
        logger.info(
            f"Fab 数据导入完成: 新{n_new} / 跳过{n_skipped} / 失败{n_failed}, "
            f"总 {result.total_points} 点"
        )
        return result

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------
    def get_import_history(self,
                           limit: Optional[int] = None,
                           ) -> List[ImportedFileRecord]:
        """获取导入历史记录"""
        records = [
            ImportedFileRecord(**r) for r in self._history
        ]
        if limit is not None:
            records = records[-limit:]
        return records

    def clear_history(self) -> int:
        """清空历史记录，返回删除的记录数"""
        n = len(self._history)
        self._history = []
        self._save_history()
        logger.info(f"已清空 {n} 条导入历史")
        return n


def import_fab_data(config: Union[FabImportConfig, Dict[str, Any]],
                    force: bool = False,
                    ) -> FabImportResult:
    """
    便捷函数：导入 Fab 新数据

    Args:
        config: FabImportConfig 或字典（会自动转换）
        force: 是否强制重新导入

    Returns:
        FabImportResult
    """
    if isinstance(config, dict):
        cfg = FabImportConfig(**{
            k: v for k, v in config.items()
            if k in FabImportConfig.__dataclass_fields__
        })
    else:
        cfg = config
    importer = FabDataImporter(cfg)
    return importer.import_new_data(force=force)
