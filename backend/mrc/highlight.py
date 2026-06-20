# -*- coding: utf-8 -*-
"""
违规区域高亮标注模块

将 MRC 检查发现的违规区域在掩模图上进行可视化标注。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np

from .violations import MRCViolation, MRCCheckResult, ViolationType
from .rules import MRCRuleType, MRCRuleSeverity

logger = logging.getLogger(__name__)


@dataclass
class HighlightStyle:
    """高亮样式配置"""
    color_fatal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    color_error: Tuple[float, float, float] = (1.0, 0.5, 0.0)
    color_warning: Tuple[float, float, float] = (1.0, 1.0, 0.0)
    color_info: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    alpha: float = 0.6
    line_width: int = 2
    show_bbox: bool = True
    show_centroid: bool = True
    centroid_marker_size: int = 5

    def get_color(self, severity: MRCRuleSeverity) -> Tuple[float, float, float]:
        """根据严重级别获取颜色"""
        mapping = {
            MRCRuleSeverity.FATAL: self.color_fatal,
            MRCRuleSeverity.ERROR: self.color_error,
            MRCRuleSeverity.WARNING: self.color_warning,
            MRCRuleSeverity.INFO: self.color_info,
        }
        return mapping.get(severity, self.color_error)


class ViolationHighlighter:
    """
    违规区域高亮器

    在掩模图上叠加显示违规区域，支持多种输出格式。
    """

    def __init__(self, style: Optional[HighlightStyle] = None):
        self.style = style or HighlightStyle()

    def generate_overlay_mask(self,
                              mask: np.ndarray,
                              violations: List[MRCViolation],
                              ) -> np.ndarray:
        """
        生成违规区域叠加掩模

        Args:
            mask: 原始二值掩模 (H, W)
            violations: 违规列表

        Returns:
            RGB 叠加图像 (H, W, 3)，值域 [0, 1]
        """
        h, w = mask.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float64)

        if mask.ndim == 2:
            gray = mask.astype(np.float64)
            if np.max(gray) > 1.0:
                gray /= 255.0
            for c in range(3):
                overlay[:, :, c] = gray
        else:
            overlay = mask.astype(np.float64).copy()
            if np.max(overlay) > 1.0:
                overlay /= 255.0

        for v in violations:
            self._mark_violation(overlay, v)

        return np.clip(overlay, 0.0, 1.0)

    def _mark_violation(self, overlay: np.ndarray, violation: MRCViolation) -> None:
        """在叠加图上标记单个违规"""
        color = self.style.get_color(violation.severity)
        bbox = violation.region.bbox
        ymin, xmin, ymax, xmax = bbox
        h, w = overlay.shape[:2]

        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(h, ymax)
        xmax = min(w, xmax)

        if self.style.show_bbox:
            lw = self.style.line_width
            for c in range(3):
                overlay[ymin:ymax, xmin:xmin + lw, c] = (
                    overlay[ymin:ymax, xmin:xmin + lw, c] * (1 - self.style.alpha)
                    + color[c] * self.style.alpha
                )
                overlay[ymin:ymax, xmax - lw:xmax, c] = (
                    overlay[ymin:ymax, xmax - lw:xmax, c] * (1 - self.style.alpha)
                    + color[c] * self.style.alpha
                )
                overlay[ymin:ymin + lw, xmin:xmax, c] = (
                    overlay[ymin:ymin + lw, xmin:xmax, c] * (1 - self.style.alpha)
                    + color[c] * self.style.alpha
                )
                overlay[ymax - lw:ymax, xmin:xmax, c] = (
                    overlay[ymax - lw:ymax, xmin:xmax, c] * (1 - self.style.alpha)
                    + color[c] * self.style.alpha
                )

        if violation.region.mask_slice is not None:
            local_h, local_w = violation.region.mask_slice.shape
            slice_h = min(local_h, ymax - ymin)
            slice_w = min(local_w, xmax - xmin)
            for c in range(3):
                region = overlay[ymin:ymin + slice_h, xmin:xmin + slice_w, c]
                fill_mask = violation.region.mask_slice[:slice_h, :slice_w]
                region[fill_mask] = (
                    region[fill_mask] * (1 - self.style.alpha * 0.5)
                    + color[c] * self.style.alpha * 0.5
                )

        if self.style.show_centroid:
            cy, cx = violation.region.centroid
            cy_int = int(round(cy))
            cx_int = int(round(cx))
            ms = self.style.centroid_marker_size
            y0 = max(0, cy_int - ms)
            y1 = min(h, cy_int + ms + 1)
            x0 = max(0, cx_int - ms)
            x1 = min(w, cx_int + ms + 1)
            for c in range(3):
                overlay[y0:y1, x0:x1, c] = color[c]

    def generate_heatmap(self,
                         mask: np.ndarray,
                         violations: List[MRCViolation],
                         ) -> np.ndarray:
        """
        生成违规密度热力图

        Args:
            mask: 原始掩模 (H, W)
            violations: 违规列表

        Returns:
            热力图 (H, W)，值域 [0, 1]
        """
        h, w = mask.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float64)

        for v in violations:
            weight = self._severity_weight(v.severity)
            ymin, xmin, ymax, xmax = v.region.bbox
            ymin = max(0, ymin)
            xmin = max(0, xmin)
            ymax = min(h, ymax)
            xmax = min(w, xmax)

            if v.region.mask_slice is not None:
                local_h, local_w = v.region.mask_slice.shape
                slice_h = min(local_h, ymax - ymin)
                slice_w = min(local_w, xmax - xmin)
                heatmap[ymin:ymin + slice_h, xmin:xmin + slice_w] += (
                    weight * v.region.mask_slice[:slice_h, :slice_w].astype(np.float64)
                )
            else:
                heatmap[ymin:ymax, xmin:xmax] += weight

        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        return heatmap

    @staticmethod
    def _severity_weight(severity: MRCRuleSeverity) -> float:
        """严重级别权重"""
        weights = {
            MRCRuleSeverity.FATAL: 1.0,
            MRCRuleSeverity.ERROR: 0.7,
            MRCRuleSeverity.WARNING: 0.4,
            MRCRuleSeverity.INFO: 0.2,
        }
        return weights.get(severity, 0.5)

    def generate_legend(self) -> List[Dict[str, Any]]:
        """生成图例信息"""
        return [
            {
                "severity": MRCRuleSeverity.FATAL.value,
                "label": "致命 (FATAL)",
                "color": list(self.style.color_fatal),
            },
            {
                "severity": MRCRuleSeverity.ERROR.value,
                "label": "错误 (ERROR)",
                "color": list(self.style.color_error),
            },
            {
                "severity": MRCRuleSeverity.WARNING.value,
                "label": "警告 (WARNING)",
                "color": list(self.style.color_warning),
            },
            {
                "severity": MRCRuleSeverity.INFO.value,
                "label": "信息 (INFO)",
                "color": list(self.style.color_info),
            },
        ]

    def save_visualization(self,
                           mask: np.ndarray,
                           result: MRCCheckResult,
                           output_path: str,
                           include_heatmap: bool = True,
                           dpi: int = 150,
                           ) -> Path:
        """
        保存可视化结果为图片

        Args:
            mask: 原始掩模
            result: MRC 检查结果
            output_path: 输出文件路径
            include_heatmap: 是否包含热力图
            dpi: 图像分辨率

        Returns:
            保存的文件路径
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib import cm

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overlay = self.generate_overlay_mask(mask, result.violations)

        n_cols = 2 if include_heatmap else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        axes[0].imshow(overlay)
        axes[0].set_title("掩模 + 违规高亮")
        axes[0].set_xlabel("X (像素)")
        axes[0].set_ylabel("Y (像素)")

        legend_elements = [
            Patch(facecolor=c["color"], label=c["label"], alpha=self.style.alpha)
            for c in self.generate_legend()
        ]
        axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

        if include_heatmap:
            heatmap = self.generate_heatmap(mask, result.violations)
            im = axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("违规密度热力图")
            axes[1].set_xlabel("X (像素)")
            axes[1].set_ylabel("Y (像素)")
            plt.colorbar(im, ax=axes[1], label="违规密度")

        fig.suptitle(
            f"MRC 检查可视化 - "
            f"总违规: {result.total_violations}, "
            f"致命: {result.fatal_count}, "
            f"错误: {result.error_count}",
            fontsize=12,
        )
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"违规可视化已保存: {output_path}")
        return output_path
