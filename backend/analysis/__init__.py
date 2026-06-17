# -*- coding: utf-8 -*-
try:
    from analysis.process_window import (
        ProcessWindowAnalyzer,
        PWMetrics,
        PrintabilityResult,
    )
except ImportError:
    from .process_window import (
        ProcessWindowAnalyzer,
        PWMetrics,
        PrintabilityResult,
    )
