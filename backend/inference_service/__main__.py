# -*- coding: utf-8 -*-
"""
python -m inference_service 入口

等价于: python -m inference_service.start_service
"""

from .start_service import main

if __name__ == "__main__":
    main()
