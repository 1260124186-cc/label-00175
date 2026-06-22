# 代理模型目录

将训练好的代理模型文件放在此目录下:

```
models/
├── model.onnx        # ONNX 格式模型 (推荐，配合 onnxruntime)
├── model.pt          # TorchScript 格式模型 (备选，配合 torch)
└── metadata.json     # 模型元数据 (推荐)
```

metadata.json 示例:
```json
{
  "model": {
    "type": "unet",
    "input_shape": [null, 1, 128, 128],
    "output_shape": [null, 1, 128, 128],
    "base_channels": 32,
    "num_levels": 4
  },
  "export": {
    "format": "onnx",
    "version": "1.0.0",
    "exported_at": "2026-01-01T00:00:00Z"
  },
  "optical_params": {
    "wavelength_nm": 193.0,
    "na": 1.35,
    "sigma": 0.75
  },
  "training": {
    "dataset_size": 100000,
    "mse": 0.000123,
    "ssim": 0.997
  }
}
```

如果不放置任何模型，服务会自动回退到 **Hopkins 轻量化近似模式**。
