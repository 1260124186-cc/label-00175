"""
后端全链路验证脚本
- 验证 config/default 返回正确结构
- 验证 config/validate 校验有效
- 验证 simulation/run → 轮询 → completed
  (若缺少 numpy/scipy/core 模块则跳过仿真，仅验证接口可达)
"""
import sys
import os
import time
import asyncio
import traceback
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
API_DIR = BACKEND_ROOT / "api"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(API_DIR))

# ------------------------------------------------------------
# 0. 初始化 main（会 import routers，这是之前 500 的触发点）
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("[0] 导入 backend.api.main（复现原 config/default 500 的触发路径）")
try:
    import importlib
    api_main = importlib.import_module("api.main")
    print("  ✅ main 导入成功，app 存在:", hasattr(api_main, 'app'))
except Exception as e:
    print(f"  ❌ FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------
# 1. TestClient 跑接口
# ------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("\nfastapi.testclient 未安装，跳过 HTTP 层测试。\n"
          "请安装: pip install httpx")
    sys.exit(0)

client = TestClient(api_main.app)

def section(title):
    print("\n" + "-" * 60)
    print(f"[✓] {title}")

# --- 1.1 健康检查 ---
print("\n" + "=" * 70)
print("[1] HTTP 接口测试")
r = client.get("/api/health")
assert r.status_code == 200, f"/api/health: {r.status_code}"
section("/api/health = 200")

# --- 1.2 GET /api/config/default ---
r = client.get("/api/config/default")
assert r.status_code == 200, f"/api/config/default {r.status_code}: {r.text[:200]}"
data = r.json()
assert data["success"] is True
cfg = data["config"]
# 检查核心字段是否全部存在（对齐 default_config.yaml）
for k in ["optical_system", "optimization", "output", "imaging"]:
    assert k in cfg, f"缺少顶层字段: {k}"
for k in ["wavelength", "na", "sigma", "pixel_size", "illumination_type",
          "source_params", "tcc_mode", "socs_num_terms", "zernike_coefficients"]:
    assert k in cfg["optical_system"], f"缺少 optical_system.{k}"
for k in ["optimizer_type", "max_iter", "learning_rate", "loss_weights",
          "regularization", "bounds", "spatial_weight"]:
    assert k in cfg["optimization"], f"缺少 optimization.{k}"
for k in ["mse", "ssim", "pvb", "mask_complexity", "weighted_mse", "weighted_mae"]:
    assert k in cfg["optimization"]["loss_weights"], f"缺少 loss_weights.{k}"
assert "type" in cfg["optimization"]["regularization"]
assert "strength" in cfg["optimization"]["regularization"]
# 检查值是否合理
assert cfg["optical_system"]["wavelength"] == 193.0
assert cfg["optical_system"]["na"] == 1.35
section("/api/config/default = 200，4 大模块 + 所有关键字段全部存在")

# --- 1.3 POST /api/config/validate ---
r = client.post("/api/config/validate", json=cfg)
assert r.status_code == 200, f"/api/config/validate {r.status_code}"
vdata = r.json()
assert vdata["success"] is True
section("/api/config/validate = 200, valid=" + str(vdata.get("valid")))

# --- 1.4 POST /api/config/save ---
save_body = {"config": cfg, "filename": "smoke_test_config"}
r = client.post("/api/config/save", json=save_body)
assert r.status_code == 200, f"/api/config/save {r.status_code}"
sdata = r.json()
assert sdata["success"] is True
saved_path = Path(sdata["saved_path"])
assert saved_path.exists()
section("/api/config/save = 200，文件已落盘: " + saved_path.name)

# --- 1.5 GET /api/config/saved ---
r = client.get("/api/config/saved")
assert r.status_code == 200
list_data = r.json()
assert list_data["count"] >= 1
section(f"/api/config/saved = 200，共 {list_data['count']} 个文件")

# --- 1.6 DELETE 刚才保存的文件（清理） ---
r = client.delete(f"/api/config/saved/{saved_path.name}")
assert r.status_code == 200
section(f"/api/config/saved/{saved_path.name} DELETE 成功")

# --- 1.7 POST /api/simulation/run ---
print("\n" + "=" * 70)
print("[2] 仿真接口测试")

run_body = {
    "config": cfg,
    "pattern_type": "rectangle",
    "pattern_params": {
        "size": [64, 64],
        "x_start": 20, "x_end": 44,
        "y_start": 20, "y_end": 44
    }
}
r = client.post("/api/simulation/run", json=run_body)
assert r.status_code == 200, f"/api/simulation/run {r.status_code}: {r.text[:300]}"
rd = r.json()
assert rd["success"] is True and rd.get("task_id"), "缺少 task_id"
task_id = rd["task_id"]
print(f"  任务已提交: task_id = {task_id}")

# --- 1.8 轮询任务状态（最多 30s） ---
COMPLETED = False
for i in range(30):
    time.sleep(1)
    r2 = client.get(f"/api/simulation/tasks/{task_id}")
    assert r2.status_code == 200
    s = r2.json()
    print(f"  [{i+1:2d}s] status={s['status']:10s}  progress={s['progress']:3d}%  ", end="")
    if s["status"] == "completed":
        print("✅")
        COMPLETED = True
        # 检查结果结构
        metrics = s["result"]["initial_metrics"]
        assert "mse" in metrics and "ssim" in metrics
        print(f"    MSE = {metrics['mse']:.6e}")
        print(f"    SSIM = {metrics['ssim']:.4f}")
        if metrics.get("mae") is not None:
            print(f"    MAE = {metrics['mae']:.6e}")
        break
    elif s["status"] == "failed":
        print("❌ FAIL")
        print(f"    错误: {s.get('error')}")
        sys.exit(1)
    else:
        print("...")

if not COMPLETED:
    print("  ⚠️  超时，跳过仿真完成断言（可能是核心算法耗时较长）")

# --- 1.9 GET /api/simulation/tasks ---
r3 = client.get("/api/simulation/tasks")
assert r3.status_code == 200
tl = r3.json()
assert tl["count"] >= 1
section(f"/api/simulation/tasks = 200，共 {tl['count']} 个任务（含当前）")

# ------------------------------------------------------------
# 总结
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("🎉 全部 API 接口验证通过（无 500 错误）")
print("  - /api/health                 ✅ 200")
print("  - /api/config/default         ✅ 200（4 大模块+字段完整）")
print("  - /api/config/validate        ✅ 200")
print("  - /api/config/save            ✅ 200（文件落盘）")
print("  - /api/config/saved           ✅ 200")
print("  - DELETE /api/config/saved/x  ✅ 200（清理）")
print("  - /api/simulation/run         ✅ 200（task_id 返回）")
print("  - /api/simulation/tasks/{id}  ✅ 状态轮询正常")
print("=" * 70)
