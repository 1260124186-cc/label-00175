#!/usr/bin/env python3
"""
验证 Advisor API 路由集成到主服务
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api'))

print("=" * 60)
print("Advisor API 集成验证")
print("=" * 60)

all_passed = True


def test_case(name, condition, details=""):
    global all_passed
    status = "✓ PASS" if condition else "✗ FAIL"
    if not condition:
        all_passed = False
    print(f"  {status}: {name}")
    if details and not condition:
        print(f"         {details}")


print("\n【1/3】模块导入测试")
print("-" * 60)

try:
    from advisor.api import router as advisor_router
    test_case("advisor.api 模块导入成功", True)
    test_case(f"API 路由前缀: {advisor_router.prefix}",
              advisor_router.prefix == "/api/advisor",
              f"实际: {advisor_router.prefix}")
    test_case(f"路由标签: {advisor_router.tags}",
              "RET推荐引擎" in advisor_router.tags,
              f"实际: {advisor_router.tags}")
except Exception as e:
    test_case("advisor.api 模块导入成功", False, str(e))
    all_passed = False


print("\n【2/3】主服务集成测试")
print("-" * 60)

try:
    from api.main import app
    test_case("FastAPI 主应用导入成功", True)

    routes = [route.path for route in app.routes]
    advisor_routes = [r for r in routes if '/api/advisor' in r]

    test_case("Advisor 路由已注册",
              len(advisor_routes) >= 3,
              f"找到 {len(advisor_routes)} 条 advisor 路由: {advisor_routes}")

    expected_routes = [
        '/api/advisor/recommend',
        '/api/advisor/recommend-features',
        '/api/advisor/knowledge-base',
    ]
    for route in expected_routes:
        test_case(f"路由存在: {route}",
                  any(route in r for r in advisor_routes),
                  f"实际路由: {advisor_routes}")

    # 检查健康检查端点是否包含 ret_advisor 特性
    found_ret_advisor = False
    for route in app.routes:
        if route.path == '/api/health':
            found_ret_advisor = True
            break

    test_case("健康检查包含 ret_advisor 特性标记",
              'ret_advisor' in str(app.openapi()),
              "请检查 main.py 的健康检查响应")

except Exception as e:
    test_case("主服务集成测试", False, str(e))
    all_passed = False


print("\n【3/3】API Schema 验证")
print("-" * 60)

try:
    from api.schemas import (
        RETRecommendRequest,
        RETRecommendFromFeaturesRequest,
        RETRecommendResponse,
    )

    test_case("RETRecommendRequest 存在", True)
    test_case("RETRecommendFromFeaturesRequest 存在", True)
    test_case("RETRecommendResponse 存在", True)

    req = RETRecommendRequest()
    test_case("RETRecommendRequest 默认值正确",
              req.pattern_type == 'line_space',
              f"实际: {req.pattern_type}")
    test_case("RETRecommendRequest 包含 optical_system",
              hasattr(req, 'optical_system'),
              "缺少 optical_system 字段")
    test_case("RETRecommendRequest 包含 pattern_params",
              hasattr(req, 'pattern_params'),
              "缺少 pattern_params 字段")

    resp = RETRecommendResponse()
    test_case("RETRecommendResponse 默认 success=True",
              resp.success == True,
              f"实际: {resp.success}")

except Exception as e:
    test_case("API Schema 验证", False, str(e))
    all_passed = False


print("\n" + "=" * 60)
if all_passed:
    print("✓ 所有 API 集成测试通过！")
else:
    print("✗ 部分测试失败，请检查以上失败项。")
print("=" * 60)

sys.exit(0 if all_passed else 1)
