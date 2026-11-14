"""货币投入产出足迹分析的回归核对。

核对矩阵（均经整体 Leontief 求解，不使用截断结果）：
  1. 单行业 A=0.5、y=10、f=2 -> 产出 20、足迹 40，直接 20 + 上游 20；
  2. 单行业 A=1 奇异 -> 明确报错，绝不能仍报 40；A>1 非生产性分开报；
  3. 跨行业循环（含周期振荡谱半径）整体闭合；
  4. 零最终需求 -> 产出与足迹均为 0 而非报错或负数；
  5. 接近临界矩阵（A=0.999）给出稳定性诊断与放大效应；
  6. 行业拆分/合并：排放金额守恒，同构结构足迹不变，系数由 Z 与 x 重建而非平均；
  7. 不完整映射保留未覆盖需求，足迹不归零；
  8. 两版差异的需求/强度/结构三因素分解并列顺序敏感性，交互项单列；
  9. 需求扰动响应、缓存跨版本隔离、截断仅作诊断。
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.input_output_footprints_analysis import (
    Basis, FootprintError, InputOutputModel, ModelCache,
    NonProductiveSystemError, SingularSystemError,
    compare_reports, demand_response, footprint, reclassify,
    solve_system, truncated_footprint, validate_crosswalk,
    FACTOR_DEMAND, FACTOR_INTENSITY, FACTOR_STRUCTURE,
)

ROOT = Path(__file__).resolve().parents[1]


def make_bundle(a, f, demands, sectors=None, currency="CNY", price_year=2026,
                power_iterations=100000):
    """由 A、f 与最终需求反推平衡的 Z 与 x，生成 schema_version=1 资料。

    生产性系统用幂级数展开 x=(I-A)^-1 y 到机器精度，Z=A diag(x)，
    保证行平衡 x=Z1+y；非生产性/临界系统（rho>=1）幂级数不收敛，
    改用单位总产出直接构造 Z=A（x=1），用于检验诊断路径而非平衡路径。
    """
    n = len(a)
    sectors = sectors or ["s%d" % i for i in range(n)]
    uses = [d[0] for d in demands]
    y_cols = [list(d[1]) for d in demands]
    y_total = [sum(col[i] for col in y_cols) for i in range(n)]
    rho_check = [row[:] for row in a]
    # 低成本判断：列和最大值 >=1 时幂级数不可靠，走直接构造
    col_sum_max = max(sum(a[i][j] for i in range(n)) for j in range(n))
    if col_sum_max >= 1.0:
        x = [1.0] * n
        z = [[float(a[i][j]) for j in range(n)] for i in range(n)]
    else:
        x = y_total[:]
        acc = y_total[:]
        for _ in range(power_iterations):
            nxt = [sum(a[i][j] * acc[j] for j in range(n)) for i in range(n)]
            x = [x[i] + nxt[i] for i in range(n)]
            acc = nxt
            if sum(map(abs, nxt)) <= 1e-15 * max(1.0, sum(map(abs, x))):
                break
        z = [[a[i][j] * x[j] for j in range(n)] for i in range(n)]
    return {
        "schema_version": 1,
        "sectors": sectors,
        "transactions": z,
        "gross_output": x,
        "final_demands": [{"use": u, "values": list(v)}
                          for (u, v) in demands],
        "direct_intensity": list(f),
        "basis": {"currency": currency, "price_year": price_year},
        "crosswalk": [{"from": s, "to": s, "weight": 1.0} for s in sectors],
    }


def model_from(a, f, demands, **kwargs):
    return InputOutputModel.from_bundle(make_bundle(a, f, demands, **kwargs))


class SingleSectorTests(unittest.TestCase):
    def test_textbook_chain_a_half(self):
        # A=0.5, y=10, f=2 -> x=20，足迹 40，直接 20、上游 20
        m = model_from([[0.5]], [2.0], [("park", [10.0])])
        r = footprint(m)
        self.assertAlmostEqual(r.output_x[0], 20.0, places=10)
        self.assertAlmostEqual(r.grand_total, 40.0, places=10)
        self.assertAlmostEqual(r.total_by_use["park"], 40.0, places=10)
        self.assertAlmostEqual(r.direct_by_sector_use[0][0], 20.0, places=10)
        self.assertAlmostEqual(r.upstream_by_sector_use[0][0], 20.0, places=10)
        # 交叉分解能对上来源行业与用途
        self.assertAlmostEqual(r.emissions_by_use("park")["s0"], 40.0, places=10)
        # 残差与加总误差必须在机器精度量级
        self.assertLess(r.residual_1norm, 1e-9)
        self.assertLess(r.decomposition_error, 1e-12)
        self.assertTrue(r.diagnosis.productive)
        self.assertFalse(r.diagnosis.near_critical)
        # Leontief 逆 = 2
        self.assertAlmostEqual(r.leontief_inverse[0][0], 2.0, places=10)

    def test_singular_a_equals_one_must_not_return_40(self):
        m = model_from([[1.0]], [2.0], [("park", [10.0])])
        with self.assertRaises(SingularSystemError) as ctx:
            footprint(m)
        diag = ctx.exception.diagnosis
        self.assertFalse(diag.nonsingular)
        self.assertAlmostEqual(diag.spectral_radius, 1.0, places=8)
        self.assertIn("不可逆", diag.explanation)
        # 截断序列自身也不收敛：层数越大越偏离，证明不能拿截断冒充精确值
        t2 = truncated_footprint(m, 2)
        t20 = truncated_footprint(m, 20)
        self.assertIn("不是精确足迹", t20["truncated_note"])
        self.assertGreater(t20["footprint_approximation"],
                           t2["footprint_approximation"])

    def test_non_productive_a_above_one_is_separate_error(self):
        m = model_from([[1.1]], [2.0], [("park", [10.0])])
        with self.assertRaises(NonProductiveSystemError):
            footprint(m)
        with self.assertRaises(NonProductiveSystemError):
            solve_system(m)

    def test_zero_final_demand_gives_zero_footprint(self):
        m = model_from([[0.5]], [2.0], [("park", [0.0])])
        r = footprint(m)
        self.assertEqual(r.output_x, [0.0])
        self.assertEqual(r.grand_total, 0.0)
        self.assertEqual(r.total_by_use["park"], 0.0)
        self.assertLess(r.residual_1norm, 1e-12)

    def test_near_critical_a_0999_flags_instability(self):
        m = model_from([[0.999]], [2.0], [("park", [10.0])])
        r = footprint(m)
        # x = 10 / 0.001 = 10000，足迹 = 20000
        self.assertAlmostEqual(r.output_x[0], 10000.0, places=6)
        self.assertAlmostEqual(r.grand_total, 20000.0, places=6)
        self.assertTrue(r.diagnosis.near_critical)
        # 1x1 系统 1-范数条件数 = cond(I-A) = 1（M 与 M^-1 范数互抵），
        # 临界信号来自谱半径余量而非条件数
        self.assertLess(r.diagnosis.margin_to_unit, 0.05)
        self.assertIn("接近临界", r.diagnosis.explanation)
        # 多维近临界系统条件数显著放大，与诊断阈值呼应
        m2 = model_from([[0.99, 0.0], [0.5, 0.98]], [1.0, 1.0],
                        [("d", [1.0, 1.0])], sectors=["a", "b"])
        r2 = footprint(m2)
        self.assertTrue(r2.diagnosis.near_critical)
        self.assertGreater(r2.diagnosis.condition_number, 1e2)
        # 与稳定系统相比的放大倍数 = L(0.999)/L(0.5) = 1000/2 = 500
        m_stable = model_from([[0.5]], [2.0], [("park", [10.0])])
        ratio = r.grand_total / footprint(m_stable).grand_total
        self.assertAlmostEqual(ratio, 500.0, places=6)


class CircularDependencyTests(unittest.TestCase):
    def test_three_sector_circular_loop_closes_exactly(self):
        # 三个行业互相供货：0->1->2->0 循环，外加部分自供
        a = [
            [0.10, 0.00, 0.30],
            [0.20, 0.10, 0.00],
            [0.00, 0.25, 0.10],
        ]
        f = [1.5, 0.8, 2.2]
        m = model_from(a, f, [("household", [8.0, 5.0, 3.0]),
                              ("export", [2.0, 0.0, 4.0])],
                       sectors=["mining", "mfg", "power"])
        r = footprint(m)
        # 残差闭合：(I-A)x = y
        self.assertLess(r.residual_1norm, 1e-9)
        # 总排放 = f·x
        fx = sum(fi * xi for fi, xi in zip(f, r.output_x))
        self.assertAlmostEqual(r.grand_total, fx, places=9)
        # 用途合计 = 行业合计 = 总足迹
        use_sum = sum(r.total_by_use.values())
        sec_sum = sum(r.total_by_source_sector.values())
        self.assertAlmostEqual(use_sum, r.grand_total, places=9)
        self.assertAlmostEqual(sec_sum, r.grand_total, places=9)
        self.assertLess(r.decomposition_error, 1e-12)
        # 直接部分 f·y 与上游部分均为正且和等于总足迹
        direct = sum(sum(row) for row in r.direct_by_sector_use)
        upstream = sum(sum(row) for row in r.upstream_by_sector_use)
        self.assertAlmostEqual(direct + upstream, r.grand_total, places=9)
        self.assertGreater(upstream, 0.0)
        # 与截断序列对比：截断随层数单调逼近精确值，精确值是上确界
        t5 = truncated_footprint(m, 5)
        t50 = truncated_footprint(m, 50)
        self.assertLess(t5["footprint_approximation"], r.grand_total)
        self.assertAlmostEqual(t50["footprint_approximation"],
                               r.grand_total, places=6)

    def test_period_two_permutation_spectral_radius(self):
        # A = [[0,0.5],[0.5,0]] 是周期 2 振荡，朴素幂迭代会震荡，需后备收敛
        a = [[0.0, 0.5], [0.5, 0.0]]
        m = model_from(a, [1.0, 1.0], [("d", [1.0, 1.0])],
                       sectors=["a", "b"])
        r = footprint(m)
        self.assertTrue(r.diagnosis.spectral_radius_converged)
        self.assertAlmostEqual(r.diagnosis.spectral_radius, 0.5, places=8)
        # L = (I-A)^-1 = 1/0.75 * [[1,0.5],[0.5,1]]
        self.assertAlmostEqual(r.leontief_inverse[0][0], 1.0 / 0.75, places=9)
        self.assertAlmostEqual(r.leontief_inverse[0][1], 0.5 / 0.75, places=9)

    def test_zero_demand_in_circular_system(self):
        a = [[0.2, 0.3], [0.4, 0.1]]
        m = model_from(a, [1.0, 2.0], [("d", [0.0, 0.0])])
        r = footprint(m)
        self.assertEqual(r.output_x, [0.0, 0.0])
        self.assertEqual(r.grand_total, 0.0)
        self.assertTrue(all(v == 0.0 for row in r.impact_matrix for v in row))


class CrosswalkTests(unittest.TestCase):
    def _two_sector_model(self):
        a = [[0.30, 0.20], [0.10, 0.40]]
        return model_from(a, [2.0, 4.0], [("d", [10.0, 8.0])],
                          sectors=["chem", "steel"])

    def test_aggregation_conserves_emissions_and_rebuilds_coefficients(self):
        m = self._two_sector_model()
        base = footprint(m)
        # 合并为单一行业 heavy
        crosswalk = [
            {"from": "chem", "to": "heavy", "weight": 1.0},
            {"from": "steel", "to": "heavy", "weight": 1.0},
        ]
        m_agg = reclassify(m, ["heavy"], crosswalk)
        r_agg = footprint(m_agg)
        # 排放金额守恒
        self.assertAlmostEqual(r_agg.grand_total, base.grand_total, places=9)
        self.assertAlmostEqual(sum(m.f[i] * m.x[i] for i in range(2)),
                               m_agg.f[0] * m_agg.x[0], places=9)
        # 总产出与最终需求守恒
        self.assertAlmostEqual(m_agg.x[0], sum(m.x), places=9)
        self.assertAlmostEqual(m_agg.y_total()[0], m.y_total()[0]
                               + m.y_total()[1], places=9)
        # 系数确实由新 Z 与新 x 重建：A_new = Z_new / x_new
        self.assertAlmostEqual(m_agg.a[0][0],
                               m_agg.z[0][0] / m_agg.x[0], places=12)
        self.assertTrue(m_agg.provenance["reclassified"])

    def test_split_one_to_many_conserves_and_homogeneous_structure_preserved(self):
        # 单行业拆成两个同构子行业：A 各元素按份额折算，足迹应保持不变
        a = [[0.4]]
        m = model_from(a, [3.0], [("d", [100.0])], sectors=["all"])
        crosswalk = [
            {"from": "all", "to": "all_east", "weight": 0.6},
            {"from": "all", "to": "all_west", "weight": 0.4},
        ]
        m_split = reclassify(m, ["all_east", "all_west"], crosswalk)
        # 守恒校验
        self.assertAlmostEqual(sum(m_split.x), m.x[0], places=9)
        self.assertAlmostEqual(sum(m_split.y_total()), m.y_total()[0], places=9)
        self.assertAlmostEqual(sum(m_split.f[i] * m_split.x[i]
                                   for i in range(2)),
                               m.f[0] * m.x[0], places=9)
        # 同构拆分：子行业强度等于原强度
        self.assertAlmostEqual(m_split.f[0], 3.0, places=9)
        self.assertAlmostEqual(m_split.f[1], 3.0, places=9)
        # 同构结构下总足迹不变
        self.assertAlmostEqual(footprint(m_split).grand_total,
                               footprint(m).grand_total, places=8)

    def test_incomplete_mapping_retains_uncovered_demand(self):
        m = self._two_sector_model()
        base = footprint(m)
        # 只映射 chem，steel 未覆盖：应自动保留为独立行业，足迹不归零
        crosswalk = [{"from": "chem", "to": "chem2", "weight": 1.0}]
        m2 = reclassify(m, ["chem2"], crosswalk)
        self.assertIn("steel", m2.sectors)
        self.assertEqual(m2.provenance["uncovered_retained"], ["steel"])
        r2 = footprint(m2)
        self.assertAlmostEqual(r2.grand_total, base.grand_total, places=9)
        # steel 的最终需求被保留而不是丢弃
        k = m2.uses.index("d")
        self.assertEqual(m2.y_columns[k][m2.sectors.index("steel")],
                         m.y_columns[k][m.sectors.index("steel")])

    def test_non_conserving_weights_rejected(self):
        m = self._two_sector_model()
        bad = [
            {"from": "chem", "to": "heavy", "weight": 0.7},
            {"from": "steel", "to": "heavy", "weight": 1.0},
        ]
        with self.assertRaises(FootprintError):
            reclassify(m, ["heavy"], bad)

    def test_validate_crosswalk_unknown_sector(self):
        with self.assertRaises(FootprintError):
            validate_crosswalk(
                [{"from": "ghost", "to": "x", "weight": 1.0}],
                ("a",), ("x",))

    def test_roundtrip_split_then_merge(self):
        m = self._two_sector_model()
        base = footprint(m)
        split_map = [
            {"from": "chem", "to": "c1", "weight": 0.5},
            {"from": "chem", "to": "c2", "weight": 0.5},
            {"from": "steel", "to": "st", "weight": 1.0},
        ]
        m_split = reclassify(m, ["c1", "c2", "st"], split_map)
        merge_map = [
            {"from": "c1", "to": "chem", "weight": 1.0},
            {"from": "c2", "to": "chem", "weight": 1.0},
            {"from": "st", "to": "steel", "weight": 1.0},
        ]
        m_back = reclassify(m_split, ["chem", "steel"], merge_map)
        self.assertAlmostEqual(footprint(m_back).grand_total,
                               base.grand_total, places=8)


class BasisAndCacheTests(unittest.TestCase):
    def test_different_basis_cannot_share_cache_entry(self):
        b1 = model_from([[0.5]], [2.0], [("d", [10.0])],
                        currency="CNY", price_year=2020)
        b2 = model_from([[0.5]], [2.0], [("d", [10.0])],
                        currency="CNY", price_year=2026)
        cache = ModelCache()
        r1 = footprint(b1, cache=cache)
        r2 = footprint(b2, cache=cache)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.stats()["hits"], 0)
        # 同口径同输入第二次命中
        footprint(model_from([[0.5]], [2.0], [("d", [10.0])],
                             currency="CNY", price_year=2020), cache=cache)
        self.assertEqual(cache.stats()["hits"], 1)
        # 键中包含口径指纹
        self.assertNotEqual(ModelCache.key_for(b1), ModelCache.key_for(b2))
        self.assertNotEqual(r1.input_summary.input_fingerprint,
                            r2.input_summary.input_fingerprint)

    def test_structure_change_misses_cache(self):
        cache = ModelCache()
        m1 = model_from([[0.5]], [2.0], [("d", [10.0])])
        m2 = model_from([[0.6]], [2.0], [("d", [10.0])])
        footprint(m1, cache=cache)
        footprint(m2, cache=cache)
        self.assertEqual(len(cache), 2)

    def test_basis_requires_year_and_currency(self):
        with self.assertRaises(FootprintError):
            Basis.from_dict({"currency": "CNY"})


class ComparisonTests(unittest.TestCase):
    def test_three_factor_decomposition_with_order_sensitivity(self):
        sectors = ["s0", "s1"]
        base = model_from([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                          [("d", [10.0, 8.0])], sectors=sectors)
        # 另一版：需求、强度、结构同时变化
        alt = model_from([[0.35, 0.15], [0.20, 0.30]], [2.5, 3.0],
                         [("d", [12.0, 7.0])], sectors=sectors)
        cmp = compare_reports(base, alt)
        # 对称贡献 + 交互项严格还原总差异
        self.assertAlmostEqual(
            sum(cmp.shapley.values()) + cmp.interaction_term,
            cmp.total_delta, places=9)
        # 两条顺序各自都能加总到总差异
        self.assertAlmostEqual(sum(cmp.order_d_i_s.values()),
                               cmp.total_delta, places=9)
        self.assertAlmostEqual(sum(cmp.order_s_i_d.values()),
                               cmp.total_delta, places=9)
        self.assertLess(cmp.aggregation_error, 1e-10)
        # 存在交互时顺序极差非负，且解释文本提示不可单因素归因
        self.assertTrue(all(v >= 0.0 for v in cmp.order_spread.values()))

    def test_pure_demand_change_attributes_only_to_demand(self):
        base = model_from([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                          [("d", [10.0, 8.0])])
        alt = model_from([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                         [("d", [13.0, 11.0])])
        cmp = compare_reports(base, alt)
        self.assertAlmostEqual(cmp.shapley[FACTOR_INTENSITY], 0.0, places=9)
        self.assertAlmostEqual(cmp.shapley[FACTOR_STRUCTURE], 0.0, places=9)
        self.assertAlmostEqual(cmp.shapley[FACTOR_DEMAND],
                               cmp.total_delta, places=9)
        self.assertAlmostEqual(cmp.interaction_term, 0.0, places=9)
        self.assertEqual(cmp.order_spread[FACTOR_DEMAND], 0.0)

    def test_pure_intensity_change(self):
        base = model_from([[0.5]], [2.0], [("d", [10.0])])
        alt = model_from([[0.5]], [3.0], [("d", [10.0])])
        cmp = compare_reports(base, alt)
        # 仅强度变化：差异 = (3-2) * x = 20
        self.assertAlmostEqual(cmp.total_delta, 20.0, places=9)
        self.assertAlmostEqual(cmp.shapley[FACTOR_INTENSITY], 20.0, places=9)
        self.assertAlmostEqual(cmp.shapley[FACTOR_DEMAND], 0.0, places=9)

    def test_comparison_rejects_different_sector_systems(self):
        base = model_from([[0.5]], [2.0], [("d", [10.0])], sectors=["a"])
        alt = model_from([[0.5, 0.1], [0.0, 0.4]], [2.0, 1.0],
                         [("d", [10.0, 5.0])], sectors=["a", "b"])
        with self.assertRaises(FootprintError):
            compare_reports(base, alt)

    def test_reclassified_report_can_be_compared(self):
        # 先重分类到同一体系再比较，证明映射后比较路径可用
        base = model_from([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                          [("d", [10.0, 8.0])], sectors=["chem", "steel"])
        alt = model_from([[0.4, 0.1], [0.2, 0.3]], [2.5, 3.5],
                         [("d", [11.0, 9.0])], sectors=["chem", "steel"])
        crosswalk = [
            {"from": "chem", "to": "heavy", "weight": 1.0},
            {"from": "steel", "to": "heavy", "weight": 1.0},
        ]
        b2 = reclassify(base, ["heavy"], crosswalk)
        a2 = reclassify(alt, ["heavy"], crosswalk)
        cmp = compare_reports(b2, a2)
        self.assertAlmostEqual(
            sum(cmp.shapley.values()) + cmp.interaction_term,
            cmp.total_delta, places=9)


class CrossValidationTests(unittest.TestCase):
    """独立路径互证：不只证明一次求逆没抛异常。"""

    def test_two_independent_paths_agree_on_circular_system(self):
        a = [
            [0.10, 0.00, 0.30],
            [0.20, 0.10, 0.00],
            [0.00, 0.25, 0.10],
        ]
        m = model_from(a, [1.5, 0.8, 2.2],
                       [("h", [8.0, 5.0, 3.0]), ("e", [2.0, 0.0, 4.0])],
                       sectors=["mining", "mfg", "power"])
        r = footprint(m)
        cv = r.cross_validation
        # Gauss-Jordan 独立求逆与 LU 路径吻合
        self.assertLess(cv["gauss_jordan_inverse_max_diff"], 1e-10)
        self.assertLess(cv["gauss_jordan_output_max_diff"], 1e-10)
        # L*(I-A) = I 恒等式残差
        self.assertLess(cv["inverse_identity_residual"], 1e-10)
        # Neumann 幂级数极限与整体解吻合，证明循环被完整闭合
        self.assertTrue(cv["neumann_series_converged"])
        self.assertLess(cv["neumann_output_max_diff"], 1e-9)
        self.assertLess(cv["linear_residual_max_abs"], 1e-10)

    def test_large_near_critical_result_is_corroborated(self):
        # A=0.999 产出放大 1000 倍，大结果必须被两条独立路径共同支持
        m = model_from([[0.999]], [2.0], [("park", [10.0])])
        r = footprint(m)
        self.assertGreater(r.grand_total, 19999.0)
        cv = r.cross_validation
        self.assertTrue(cv["neumann_series_converged"])
        # 即便条件接近临界，两条独立数值路径仍在机器精度内一致
        self.assertLess(cv["gauss_jordan_output_max_diff"], 1e-8)
        self.assertLess(cv["neumann_output_max_diff"], 1e-6)
        self.assertLess(cv["linear_residual_max_abs"], 1e-8)
        # 解析真值 10000 也对上
        self.assertAlmostEqual(r.output_x[0], 10000.0, places=6)

    def test_near_critical_multidim_corroboration(self):
        a = [[0.99, 0.0], [0.5, 0.98]]
        m = model_from(a, [1.0, 1.0], [("d", [1.0, 1.0])],
                       sectors=["a", "b"])
        r = footprint(m)
        cv = r.cross_validation
        self.assertTrue(r.diagnosis.near_critical)
        self.assertTrue(cv["neumann_series_converged"])
        self.assertLess(cv["gauss_jordan_inverse_max_diff"], 1e-8)
        self.assertLess(cv["neumann_output_max_diff"], 1e-6)

    def test_identity_basis_chain_totals_reconcile(self):
        # 用户指定核对：直接满足最终需求部分与其余上游部分对得上
        m = model_from([[0.5]], [2.0], [("park", [10.0])])
        r = footprint(m)
        direct = sum(sum(row) for row in r.direct_by_sector_use)
        upstream = sum(sum(row) for row in r.upstream_by_sector_use)
        self.assertAlmostEqual(direct, 20.0, places=10)
        self.assertAlmostEqual(upstream, 20.0, places=10)
        self.assertAlmostEqual(direct + upstream, r.grand_total, places=12)


class PerturbationTests(unittest.TestCase):
    def test_delta_y_response_matches_full_resolve(self):
        m = model_from([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                       [("d", [10.0, 8.0])])
        resp = demand_response(m, delta_y=[1.0, 0.0])
        # 独立完整求解验证：新需求下重算
        bundle_alt = make_bundle([[0.3, 0.2], [0.1, 0.4]], [2.0, 4.0],
                                 [("d", [11.0, 8.0])])
        r_alt = footprint(InputOutputModel.from_bundle(bundle_alt))
        self.assertAlmostEqual(resp["perturbed_footprint"],
                               r_alt.grand_total, places=9)
        # dx = L delta_y
        base = footprint(m)
        expected_dx = [sum(base.leontief_inverse[i][j] * (1 if j == 0 else 0)
                           for j in range(2)) for i in range(2)]
        for got, want in zip(resp["delta_output"], expected_dx):
            self.assertAlmostEqual(got, want, places=10)
        # 来源行业增量合计 = f·dx
        self.assertAlmostEqual(
            sum(resp["delta_emissions_by_source_sector"].values()),
            2.0 * expected_dx[0] + 4.0 * expected_dx[1], places=9)

    def test_remove_use_perturbation(self):
        m = model_from([[0.5]], [2.0], [("d", [10.0]), ("e", [5.0])])
        resp = demand_response(m, use="e")
        self.assertEqual(resp["delta_y"], [-5.0])
        self.assertAlmostEqual(resp["perturbed_footprint"],
                               footprint(m).total_by_use["d"], places=9)
        self.assertIn("可行", resp["feasibility"])

    def test_infeasible_perturbation_flagged(self):
        m = model_from([[0.5]], [2.0], [("d", [10.0])])
        resp = demand_response(m, delta_y=[-50.0])
        self.assertIn("不可行", resp["feasibility"])


class InputValidationTests(unittest.TestCase):
    def test_negative_transactions_rejected(self):
        b = make_bundle([[0.5]], [2.0], [("d", [10.0])])
        b["transactions"][0][0] = -1.0
        with self.assertRaises(FootprintError):
            InputOutputModel.from_bundle(b)

    def test_negative_intensity_rejected(self):
        b = make_bundle([[0.5]], [2.0], [("d", [10.0])])
        b["direct_intensity"][0] = -0.1
        with self.assertRaises(FootprintError):
            InputOutputModel.from_bundle(b)

    def test_zero_gross_output_allowed_when_idle_but_inconsistent_rejected(self):
        # 全零需求的空闲行业：x=0、Z=0、f=0 自洽，应允许并给出零足迹
        b = make_bundle([[0.0]], [0.0], [("d", [0.0])])
        b["gross_output"][0] = 0.0
        m = InputOutputModel.from_bundle(b)
        self.assertEqual(footprint(m).grand_total, 0.0)
        # 零产出却有中间投入 -> 不自洽，拒绝
        b["transactions"][0][0] = 1.0
        with self.assertRaises(FootprintError):
            InputOutputModel.from_bundle(b)
        # 零产出但强度非零不再拒绝：行业不运行时 f*x=0，给出口径提示
        b["transactions"][0][0] = 0.0
        b["direct_intensity"][0] = 2.0
        m2 = InputOutputModel.from_bundle(b)
        self.assertTrue(any("未被激活" in w for w in m2.balance_warnings))
        self.assertEqual(footprint(m2).grand_total, 0.0)

    def test_duplicate_sector_rejected(self):
        b = make_bundle([[0.5, 0.0], [0.0, 0.5]], [1.0, 1.0],
                        [("d", [1.0, 1.0])])
        b["sectors"] = ["x", "x"]
        with self.assertRaises(FootprintError):
            InputOutputModel.from_bundle(b)

    def test_row_imbalance_warns(self):
        b = make_bundle([[0.5]], [2.0], [("d", [10.0])])
        b["final_demands"][0]["values"][0] = 9.0  # 破坏行平衡
        m = InputOutputModel.from_bundle(b)
        self.assertTrue(m.balance_warnings)
        r = footprint(m)
        self.assertTrue(r.balance_warnings)

    def test_non_finite_rejected_at_model_layer(self):
        b = make_bundle([[0.5]], [2.0], [("d", [10.0])])
        b["direct_intensity"][0] = float("inf")
        with self.assertRaises(FootprintError):
            InputOutputModel.from_bundle(b)


class CliTests(unittest.TestCase):
    def test_cli_on_fixture(self):
        proc = subprocess.run(
            [sys.executable, "-m", "src.input_output_footprints_analysis",
             str(ROOT / "fixtures" / "sample.json")],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        # sample.json: A=10/20=0.5, y=10, f=2 -> 40
        self.assertAlmostEqual(payload["grand_total"], 40.0, places=9)
        self.assertLess(payload["residual_1norm"], 1e-9)
        self.assertIn("input_fingerprint", payload["input_summary"])

    def test_cli_singular_returns_distinct_code(self):
        bundle = make_bundle([[1.0]], [2.0], [("d", [10.0])])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "singular.json"
            p.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "src.input_output_footprints_analysis",
                 str(p)], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("singular", proc.stdout)


if __name__ == "__main__":
    unittest.main()
