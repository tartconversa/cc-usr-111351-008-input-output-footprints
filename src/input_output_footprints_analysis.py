"""货币投入产出（Leontief）足迹分析。

口径约定
========
* ``A[i, j]`` 为行业 j 每产出一元所需行业 i 的投入金额（供给行业为行、使用行业为列）。
* 交易矩阵 ``Z`` 与总产出 ``x`` 同价格基期、同币种，由 ``A = Z / x`` 重建系数，
  不接受对不同版本的系数矩阵做平均。
* 线性系统为 ``x = A x + y``，即 ``(I - A) x = y``；循环依赖通过整体求解
  Leontief 逆 ``L = (I - A)^{-1}`` 处理，不做任何逐层截断。
* 直接排放强度 ``f`` 的分母为同价格基期一元产出；行业 i 排放为 ``f_i * x_i``。
* 需求用途 k 承担的足迹：``M[:, k] = diag(f) @ L @ y_k``，
  其中 ``f * y_k`` 是直接满足最终需求的部分，其余为上游间接部分。

本模块不模拟生产设备或物理反应，只做货币口径下的线性核算。
不同价格基期、币种或行业分类不能仅按列名拼接，须先经 ``reclassify``
（带覆盖与守恒校验的分类映射）处理；不完整映射保留未覆盖需求，不把其足迹归零。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import permutations

# ---------------------------------------------------------------------------
# 常量与异常
# ---------------------------------------------------------------------------

EPS = math.ulp(1.0)
BALANCE_TOL = 1e-9          # 行平衡（Z 行和 + 最终需求行和 == 总产出）相对容差
CROSSWALK_TOL = 1e-12       # 分类映射权重守恒容差
NEAR_CRITICAL_MARGIN = 0.05  # 谱半径距 1 的警戒距离
NEAR_SINGULAR_COND = 1e12   # 条件数超过该值视为数值上接近奇异

FACTOR_DEMAND = "demand"
FACTOR_INTENSITY = "intensity"
FACTOR_STRUCTURE = "structure"
FACTORS = (FACTOR_DEMAND, FACTOR_INTENSITY, FACTOR_STRUCTURE)


class FootprintError(ValueError):
    """足迹分析输入或口径错误。"""


class SingularSystemError(FootprintError):
    """``I - A`` 不可逆（如单行业 A=1），精确足迹不存在。"""


class NonProductiveSystemError(FootprintError):
    """系统非生产性（谱半径 >= 1 且非精确奇异），Leontief 逆非负解不存在。"""


# ---------------------------------------------------------------------------
# 纯 Python 线性代数（无第三方依赖）
# ---------------------------------------------------------------------------

def _shape(matrix):
    if not isinstance(matrix, list) or not matrix or not all(isinstance(r, list) for r in matrix):
        raise FootprintError("矩阵必须为列表的列表")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise FootprintError("矩阵必须为方阵")
    if n == 0:
        raise FootprintError("矩阵阶数必须为正")
    return n


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FootprintError(label + "必须为有限数值")
    value = float(value)
    if not math.isfinite(value):
        raise FootprintError(label + "包含非有限数")
    return value


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_copy(matrix):
    return [row[:] for row in matrix]


def matmul(a, b):
    n = len(a)
    m = len(b[0])
    k = len(b)
    if len(a[0]) != k:
        raise FootprintError("矩阵乘法维度不匹配")
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(m):
                oi[j] += ait * bt[j]
    return out


def matvec(a, v):
    n = len(a)
    out = [0.0] * n
    for i in range(n):
        total = 0.0
        row = a[i]
        for j in range(len(v)):
            total += row[j] * v[j]
        out[i] = total
    return out


def transpose(a):
    rows = len(a)
    cols = len(a[0])
    return [[a[j][i] for j in range(rows)] for i in range(cols)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_abs_sum(v):
    return sum(abs(x) for x in v)


def mat_abs(a):
    return [[abs(x) for x in row] for row in a]


def lu_decompose(matrix):
    """带部分主元的 LU 分解，返回 (lu, piv)；奇异时抛 SingularSystemError。"""
    n = _shape(matrix)
    lu = mat_copy(matrix)
    piv = list(range(n))
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(lu[r][k]))
        pivot_val = abs(lu[pivot_row][k])
        scale = max(abs(lu[i][j]) for i in range(n) for j in range(n))
        if pivot_val <= EPS * max(1.0, scale):
            raise SingularSystemError("主元接近零，I - A 奇异或条件极差")
        if pivot_row != k:
            lu[k], lu[pivot_row] = lu[pivot_row], lu[k]
            piv[k], piv[pivot_row] = piv[pivot_row], piv[k]
        pivot = lu[k][k]
        for i in range(k + 1, n):
            factor = lu[i][k] / pivot
            lu[i][k] = factor
            row_i = lu[i]
            row_k = lu[k]
            for j in range(k + 1, n):
                row_i[j] -= factor * row_k[j]
    return lu, piv


def lu_solve(lu, piv, rhs):
    """用 ``lu_decompose`` 的结果解线性方程组。"""
    n = len(lu)
    b = [float(rhs[piv[i]]) for i in range(n)]
    # 前向代入 L y = P b（L 单位下三角，乘数存于 lu 下三角）
    for i in range(n):
        total = b[i]
        row = lu[i]
        for j in range(i):
            total -= row[j] * b[j]
        b[i] = total
    # 反向代入 U x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        row = lu[i]
        for j in range(i + 1, n):
            total -= row[j] * x[j]
        diag = row[i]
        if abs(diag) <= EPS:
            raise SingularSystemError("回代主元为零")
        x[i] = total / diag
    return x


def lu_determinant(lu, piv):
    det = 1.0
    swapped = 0
    prev = list(range(len(piv)))
    for k, target in enumerate(piv):
        if prev[k] != target:
            swapped += 1
            idx = prev.index(target)
            prev[k], prev[idx] = prev[idx], prev[k]
    for i in range(len(lu)):
        det *= lu[i][i]
    return -det if swapped % 2 else det


def gauss_jordan_inverse(matrix):
    """Gauss–Jordan 求逆，作为 LU 路径的独立校验/后备。"""
    n = _shape(matrix)
    aug = [list(map(float, matrix[i])) + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(aug[r][k]))
        if abs(aug[pivot_row][k]) <= EPS:
            raise SingularSystemError("Gauss-Jordan 主元为零，矩阵不可逆")
        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
        pivot = aug[k][k]
        aug[k] = [v / pivot for v in aug[k]]
        for i in range(n):
            if i == k:
                continue
            factor = aug[i][k]
            if factor == 0.0:
                continue
            aug[i] = [aug[i][j] - factor * aug[k][j] for j in range(2 * n)]
    return [row[n:] for row in aug]


def inverse(matrix):
    """矩阵求逆：先 LU 逐列求解，失败再用 Gauss–Jordan 复核。"""
    n = _shape(matrix)
    try:
        lu, piv = lu_decompose(matrix)
        inv = [lu_solve(lu, piv, [1.0 if j == k else 0.0 for j in range(n)])
               for k in range(n)]
        # lu_solve 返回的是各列，需要转置成行
        inv = transpose(inv)
    except SingularSystemError:
        inv = gauss_jordan_inverse(matrix)
    # 独立残差校验
    check = matmul(matrix, inv)
    err = max(abs(check[i][j] - (1.0 if i == j else 0.0))
              for i in range(n) for j in range(n))
    scale = max(1.0, max(abs(v) for row in matrix for v in row))
    if err > 1e-9 * scale:
        # LU 与 Gauss-Jordan 结论不一致时以后者为准再验一次
        inv2 = gauss_jordan_inverse(matrix)
        check2 = matmul(matrix, inv2)
        err2 = max(abs(check2[i][j] - (1.0 if i == j else 0.0))
                   for i in range(n) for j in range(n))
        if err2 >= err:
            raise SingularSystemError("逆矩阵残差过大，系统数值奇异")
        return inv2
    return inv


def condition_number_1norm(matrix, inv):
    """1-范数条件数 ``||M||_1 * ||M^-1||_1``。"""
    norm_m = max(sum(abs(matrix[i][j]) for i in range(len(matrix)))
                 for j in range(len(matrix)))
    norm_inv = max(sum(abs(inv[i][j]) for i in range(len(inv)))
                   for j in range(len(inv)))
    return norm_m * norm_inv


def spectral_radius_nonneg(a, max_iter=20000, tol=1e-12):
    """非负矩阵谱半径的幂迭代估计，返回 (rho, 收敛标志, 迭代次数)。

    对周期振荡（如二维置换矩阵）额外用 A^2 检测，避免误判不收敛。
    """
    n = _shape(a)
    v = [1.0 / n] * n
    rho_old = 0.0
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        w = matvec(a, v)
        norm = vec_abs_sum(w)
        if norm == 0.0:
            return 0.0, True, iterations
        v = [x / norm for x in w]
        if abs(norm - rho_old) <= tol * max(1.0, abs(norm)):
            converged = True
            break
        rho_old = norm
    rho = norm
    if not converged:
        # 周期图后备：A^2 的谱半径开平方
        a2 = matmul(a, a)
        rho2, conv2, _ = spectral_radius_nonneg(a2, max_iter=max_iter, tol=tol)
        if conv2:
            return math.sqrt(max(0.0, rho2)), True, iterations
    return rho, converged, iterations


def hawkins_simon_minors(matrix):
    """Hawkins–Simon 条件：I-A 的全部顺序主子式为正。返回 (是否成立, 各阶行列式)。"""
    n = _shape(matrix)
    determinants = []
    ok = True
    for k in range(1, n + 1):
        sub = [row[:k] for row in matrix[:k]]
        try:
            lu, piv = lu_decompose(sub)
            det = lu_determinant(lu, piv)
        except SingularSystemError:
            det = 0.0
        determinants.append(det)
        if det <= 0.0:
            ok = False
    return ok, determinants


# ---------------------------------------------------------------------------
# 口径与诊断数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Basis:
    """价格口径：币种与价格基期。口径不同的资料禁止直接拼接。"""
    currency: str
    price_year: int

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "currency" not in data or "price_year" not in data:
            raise FootprintError("basis 必须包含 currency 与 price_year")
        if isinstance(data["price_year"], bool) or not isinstance(data["price_year"], int):
            raise FootprintError("price_year 必须为整数")
        return cls(str(data["currency"]), data["price_year"])

    def key(self):
        return (self.currency, self.price_year)

    def as_dict(self):
        return {"currency": self.currency, "price_year": self.price_year}


@dataclass
class StabilityDiagnosis:
    """线性系统稳定性诊断。诊断与足迹结果一同返回，不允许只报数不报稳定性。"""
    nonsingular: bool
    productive: bool
    spectral_radius: float
    spectral_radius_converged: bool
    margin_to_unit: float
    condition_number: float | None
    hawkins_simon_ok: bool
    hawkins_simon_determinants: list[float]
    residual_1norm: float
    near_critical: bool
    explanation: str


@dataclass
class InputSummary:
    """每次结果保留的输入摘要（不含完整大矩阵，只留可复核的指纹与规模）。"""
    n_sectors: int
    sectors: tuple[str, ...]
    uses: tuple[str, ...]
    basis: dict
    input_fingerprint: str
    column_sum_max: float
    final_demand_total: float
    gross_output_total: float
    direct_emissions_at_demand: float


@dataclass
class FootprintResult:
    """一次完整足迹核算的结果与全部质量诊断。"""
    input_summary: InputSummary
    output_x: list[float]
    leontief_inverse: list[list[float]]
    impact_matrix: list[list[float]]          # M[i, k]：用途 k 由来源行业 i 承担的排放
    direct_by_sector_use: list[list[float]]    # diag(f) y_k
    upstream_by_sector_use: list[list[float]]  # diag(f) (L - I) y_k
    total_by_use: dict[str, float]
    total_by_source_sector: dict[str, float]
    grand_total: float
    residual: list[float]                      # (I-A)x - y_合计
    residual_1norm: float
    decomposition_error: float                 # 交叉分解加总与总足迹的相对误差
    diagnosis: StabilityDiagnosis
    cross_validation: dict
    balance_warnings: list[str] = field(default_factory=list)
    truncated_note: str | None = None          # 截断近似结果时强制标注

    def emissions_by_use(self, use):
        """按需求用途查询：返回来源行业 -> 排放的交叉分解。"""
        if use not in self.input_summary.uses:
            raise FootprintError("未知需求用途：" + str(use))
        k = self.input_summary.uses.index(use)
        sectors = self.input_summary.sectors
        return {sectors[i]: self.impact_matrix[i][k] for i in range(len(sectors))}

    def as_dict(self):
        return {
            "input_summary": {
                "n_sectors": self.input_summary.n_sectors,
                "sectors": list(self.input_summary.sectors),
                "uses": list(self.input_summary.uses),
                "basis": self.input_summary.basis,
                "input_fingerprint": self.input_summary.input_fingerprint,
                "column_sum_max": self.input_summary.column_sum_max,
                "final_demand_total": self.input_summary.final_demand_total,
                "gross_output_total": self.input_summary.gross_output_total,
                "direct_emissions_at_demand": self.input_summary.direct_emissions_at_demand,
            },
            "output_x": self.output_x,
            "total_by_use": self.total_by_use,
            "total_by_source_sector": self.total_by_source_sector,
            "grand_total": self.grand_total,
            "impact_matrix": self.impact_matrix,
            "residual_1norm": self.residual_1norm,
            "decomposition_error": self.decomposition_error,
            "balance_warnings": self.balance_warnings,
            "truncated_note": self.truncated_note,
            "cross_validation": self.cross_validation,
            "diagnosis": {
                "nonsingular": self.diagnosis.nonsingular,
                "productive": self.diagnosis.productive,
                "spectral_radius": self.diagnosis.spectral_radius,
                "spectral_radius_converged": self.diagnosis.spectral_radius_converged,
                "margin_to_unit": self.diagnosis.margin_to_unit,
                "condition_number": self.diagnosis.condition_number,
                "hawkins_simon_ok": self.diagnosis.hawkins_simon_ok,
                "hawkins_simon_determinants": self.diagnosis.hawkins_simon_determinants,
                "near_critical": self.diagnosis.near_critical,
                "explanation": self.diagnosis.explanation,
            },
        }


# ---------------------------------------------------------------------------
# 模型构建与校验
# ---------------------------------------------------------------------------

class InputOutputModel:
    """校验后的投入产出模型；系数只由本版 Z 与 x 重建。"""

    def __init__(self, sectors, transactions, gross_output, final_demands,
                 direct_intensity, basis, crosswalk=None, *,
                 balance_tol=BALANCE_TOL, provenance=None):
        self.basis = basis if isinstance(basis, Basis) else Basis.from_dict(basis)
        self.sectors = self._check_sectors(sectors)
        n = len(self.sectors)
        self.z = self._check_matrix(transactions, n, "transactions")
        self.x = [_finite_number(v, "gross_output[%d]" % i) for i, v in enumerate(gross_output)]
        if len(self.x) != n:
            raise FootprintError("gross_output 长度与行业数不一致")
        self.f = [_finite_number(v, "direct_intensity[%d]" % i)
                  for i, v in enumerate(direct_intensity)]
        if len(self.f) != n:
            raise FootprintError("direct_intensity 长度与行业数不一致")
        if any(v < 0.0 for v in self.z for v in v):
            raise FootprintError("交易额含负数，货币交易矩阵必须非负")
        if any(v < 0.0 for v in self.f):
            raise FootprintError("直接排放强度必须非负")
        if any(v < 0.0 for v in self.x):
            raise FootprintError("总产出必须非负")
        zero_rows = [self.sectors[i] for i, v in enumerate(self.x) if v < 0.0]
        if zero_rows:
            raise FootprintError("总产出为负的行业无法定义一元产出系数："
                                 + ", ".join(zero_rows))
        self.uses, self.y_columns = self._check_demands(final_demands, n)
        self.crosswalk = crosswalk
        self.provenance = provenance or {
            "coefficients_rebuilt_from": "本版 transactions(Z) 与 gross_output(x)，"
                                         "A = Z 逐列除以 x，未对任何系数矩阵做平均",
            "reclassified": False,
        }
        self.balance_tol = balance_tol
        self.balance_warnings = self._check_balance()
        # A = Z / x：列 j 除以 x_j。零产出行业没有一元产出口径，
        # 其系数列无定义；经济上该行业不运行，Z 对应列必为零，按零列处理。
        self.a = [[(self.z[i][j] / self.x[j] if self.x[j] > 0.0 else 0.0)
                   for j in range(n)] for i in range(n)]
        for j, name in enumerate(self.sectors):
            if self.x[j] == 0.0:
                if any(self.z[i][j] != 0.0 for i in range(n)):
                    raise FootprintError(
                        "行业 %s 总产出为零却存在中间投入交易额，资料不自洽" % name)
                # 强度可保留正值：行业不运行时 f*x=0 排放自然为零，
                # 仅提示该强度在当前基期不被任何产出激活。
                if self.f[j] != 0.0:
                    self.balance_warnings.append(
                        "行业 %s 总产出为零：直接强度 %.10g 未被激活，排放为零"
                        % (name, self.f[j]))
        self.column_sum_max = max(sum(self.a[i][j] for i in range(n))
                                  for j in range(n)) if n else 0.0
        self._fingerprint = self._compute_fingerprint()

    @staticmethod
    def _check_sectors(sectors):
        if not isinstance(sectors, list) or not sectors:
            raise FootprintError("sectors 必须为非空列表")
        if any(not isinstance(s, str) or not s for s in sectors):
            raise FootprintError("行业名称必须为非空字符串")
        if len(set(sectors)) != len(sectors):
            raise FootprintError("行业名称重复")
        return tuple(sectors)

    @staticmethod
    def _check_matrix(matrix, n, label):
        if not isinstance(matrix, list) or len(matrix) != n:
            raise FootprintError(label + " 维度必须与行业数一致")
        out = []
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n:
                raise FootprintError(label + " 必须为 %d 阶方阵" % n)
            out.append([_finite_number(v, "%s[%d]" % (label, i)) for v in row])
        return out

    @staticmethod
    def _check_demands(final_demands, n):
        if not isinstance(final_demands, list) or not final_demands:
            raise FootprintError("final_demands 必须为非空列表")
        uses = []
        columns = []
        for entry in final_demands:
            if not isinstance(entry, dict) or "use" not in entry or "values" not in entry:
                raise FootprintError("每个最终需求条目需包含 use 与 values")
            use = entry["use"]
            if not isinstance(use, str) or not use:
                raise FootprintError("需求用途名称必须为非空字符串")
            if use in uses:
                raise FootprintError("需求用途重复：" + use)
            values = entry["values"]
            if not isinstance(values, list) or len(values) != n:
                raise FootprintError("用途 %s 的 values 维度与行业数不一致" % use)
            col = [_finite_number(v, "final_demand[%s]" % use) for v in values]
            if any(v < 0.0 for v in col):
                raise FootprintError("最终需求必须非负：" + use)
            uses.append(use)
            columns.append(col)
        return tuple(uses), columns

    def _check_balance(self):
        warnings = []
        n = len(self.sectors)
        y_sum = [sum(self.y_columns[k][i] for k in range(len(self.uses)))
                 for i in range(n)]
        for i in range(n):
            z_row = sum(self.z[i][j] for j in range(n))
            scale = max(1.0, abs(self.x[i]), abs(z_row), abs(y_sum[i]))
            gap = self.x[i] - z_row - y_sum[i]
            if abs(gap) > self.balance_tol * scale:
                warnings.append(
                    "行业 %s 行不平衡：总产出 %.10g - 中间使用 %.10g - 最终需求 %.10g = %.10g"
                    % (self.sectors[i], self.x[i], z_row, y_sum[i], gap))
        return warnings

    def _compute_fingerprint(self):
        payload = {
            "basis": self.basis.key(),
            "sectors": list(self.sectors),
            "uses": list(self.uses),
            "A": self.a,
            "f": self.f,
            "Y": self.y_columns,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def fingerprint(self):
        """绑定行业分类、价格口径、系数、强度与需求的完整指纹。"""
        return self._fingerprint

    def y_total(self):
        n = len(self.sectors)
        return [sum(self.y_columns[k][i] for k in range(len(self.uses)))
                for i in range(n)]

    @classmethod
    def from_bundle(cls, bundle, **kwargs):
        """从 ``load_bundle`` 读出的资料字典构建模型。"""
        return cls(
            sectors=list(bundle["sectors"]),
            transactions=bundle["transactions"],
            gross_output=bundle["gross_output"],
            final_demands=bundle["final_demands"],
            direct_intensity=bundle["direct_intensity"],
            basis=Basis.from_dict(bundle["basis"]),
            crosswalk=bundle.get("crosswalk"),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 稳定性诊断与整体求解
# ---------------------------------------------------------------------------

def cross_validate_solution(model, x_solved, leontief_inv, *,
                            series_max_iter=200000, series_tol=1e-13):
    """用两条独立路径复核 LU 解，避免"一次求逆没抛异常就算对"。

    1. Gauss–Jordan 独立求逆（与 LU 不同的消元路径）重算 L 与 x；
    2. Neumann 幂级数 ``x = (I + A + A^2 + ...) y`` 迭代到收敛，
       生产性系统必收敛到同一极限，用于核对循环是否被完整闭合；
    返回各路径的最大偏差与收敛信息，供结果对象与测试断言。
    """
    n = len(model.a)
    m = [[(1.0 if i == j else 0.0) - model.a[i][j] for j in range(n)]
         for i in range(n)]
    y = model.y_total()
    # 路径一：Gauss-Jordan 独立求逆
    gj_inv = gauss_jordan_inverse(m)
    gj_x = matvec(gj_inv, y)
    inv_diff = max(abs(gj_inv[i][j] - leontief_inv[i][j])
                   for i in range(n) for j in range(n))
    x_diff_gj = max(abs(gj_x[i] - x_solved[i]) for i in range(n))
    identity_check = matmul(m, leontief_inv)
    inverse_residual = max(
        abs(identity_check[i][j] - (1.0 if i == j else 0.0))
        for i in range(n) for j in range(n))
    # 路径二：Neumann 幂级数极限（仅生产性系统可收敛）
    x_series = y[:]
    term = y[:]
    iterations = 0
    converged = False
    for iterations in range(1, series_max_iter + 1):
        term = matvec(model.a, term)
        x_series = [x_series[i] + term[i] for i in range(n)]
        if vec_abs_sum(term) <= series_tol * max(1.0, vec_abs_sum(x_series)):
            converged = True
            break
    x_diff_series = max(abs(x_series[i] - x_solved[i]) for i in range(n))
    return {
        "gauss_jordan_inverse_max_diff": inv_diff,
        "gauss_jordan_output_max_diff": x_diff_gj,
        "inverse_identity_residual": inverse_residual,
        "neumann_series_converged": converged,
        "neumann_series_iterations": iterations,
        "neumann_output_max_diff": x_diff_series,
        "linear_residual_max_abs": max(
            abs(matvec(m, x_solved)[i] - y[i]) for i in range(n)),
    }


def _build_diagnosis(a, leontief_inv, y_total, x_solved):
    n = len(a)
    m = [[(1.0 if i == j else 0.0) - a[i][j] for j in range(n)] for i in range(n)]
    rho, rho_converged, _ = spectral_radius_nonneg(a)
    margin = 1.0 - rho
    hs_ok, hs_dets = hawkins_simon_minors(m)
    cond = condition_number_1norm(m, leontief_inv)
    residual = [matvec(m, x_solved)[i] - y_total[i] for i in range(n)]
    residual_norm = vec_abs_sum(residual)
    nonsingular = True
    productive = rho < 1.0 and hs_ok
    near_critical = (0.0 <= margin <= NEAR_CRITICAL_MARGIN) or cond >= NEAR_SINGULAR_COND

    if not rho_converged:
        rho_text = "谱半径幂迭代未完全收敛（已用 A^2 周期后备），数值仅供方向判断"
    else:
        rho_text = "谱半径 rho(A)=%.10g，距临界值 1 的余量 %.10g" % (rho, margin)
    if not productive:
        explanation = (
            "系统非生产性：%s；Hawkins-Simon 条件%s成立。"
            "非负的 Leontief 扩张不存在，不能输出有限精确足迹，也不允许用截断序列冒充。"
            % (rho_text, "" if hs_ok else "不"))
    elif near_critical:
        explanation = (
            "系统可逆且生产性成立，但接近临界：%s；1-范数条件数 %.4g。"
            "足迹对系数与需求扰动高度敏感，应结合条件数与残差阅读，微小输入误差会被放大。"
            % (rho_text, cond))
    else:
        explanation = (
            "系统可逆且具生产性：%s；Hawkins-Simon 条件成立；1-范数条件数 %.4g，"
            "残差 1-范数 %.4g，Leontief 逆整体求解稳定，循环依赖已完整闭合。"
            % (rho_text, cond, residual_norm))
    return StabilityDiagnosis(
        nonsingular=nonsingular,
        productive=productive,
        spectral_radius=rho,
        spectral_radius_converged=rho_converged,
        margin_to_unit=margin,
        condition_number=cond,
        hawkins_simon_ok=hs_ok,
        hawkins_simon_determinants=hs_dets,
        residual_1norm=residual_norm,
        near_critical=near_critical,
        explanation=explanation,
    )


def solve_system(model):
    """整体求解 ``(I-A)x=y`` 并返回 (x, L, 诊断)。

    不可逆抛 :class:`SingularSystemError`；可逆但非生产性抛
    :class:`NonProductiveSystemError`（诊断随异常携带）。
    """
    if not isinstance(model, InputOutputModel):
        raise FootprintError("需要 InputOutputModel 实例")
    n = len(model.sectors)
    m = [[(1.0 if i == j else 0.0) - model.a[i][j] for j in range(n)]
         for i in range(n)]
    y = model.y_total()
    try:
        lu, piv = lu_decompose(m)
        x_solved = lu_solve(lu, piv, y)
        leontief_inv = transpose([lu_solve(lu, piv, [1.0 if j == k else 0.0
                                                      for j in range(n)])
                                  for k in range(n)])
    except SingularSystemError as exc:
        # 奇异也要给出谱半径等诊断，便于说明 A=1 这类情形为何不能给 40
        rho, rho_converged, _ = spectral_radius_nonneg(model.a)
        hs_ok, hs_dets = hawkins_simon_minors(m)
        diagnosis = StabilityDiagnosis(
            nonsingular=False, productive=False, spectral_radius=rho,
            spectral_radius_converged=rho_converged, margin_to_unit=1.0 - rho,
            condition_number=None, hawkins_simon_ok=hs_ok,
            hawkins_simon_determinants=hs_dets, residual_1norm=float("nan"),
            near_critical=True,
            explanation="I-A 不可逆（谱半径 %.10g）：%s。精确足迹不存在，"
                        "截断展开不收敛，禁止报有限足迹。" % (rho, exc))
        err = SingularSystemError(diagnosis.explanation)
        err.diagnosis = diagnosis
        raise err
    # 数值非负性校验（生产性系统理论上 L 与 x 非负）
    neg_x = min(x_solved)
    neg_l = min(min(row) for row in leontief_inv)
    diagnosis = _build_diagnosis(model.a, leontief_inv, y, x_solved)
    if neg_x < -1e-9 * max(1.0, max(map(abs, x_solved))):
        raise NonProductiveSystemError("解出的总产出含显著负分量，系统不具经济生产性")
    if neg_l < -1e-9 * max(1.0, max(abs(v) for row in leontief_inv for v in row)):
        diagnosis.explanation += " 警告：Leontief 逆出现负分量，非负性失效。"
        diagnosis.productive = False
        err = NonProductiveSystemError(diagnosis.explanation)
        err.diagnosis = diagnosis
        raise err
    if not diagnosis.productive:
        err = NonProductiveSystemError(diagnosis.explanation)
        err.diagnosis = diagnosis
        raise err
    return x_solved, leontief_inv, diagnosis


def _input_summary(model, x_solved, leontief_inv):
    y = model.y_total()
    direct_at_demand = dot(model.f, y)
    return InputSummary(
        n_sectors=len(model.sectors),
        sectors=model.sectors,
        uses=model.uses,
        basis=model.basis.as_dict(),
        input_fingerprint=model.fingerprint,
        column_sum_max=model.column_sum_max,
        final_demand_total=sum(y),
        gross_output_total=sum(x_solved),
        direct_emissions_at_demand=direct_at_demand,
    )


def footprint(model, cache=None):
    """计算各需求用途承担的直接与间接（上游）排放交叉分解。"""
    if cache is not None:
        cached = cache.get(model)
        if cached is not None:
            return cached
    x_solved, leontief_inv, diagnosis = solve_system(model)
    n = len(model.sectors)
    f = model.f
    # M = diag(f) L Y；同时拆 direct = diag(f) Y 与 upstream = diag(f)(L-I)Y
    impact = [[0.0] * len(model.uses) for _ in range(n)]
    direct = [[0.0] * len(model.uses) for _ in range(n)]
    upstream = [[0.0] * len(model.uses) for _ in range(n)]
    for k, yk in enumerate(model.y_columns):
        ly = matvec(leontief_inv, yk)
        for i in range(n):
            d = f[i] * yk[i]
            u = f[i] * (ly[i] - yk[i])
            direct[i][k] = d
            upstream[i][k] = u
            impact[i][k] = d + u
    total_by_use = {use: sum(impact[i][k] for i in range(n))
                    for k, use in enumerate(model.uses)}
    total_by_sector = {model.sectors[i]: sum(impact[i][k]
                                             for k in range(len(model.uses)))
                       for i in range(n)}
    grand_total = sum(total_by_use.values())
    # 分解加总误差：用途合计、来源行业合计、f·x 三者必须对上
    sector_sum = sum(total_by_sector.values())
    fx_total = dot(f, x_solved)
    decomp_err = max(
        abs(grand_total - sector_sum),
        abs(grand_total - fx_total),
        abs(sector_sum - fx_total),
    ) / max(1.0, abs(grand_total), abs(fx_total))
    # 残差 (I-A)x - y
    m = [[(1.0 if i == j else 0.0) - model.a[i][j] for j in range(n)]
         for i in range(n)]
    residual = [matvec(m, x_solved)[i] - model.y_total()[i] for i in range(n)]
    cross_validation = cross_validate_solution(model, x_solved, leontief_inv)
    result = FootprintResult(
        input_summary=_input_summary(model, x_solved, leontief_inv),
        output_x=x_solved,
        leontief_inverse=leontief_inv,
        impact_matrix=impact,
        direct_by_sector_use=direct,
        upstream_by_sector_use=upstream,
        total_by_use=total_by_use,
        total_by_source_sector=total_by_sector,
        grand_total=grand_total,
        residual=residual,
        residual_1norm=vec_abs_sum(residual),
        decomposition_error=decomp_err,
        diagnosis=diagnosis,
        cross_validation=cross_validation,
        balance_warnings=list(model.balance_warnings),
    )
    if cache is not None:
        cache.put(model, result)
    return result


# ---------------------------------------------------------------------------
# 截断序列（仅用于对照诊断，绝不能冒充精确足迹）
# ---------------------------------------------------------------------------

def truncated_footprint(model, layers):
    """用有限层 ``sum_{t=0..layers} A^t y`` 近似，仅用于展示截断偏差。

    返回的字典带 ``truncated_note`` 强制标注；调用方不得将其作为精确足迹。
    """
    if not isinstance(layers, int) or layers < 0:
        raise FootprintError("layers 必须为非负整数")
    n = len(model.sectors)
    y = model.y_total()
    x_approx = y[:]
    power_y = y[:]
    for _ in range(layers):
        power_y = matvec(model.a, power_y)
        x_approx = [x_approx[i] + power_y[i] for i in range(n)]
    total = dot(model.f, x_approx)
    direct_total = dot(model.f, y)
    return {
        "layers": layers,
        "output_approximation": x_approx,
        "footprint_approximation": total,
        "direct_at_demand": direct_total,
        "upstream_approximation": total - direct_total,
        "truncated_note": (
            "这是 %d 层截断的有限近似，不是精确足迹；循环依赖下结果随层数变化，"
            "精确值须经 solve_system/footprint 整体求解。" % layers),
    }


# ---------------------------------------------------------------------------
# 需求扰动响应
# ---------------------------------------------------------------------------

def demand_response(model, delta_y=None, *, use=None, delta_values=None,
                    cache=None):
    """查询某需求扰动的边际响应。

    三种用法：
      * ``delta_y``：长度 n 的总需求增量向量；
      * ``use`` 给定且 ``delta_values=None``：该用途整体移除（增量为 -y_k）；
      * ``use`` + ``delta_values``：在该用途上叠加指定增量。
    返回基线/扰动产出与排放、增量排放的来源行业分解，以及乘数 ``f L``。
    """
    n = len(model.sectors)
    base = footprint(model, cache=cache)
    if delta_y is not None:
        if len(delta_y) != n:
            raise FootprintError("delta_y 维度与行业数不一致")
        delta = [_finite_number(v, "delta_y") for v in delta_y]
        label = "外部增量向量"
    elif use is not None:
        if use not in model.uses:
            raise FootprintError("未知需求用途：" + use)
        k = model.uses.index(use)
        if delta_values is None:
            delta = [-model.y_columns[k][i] for i in range(n)]
            label = "移除用途 " + use
        else:
            if len(delta_values) != n:
                raise FootprintError("delta_values 维度与行业数不一致")
            delta = [_finite_number(v, "delta_values") for v in delta_values]
            label = "用途 %s 叠加增量" % use
    else:
        raise FootprintError("必须提供 delta_y 或 use")
    dx = matvec(base.leontief_inverse, delta)
    # 排放乘数矩阵 fL（行 i 为来源行业，列 j 为受扰动需求行业）
    fl = [[model.f[i] * base.leontief_inverse[i][j] for j in range(n)]
          for i in range(n)]
    delta_emissions_by_sector = {
        model.sectors[i]: model.f[i] * dx[i] for i in range(n)}
    perturbed_x = [base.output_x[i] + dx[i] for i in range(n)]
    perturbed_total = base.grand_total + sum(delta_emissions_by_sector.values())
    if min(perturbed_x) < -1e-9 * max(1.0, max(map(abs, perturbed_x))):
        feasibility = "扰动后总产出出现负分量，该需求情景在非负生产下不可行"
    else:
        feasibility = "扰动后总产出非负，情景可行"
    return {
        "label": label,
        "delta_y": delta,
        "delta_output": dx,
        "emission_multiplier_fL": fl,
        "delta_emissions_by_source_sector": delta_emissions_by_sector,
        "delta_emissions_total": sum(delta_emissions_by_sector.values()),
        "baseline_footprint": base.grand_total,
        "perturbed_footprint": perturbed_total,
        "perturbed_output_x": perturbed_x,
        "feasibility": feasibility,
        "input_fingerprint": model.fingerprint,
    }


# ---------------------------------------------------------------------------
# 行业分类映射（一对多权重、覆盖与守恒、不完整映射保留未覆盖需求）
# ---------------------------------------------------------------------------

def validate_crosswalk(crosswalk, source_sectors, target_sectors,
                       tol=CROSSWALK_TOL):
    """校验分类映射的覆盖与权重守恒。返回 (S 映射矩阵, 未覆盖的来源行业)。

    S[t, s] 为来源行业 s 落到目标行业 t 的权重；每个来源行业权重和必须为 1，
    未出现在映射中的来源行业按未覆盖保留，不静默丢弃。
    """
    if not isinstance(crosswalk, list) or not crosswalk:
        raise FootprintError("crosswalk 必须为非空列表")
    src = tuple(source_sectors)
    tgt = tuple(target_sectors)
    s_index = {s: i for i, s in enumerate(src)}
    t_index = {t: i for i, t in enumerate(tgt)}
    weights = {s: [] for s in src}
    for row in crosswalk:
        if not isinstance(row, dict) or "from" not in row or "to" not in row \
                or "weight" not in row:
            raise FootprintError("crosswalk 每行需包含 from/to/weight")
        frm, to, w = row["from"], row["to"], row["weight"]
        if frm not in s_index:
            raise FootprintError("crosswalk 引用未知来源行业：" + str(frm))
        if to not in t_index:
            raise FootprintError("crosswalk 引用未知目标行业：" + str(to))
        w = _finite_number(w, "crosswalk.weight")
        if w < 0.0:
            raise FootprintError("crosswalk 权重必须非负：%s->%s" % (frm, to))
        weights[frm].append((t_index[to], w))
    uncovered = []
    s_matrix = [[0.0] * len(src) for _ in range(len(tgt))]
    for s_name, pairs in weights.items():
        s_idx = s_index[s_name]
        if not pairs:
            uncovered.append(s_name)
            continue
        total = sum(w for _, w in pairs)
        if abs(total - 1.0) > tol:
            raise FootprintError(
                "来源行业 %s 的映射权重和为 %.12g，不满足守恒（应为 1）"
                % (s_name, total))
        for t_idx, w in pairs:
            s_matrix[t_idx][s_idx] = w
    if uncovered:
        # 不完整映射：未覆盖来源行业作为独立目标行业保留，足迹不归零
        existing = set(tgt)
        extra = [s for s in uncovered if s not in existing]
        for s_name in extra:
            s_idx = s_index[s_name]
            tgt = tgt + (s_name,)
            s_matrix.append([0.0] * len(src))
            s_matrix[-1][s_idx] = 1.0
    return s_matrix, list(tgt), uncovered


def reclassify(model, target_sectors, crosswalk, *, tol=CROSSWALK_TOL):
    """按一对多权重把模型重分类到目标行业体系。

    系数矩阵由重分类后的中间交易额 S Z S' 与总产出 S x 重新相除重建，
    绝不对两个 A 矩阵做平均；强度按 ``S diag(f x) 1 / S x`` 以排放金额守恒聚合。
    未覆盖的来源行业整体保留为独立行业，其需求与足迹不归零。
    """
    if not isinstance(model, InputOutputModel):
        raise FootprintError("需要 InputOutputModel 实例")
    target_sectors = tuple(target_sectors)
    if len(set(target_sectors)) != len(target_sectors):
        raise FootprintError("目标行业名称重复")
    s_matrix, all_targets, uncovered = validate_crosswalk(
        crosswalk, model.sectors, target_sectors, tol)
    # 守恒校验：S 是列随机矩阵，每个来源行业一列、列和为 1
    # （未覆盖来源行业以恒等行保留，对应列同样和为 1）
    n_src = len(model.sectors)
    for s in range(n_src):
        col_sum = sum(s_matrix[t][s] for t in range(len(all_targets)))
        if abs(col_sum - 1.0) > tol:
            raise FootprintError("来源行业 %s 的映射列不守恒（列和 %.12g）"
                                 % (model.sectors[s], col_sum))
    s_t = transpose(s_matrix)
    # 总产出与最终需求：线性映射
    x_new = matvec(s_matrix, model.x)
    y_new_cols = [matvec(s_matrix, col) for col in model.y_columns]
    # 中间交易额重建：Z_new = S Z S^T
    z_new = matmul(matmul(s_matrix, model.z), s_t)
    # 直接排放金额 e = f * x 守恒聚合，强度以新总产出为分母
    emissions = [model.f[i] * model.x[i] for i in range(n_src)]
    e_new = matvec(s_matrix, emissions)
    f_new = [e_new[t] / x_new[t] if x_new[t] > 0 else 0.0
             for t in range(len(all_targets))]
    for t, name in enumerate(all_targets):
        if x_new[t] < 0.0:
            raise FootprintError("重分类后行业 %s 总产出为负" % name)
        if x_new[t] == 0.0 and any(z_new[i][t] != 0.0
                                   for i in range(len(all_targets))):
            raise FootprintError("重分类后行业 %s 零产出却存在中间投入" % name)
    new_crosswalk = [{"from": t, "to": t, "weight": 1.0} for t in all_targets]
    provenance = {
        "coefficients_rebuilt_from":
            "重分类后中间交易额 S Z S^T 与总产出 S x 重新相除重建 A，"
            "未对系数矩阵做平均；强度按排放金额 S(f*x) 守恒后除以新总产出",
        "reclassified": True,
        "source_basis": model.basis.as_dict(),
        "source_sectors": list(model.sectors),
        "target_sectors": list(all_targets),
        "uncovered_retained": uncovered,
        "mapping": "one-to-many weighted" if any(
            sum(1 for w in row if w > 0) > 1 for row in s_matrix) else "aggregation/identity",
    }
    new_model = InputOutputModel(
        sectors=list(all_targets),
        transactions=z_new,
        gross_output=x_new,
        final_demands=[{"use": u, "values": y_new_cols[k]}
                       for k, u in enumerate(model.uses)],
        direct_intensity=f_new,
        basis=model.basis,
        crosswalk=new_crosswalk,
        provenance=provenance,
    )
    return new_model


# ---------------------------------------------------------------------------
# 缓存：绑定行业映射与价格口径，禁止跨版本借用逆矩阵
# ---------------------------------------------------------------------------

class ModelCache:
    """以口径 + 行业分类 + 系数/强度/需求指纹为键的结果缓存。"""

    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(model):
        if not isinstance(model, InputOutputModel):
            raise FootprintError("需要 InputOutputModel 实例")
        return (model.basis.key(), model.sectors, model.uses, model.fingerprint)

    def get(self, model):
        key = self.key_for(model)
        value = self._store.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, model, result):
        self._store[self.key_for(model)] = result

    def __len__(self):
        return len(self._store)

    def stats(self):
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# 两版报告差异分解（需求 / 直接强度 / 供应结构，并列顺序敏感性）
# ---------------------------------------------------------------------------

@dataclass
class ReportComparison:
    base_fingerprint: str
    alt_fingerprint: str
    base_footprint: float
    alt_footprint: float
    total_delta: float
    # 两条固定顺序的逐因素贡献
    order_d_i_s: dict[str, float]
    order_s_i_d: dict[str, float]
    # 六个排列的对称（Shapley）平均贡献与交互项
    shapley: dict[str, float]
    order_spread: dict[str, float]
    interaction_term: float
    aggregation_error: float
    explanation: str


def _counterfactual_total(base, alt, demand_model, intensity_model, structure_model):
    """用指定三因素组合求解总足迹；因素取值 'base' 或 'alt'。"""
    sectors = base.sectors
    if sectors != alt.sectors or base.basis.key() != alt.basis.key():
        raise FootprintError(
            "两版报告行业分类或价格口径不一致，须先 reclassify 到同一体系再比较")
    uses = base.uses
    if uses != alt.uses:
        raise FootprintError("两版报告需求用途集合不一致，无法逐用途比较")
    src = alt if structure_model == "alt" else base
    demand = alt if demand_model == "alt" else base
    inten = alt if intensity_model == "alt" else base
    model = InputOutputModel(
        sectors=list(sectors),
        transactions=mat_copy(src.z),
        gross_output=list(src.x),
        final_demands=[{"use": u, "values": list(demand.y_columns[k])}
                       for k, u in enumerate(uses)],
        direct_intensity=list(inten.f),
        basis=base.basis,
    )
    return footprint(model).grand_total


def compare_reports(base, alt):
    """把两版总足迹差异分解为需求、直接强度、供应结构三因素。

    同时给出两条顺序的序贯分解、六个排列的对称 Shapley 平均，以及顺序极差；
    交互项单独列示，不塞给任何单一因素。
    """
    if not isinstance(base, InputOutputModel) or not isinstance(alt, InputOutputModel):
        raise FootprintError("比较需要两个 InputOutputModel 实例")
    total_base = footprint(base).grand_total
    total_alt = footprint(alt).grand_total
    total_delta = total_alt - total_base

    def sequential(order):
        state = {FACTOR_DEMAND: "base", FACTOR_INTENSITY: "base",
                 FACTOR_STRUCTURE: "base"}
        contributions = {}
        previous = total_base
        for factor in order:
            state[factor] = "alt"
            current = _counterfactual_total(
                base, alt, state[FACTOR_DEMAND],
                state[FACTOR_INTENSITY], state[FACTOR_STRUCTURE])
            contributions[factor] = current - previous
            previous = current
        return contributions

    order1 = sequential((FACTOR_DEMAND, FACTOR_INTENSITY, FACTOR_STRUCTURE))
    order2 = sequential((FACTOR_STRUCTURE, FACTOR_INTENSITY, FACTOR_DEMAND))
    # 六排列对称平均（三因素 Shapley）
    accum = {f: 0.0 for f in FACTORS}
    perms = list(permutations(FACTORS))
    for order in perms:
        for factor, value in sequential(order).items():
            accum[factor] += value
    shapley = {f: accum[f] / len(perms) for f in FACTORS}
    # 顺序敏感性：各因素在六排列中的极差
    per_factor_values = {f: [] for f in FACTORS}
    for order in perms:
        seq = sequential(order)
        for f in FACTORS:
            per_factor_values[f].append(seq[f])
    spread = {f: max(per_factor_values[f]) - min(per_factor_values[f])
              for f in FACTORS}
    # 交互项 = 总差异 - 三因素主效应（对称均值）之和
    interaction = total_delta - sum(shapley.values())
    agg_err = abs(sum(order1.values()) - total_delta) / max(1.0, abs(total_delta))
    explanation = (
        "对称(Shapley)贡献为六个分解排列的平均，顺序极差反映交互作用导致的"
        "顺序敏感性；交互项 %.6g 单独列示，未归入任何单一因素。"
        "若某因素顺序极差接近其贡献量级，说明该因素与其他因素存在强交互，"
        "不宜用单一顺序的序贯数字下结论。" % interaction)
    return ReportComparison(
        base_fingerprint=base.fingerprint,
        alt_fingerprint=alt.fingerprint,
        base_footprint=total_base,
        alt_footprint=total_alt,
        total_delta=total_delta,
        order_d_i_s=order1,
        order_s_i_d=order2,
        shapley=shapley,
        order_spread=spread,
        interaction_term=interaction,
        aggregation_error=agg_err,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse
    from pathlib import Path
    from .input_output_footprints_input import load_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="schema_version=1 的投入产出资料 JSON")
    args = parser.parse_args(argv)
    bundle = load_bundle(args.path)
    model = InputOutputModel.from_bundle(bundle)
    try:
        result = footprint(model)
    except SingularSystemError as exc:
        print(json.dumps({"status": "singular", "explanation": str(exc)},
                         ensure_ascii=False, indent=2))
        return 2
    except NonProductiveSystemError as exc:
        print(json.dumps({"status": "non_productive", "explanation": str(exc)},
                         ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
