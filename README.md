# 跨行业投入产出足迹与循环依赖归因

货币口径下的 Leontief 投入产出足迹分析。交易额按供给行业为行、使用行业为列，
价格基期和币种独立登记；只做线性核算，不模拟生产设备或物理反应。

## 资料读取

`src/input_output_footprints_input.py` 读取 schema_version=1 的 JSON 资料
（`sectors / transactions / gross_output / final_demands / direct_intensity /
basis / crosswalk`），读取时不执行推断、匹配或核算，保留原始数据。
`fixtures/sample.json` 使用合成记录。

## 足迹分析（新增）

`src/input_output_footprints_analysis.py` 在读取器之上提供可调用分析接口，
零第三方运行依赖，线性代数（LU、Gauss–Jordan、幂迭代、条件数、主子式）均为纯 Python。

口径与方法：

* `A[i,j]` 为行业 j 每产出一元所需行业 i 的投入金额；系数只由**本版**交易额 `Z`
  与总产出 `x` 重建（`A = Z / x_j`），不对不同版本的系数矩阵做平均。
* 整体求解 `(I-A)x = y`，循环依赖通过 Leontief 逆 `L=(I-A)^{-1}` 一次闭合，
  不做逐层截断；`truncated_footprint` 仅用于展示截断偏差，结果强制带
  `truncated_note`，不能冒充精确足迹。
* 用途 k 的足迹 `M[:,k] = diag(f) L y_k`，并拆成直接部分 `diag(f) y_k`
  与上游间接部分 `diag(f)(L-I)y_k`，按需求用途 × 排放来源行业交叉分解。
* 每次结果附带输入指纹、线性系统残差、分解加总误差，以及两条独立路径的
  交叉验证（Gauss–Jordan 独立求逆 + Neumann 幂级数极限）。
* 稳定性诊断分别报告：不可逆（`SingularSystemError`，如单行业 A=1）、
  非生产性（`NonProductiveSystemError`，谱半径 ≥1）、接近临界
  （谱半径余量或条件数触发），三类分开说明，不用截断有限值冒充精确足迹。
* `reclassify` 处理一对多权重映射：先校验覆盖与权重守恒，再用重分类后的
  `S Z S'` 与 `S x` 重建系数；不完整映射把未覆盖行业整体保留，需求与足迹不归零。
* `ModelCache` 以价格口径 + 行业分类 + 输入指纹为键，禁止跨版本借用逆矩阵。
* `compare_reports` 把两版差异分解为需求、直接强度、供应结构三因素，
  并列两条顺序与六个排列的 Shapley 对称均值，交互项单独列示并报告顺序极差。
* `demand_response` 查询需求扰动的边际响应 `dx = L Δy`、排放乘数 `fL`
  与来源行业增量分解，并校验情景非负可行性。

调用示例：

```python
from src.input_output_footprints_input import load_bundle
from src.input_output_footprints_analysis import (
    InputOutputModel, footprint, compare_reports, demand_response,
    reclassify, ModelCache,
)

model = InputOutputModel.from_bundle(load_bundle("fixtures/sample.json"))
result = footprint(model)                 # 单行业 A=0.5,y=10,f=2 -> 足迹 40
result.total_by_use                       # {'park': 40.0}
result.emissions_by_use("park")           # 来源行业交叉分解
result.cross_validation                    # 独立路径互证
result.diagnosis.explanation               # 稳定性解释

cache = ModelCache()
footprint(model, cache=cache)             # 同口径同输入才命中
resp = demand_response(model, use="park") # 需求扰动响应
```

命令行：

    python -m src.input_output_footprints_analysis fixtures/sample.json

不可逆返回码 2，非生产性返回码 3，正常为 0。

## 测试

    python -m unittest discover -s tests -v
    python -m compileall -q src tests

核对场景包括：单行业 A=0.5 链（产出 20/足迹 40，直接 20+上游 20）、A=1 奇异
不得仍报 40、A>1 非生产性、跨行业循环与周期振荡谱半径、零最终需求、
接近临界 A=0.999 的大结果双路径互证、行业拆分/合并守恒、不完整映射保留、
三因素差异分解的顺序敏感性、扰动响应与缓存口径隔离。
当前代码没有第三方运行依赖，使用 Python 3（开发环境 3.11，目标 3.14）。
