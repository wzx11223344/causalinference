# 贡献指南

感谢你考虑为 CausalInference 做出贡献！因果推断是计量经济学与机器学习交叉的最前沿领域，请确保你的贡献满足方法论正确性要求。

## 行为准则

本项目遵守 [Contributor Covenant](CODE_OF_CONDUCT.md) 行为准则。

## 贡献范围

- **新因果方法**：Generalized Random Forest 扩展、正交随机森林 (Oprescu et al. 2019)
- **推断方法**：Conformal inference for CATE、Uniform confidence bands
- **Nuisance 函数学习器**：支持自定义 sklearn 兼容的回归器/分类器
- **文档与测试**：理论推导注释、仿真研究、基准测试

## 开发环境

```bash
git clone https://github.com/wzx11223344/causalinference.git
cd causalinference
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -e .
```

## 代码规范

### 估计量接口

任何因果推断方法必须遵循统一 API：

```python
class NewCausalMethod:
    def __init__(self, model_y=None, model_t=None, **kwargs):
        """Initialize with optional nuisance function learners."""
        ...

    def fit(self, X, y, T, **kwargs) -> NewCausalResult:
        """Fit the model and return a result object."""
        ...

class NewCausalResult:
    @property
    def ate(self) -> float: ...
    @property
    def cate(self) -> np.ndarray: ...
    def summary(self) -> str: ...
    def predict(self, X_new) -> np.ndarray: ...
```

### 方法论要求

- **Double ML 实现**必须使用 Neyman 正交得分的通用公式
- **Causal Forest** 必须实现 Honest Splitting (half-sampling)
- **Meta-Learners** 必须支持任意 sklearn 兼容模型作为基学习器
- 所有推断方法需提供分析标准误和 bootstrap 备选方案

## 测试

```bash
python -m pytest tests/ -v
```

仿真测试要求：
- 数据生成过程 (DGP) 使用已知真值
- 检查覆盖率: 95% CI 覆盖真值的比例应在 92%-98% 范围内
- 检验 Empirical Bias 和 RMSE

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：
```
feat(dml): add interactive regression model support
fix(causal_forest): correct honest split bias correction
docs(readme): add Neyman orthogonality derivation
```

## 问题反馈

- 安全漏洞: `3521257027@QQ.com`
- 一般问题: [GitHub Issues](https://github.com/wzx11223344/causalinference/issues)
