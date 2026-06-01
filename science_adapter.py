"""
科学计算适配器 — 爱因斯坦/达尔文/伽利略灵魂的科研工具链
=================================================
通过三条路径控制科学计算软件：

路径1: Python 科学计算生态（推荐，精确计算）
  - NumPy/SciPy: 数值计算、线性代数、优化
  - SymPy: 符号计算、公式推导
  - Matplotlib/Plotly: 数据可视化
  - Astropy: 天文计算（伽利略）
  - Biopython: 生物信息学（达尔文）
  - QuantLib: 量化金融（爱因斯坦/纳什）

路径2: 外部计算引擎
  - Wolfram Alpha API: 数学/物理/化学查询
  - MATLAB Engine API: 工程计算
  - Jupyter Kernel: 交互式计算

路径3: GUI 感知（兜底）
  - Mano-P 操作 SPSS、Origin、GraphPad 等

支持软件列表：
  - Python 科学栈（NumPy/SciPy/SymPy/Matplotlib）
  - Wolfram Alpha（API）
  - MATLAB（Engine API）
  - Jupyter（Kernel 协议）
  - SPSS（GUI 感知）
  - Origin（GUI 感知）
  - GraphPad Prism（GUI 感知）

集成点：
  - soul_orchestrator: einstein/galileo/darwin 灵魂注册 science_* 工具
  - harness_runtime: Wolfram API + 计算端口白名单
  - guardian: MATLAB/Jupyter 进程监控
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ScienceDomain(Enum):
    PHYSICS = "physics"
    QUANTUM = "quantum_mechanics"
    ASTRONOMY = "astronomy"
    ASTROPHYSICS = "astrophysics"
    BIOLOGY = "biology"
    GENETICS = "genetics"
    BIOINFORMATICS = "bioinformatics"
    FINANCE = "quantitative_finance"
    MATHEMATICS = "mathematics"
    STATISTICS = "statistics"
    CHEMISTRY = "chemistry"


class ComputeEngine(Enum):
    PYTHON_NATIVE = "python_native"
    WOLFRAM_ALPHA = "wolfram_alpha"
    MATLAB = "matlab"
    JUPYTER = "jupyter"
    JULIA = "julia"
    R_LANGUAGE = "r"


SCIENCE_CONFIG = {
    "wolfram_api_base": "https://api.wolframalpha.com/v2/query",
    "wolfram_appid_env": "WOLFRAM_APPID",
    "matlab_paths": {
        "windows": [r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"],
        "darwin": ["/Applications/MATLAB_R2024b.app/bin/matlab"],
        "linux": ["/usr/local/MATLAB/R2024b/bin/matlab"],
    },
    "jupyter_kernel_port_range": (5000, 5100),
    "r_paths": {
        "windows": [r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe"],
        "darwin": ["/usr/local/bin/Rscript"],
        "linux": ["/usr/bin/Rscript"],
    },
    "julia_paths": {
        "windows": [r"C:\Users\julia\julia-1.10\bin\julia.exe"],
        "darwin": ["/usr/local/bin/julia"],
        "linux": ["/usr/bin/julia"],
    },
    "plot_output_dir": str(Path("D:/renders/science")),
    "compute_timeout_seconds": 120,
    "max_array_size": 10_000_000,
}


@dataclass
class ComputeResult:
    status: str
    result_type: str
    value: Any = None
    expression: str = ""
    latex: str = ""
    plot_path: str = ""
    error: str = ""
    engine: str = ""
    compute_time_ms: float = 0.0


@dataclass
class PhysicsConstant:
    name: str
    symbol: str
    value: float
    unit: str
    uncertainty: str = ""


PHYSICS_CONSTANTS = {
    "c": PhysicsConstant("光速", "c", 299792458.0, "m/s"),
    "h": PhysicsConstant("普朗克常数", "h", 6.62607015e-34, "J·s"),
    "hbar": PhysicsConstant("约化普朗克常数", "ℏ", 1.054571817e-34, "J·s"),
    "G": PhysicsConstant("万有引力常数", "G", 6.67430e-11, "m³/(kg·s²)"),
    "k_B": PhysicsConstant("玻尔兹曼常数", "k_B", 1.380649e-23, "J/K"),
    "e": PhysicsConstant("元电荷", "e", 1.602176634e-19, "C"),
    "m_e": PhysicsConstant("电子质量", "m_e", 9.1093837015e-31, "kg"),
    "m_p": PhysicsConstant("质子质量", "m_p", 1.67262192369e-27, "kg"),
    "N_A": PhysicsConstant("阿伏伽德罗常数", "N_A", 6.02214076e23, "mol⁻¹"),
    "R": PhysicsConstant("气体常数", "R", 8.314462618, "J/(mol·K)"),
    "sigma": PhysicsConstant("斯特藩-玻尔兹曼常数", "σ", 5.670374419e-8, "W/(m²·K⁴)"),
    "mu_0": PhysicsConstant("真空磁导率", "μ₀", 1.25663706212e-6, "H/m"),
    "epsilon_0": PhysicsConstant("真空介电常数", "ε₀", 8.8541878128e-12, "F/m"),
}

ASTRONOMY_DATA = {
    "sun_mass_kg": 1.989e30,
    "earth_mass_kg": 5.972e24,
    "moon_mass_kg": 7.342e22,
    "earth_radius_km": 6371.0,
    "sun_radius_km": 696340.0,
    "earth_sun_distance_km": 1.496e8,
    "earth_moon_distance_km": 384400.0,
    "earth_orbital_period_days": 365.256,
    "moon_orbital_period_days": 27.322,
    "solar_luminosity_w": 3.828e26,
    "earth_escape_velocity_km_s": 11.186,
    "solar_escape_velocity_km_s": 617.7,
}


class PythonComputeEngine:
    def __init__(self):
        self._available_packages: Optional[Dict[str, bool]] = None

    def _check_packages(self) -> Dict[str, bool]:
        if self._available_packages is not None:
            return self._available_packages
        packages = {}
        for pkg in ["numpy", "scipy", "sympy", "matplotlib", "astropy", "biopython", "pandas", "plotly"]:
            try:
                __import__(pkg)
                packages[pkg] = True
            except ImportError:
                packages[pkg] = False
        self._available_packages = packages
        return packages

    def is_available(self) -> bool:
        pkgs = self._check_packages()
        return pkgs.get("numpy", False)

    def evaluate(self, expression: str, variables: Dict[str, float] = None) -> ComputeResult:
        start = time.time()
        try:
            import numpy as np
            safe_globals = {
                "np": np, "numpy": np,
                "abs": abs, "max": max, "min": min,
                "sum": sum, "round": round, "len": len,
                "range": range, "list": list, "tuple": tuple,
                "pi": np.pi, "e": np.e, "inf": np.inf,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
                "exp": np.exp, "power": np.power,
                "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
                "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            }
            if variables:
                safe_globals.update(variables)
            for const_name, const in PHYSICS_CONSTANTS.items():
                safe_globals[const_name] = const.value
            safe_globals.update(ASTRONOMY_DATA)

            result = eval(expression, {"__builtins__": {}}, safe_globals)
            elapsed = (time.time() - start) * 1000

            return ComputeResult(
                status="success",
                result_type=type(result).__name__,
                value=result.tolist() if hasattr(result, "tolist") else result,
                expression=expression,
                engine="python_native",
                compute_time_ms=elapsed,
            )
        except Exception as e:
            return ComputeResult(
                status="error",
                result_type="error",
                expression=expression,
                error=str(e),
                engine="python_native",
                compute_time_ms=(time.time() - start) * 1000,
            )

    def symbolic(self, expression: str) -> ComputeResult:
        start = time.time()
        try:
            import sympy
            parsed = sympy.sympify(expression)
            latex_str = sympy.latex(parsed)
            simplified = sympy.simplify(parsed)
            elapsed = (time.time() - start) * 1000

            return ComputeResult(
                status="success",
                result_type="symbolic",
                value=str(simplified),
                expression=expression,
                latex=latex_str,
                engine="sympy",
                compute_time_ms=elapsed,
            )
        except ImportError:
            return ComputeResult(status="error", error="SymPy 未安装", engine="sympy")
        except Exception as e:
            return ComputeResult(status="error", error=str(e), engine="sympy")

    def plot(self, expression: str, x_range: Tuple[float, float] = (-10, 10),
             title: str = "", output_path: str = "") -> ComputeResult:
        start = time.time()
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x = np.linspace(x_range[0], x_range[1], 1000)
            safe_globals = {"np": np, "x": x, "pi": np.pi, "e": np.e}
            safe_globals.update({k: v.value for k, v in PHYSICS_CONSTANTS.items()})
            y = eval(expression, {"__builtins__": {}}, safe_globals)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y)
            ax.set_title(title or expression)
            ax.grid(True)

            output = output_path or str(Path(SCIENCE_CONFIG["plot_output_dir"]) / f"plot_{int(time.time())}.png")
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)

            elapsed = (time.time() - start) * 1000
            return ComputeResult(
                status="success",
                result_type="plot",
                expression=expression,
                plot_path=output,
                engine="matplotlib",
                compute_time_ms=elapsed,
            )
        except ImportError:
            return ComputeResult(status="error", error="Matplotlib 未安装", engine="matplotlib")
        except Exception as e:
            return ComputeResult(status="error", error=str(e), engine="matplotlib")


class WolframAlphaClient:
    def __init__(self, appid: str = None):
        self._appid = appid or os.environ.get(SCIENCE_CONFIG["wolfram_appid_env"], "")
        self._base = SCIENCE_CONFIG["wolfram_api_base"]

    def is_available(self) -> bool:
        return bool(self._appid)

    def query(self, query: str, output: str = "json") -> dict:
        if not self._appid:
            return {"error": "未配置 Wolfram Alpha APPID (设置 WOLFRAM_APPID 环境变量)"}
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({
            "appid": self._appid,
            "input": query,
            "output": output,
            "format": "plaintext",
        })
        url = f"{self._base}?{params}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}


class ScienceAdapter:
    """
    科学计算适配器 — 爱因斯坦/达尔文/伽利略灵魂专用

    三层控制：
    1. Python 原生计算：NumPy/SciPy/SymPy/Astropy/Biopython
    2. 外部引擎：Wolfram Alpha API / MATLAB / Jupyter
    3. GUI 感知：Mano-P 操作 SPSS/Origin 等

    使用：
        adapter = ScienceAdapter()
        adapter.evaluate("2 * pi * 6371")  # 地球周长
        adapter.symbolic("diff(x**3, x)")  # 符号微分
        adapter.plot("sin(x) * exp(-x/10)")  # 绘图
        adapter.wolfram_query("mass of sun")  # Wolfram查询
    """

    def __init__(self, domain: ScienceDomain = ScienceDomain.PHYSICS):
        self._domain = domain
        self._python_engine = PythonComputeEngine()
        self._wolfram = WolframAlphaClient()
        self._task_counter = 0

    def status(self) -> dict:
        pkgs = self._python_engine._check_packages()
        return {
            "adapter": "ScienceAdapter",
            "domain": self._domain.value,
            "python_available": self._python_engine.is_available(),
            "packages": pkgs,
            "wolfram_available": self._wolfram.is_available(),
            "physics_constants_count": len(PHYSICS_CONSTANTS),
            "astronomy_data_keys": len(ASTRONOMY_DATA),
            "tools": [
                "science_evaluate", "science_symbolic", "science_plot",
                "science_wolfram_query", "science_get_constant",
                "science_orbital_calc", "science_genetic_calc",
                "science_statistical_test", "science_curve_fit",
                "science_screenshot",
            ],
        }

    def evaluate(self, expression: str, variables: Dict[str, float] = None) -> ComputeResult:
        return self._python_engine.evaluate(expression, variables)

    def symbolic(self, expression: str) -> ComputeResult:
        return self._python_engine.symbolic(expression)

    def plot(self, expression: str, x_range: Tuple[float, float] = (-10, 10),
             title: str = "", output_path: str = "") -> ComputeResult:
        return self._python_engine.plot(expression, x_range, title, output_path)

    def wolfram_query(self, query: str) -> dict:
        return self._wolfram.query(query)

    def get_constant(self, name: str) -> dict:
        const = PHYSICS_CONSTANTS.get(name)
        if const:
            return {"status": "found", "constant": asdict(const)}
        return {"status": "not_found", "available": list(PHYSICS_CONSTANTS.keys())}

    def orbital_calc(self, body1_mass: float, body2_mass: float,
                     distance: float, output: str = "period") -> ComputeResult:
        expr = f"2 * pi * sqrt(({distance}**3) / (6.67430e-11 * ({body1_mass} + {body2_mass})))"
        return self._python_engine.evaluate(expr)

    def genetic_calc(self, dominant_freq: float, recessive_freq: float,
                     population: int = 1000) -> ComputeResult:
        p = dominant_freq
        q = recessive_freq
        variables = {
            "p": p, "q": q, "N": population,
            "AA": p * p * population,
            "Aa": 2 * p * q * population,
            "aa": q * q * population,
        }
        expr = "{'AA': AA, 'Aa': Aa, 'aa': aa, 'chi_check': p + q}"
        return self._python_engine.evaluate(expr, variables)

    def statistical_test(self, data: List[float], test: str = "ttest",
                         mu: float = 0.0) -> ComputeResult:
        start = time.time()
        try:
            from scipy import stats
            import numpy as np
            arr = np.array(data)
            if test == "ttest":
                stat, pval = stats.ttest_1samp(arr, mu)
                result = {"statistic": float(stat), "p_value": float(pval)}
            elif test == "normaltest":
                stat, pval = stats.normaltest(arr)
                result = {"statistic": float(stat), "p_value": float(pval)}
            elif test == "shapiro":
                stat, pval = stats.shapiro(arr)
                result = {"statistic": float(stat), "p_value": float(pval)}
            else:
                return ComputeResult(status="error", error=f"未知检验: {test}", engine="scipy")
            elapsed = (time.time() - start) * 1000
            return ComputeResult(
                status="success", result_type="statistical",
                value=result, engine="scipy", compute_time_ms=elapsed,
            )
        except ImportError:
            return ComputeResult(status="error", error="SciPy 未安装", engine="scipy")
        except Exception as e:
            return ComputeResult(status="error", error=str(e), engine="scipy")

    def curve_fit(self, x_data: List[float], y_data: List[float],
                  func_type: str = "linear") -> ComputeResult:
        start = time.time()
        try:
            from scipy.optimize import curve_fit
            import numpy as np

            def linear(x, a, b): return a * x + b
            def quadratic(x, a, b, c): return a * x**2 + b * x + c
            def exponential(x, a, b, c): return a * np.exp(b * x) + c

            func_map = {"linear": linear, "quadratic": quadratic, "exponential": exponential}
            func = func_map.get(func_type, linear)
            x_arr, y_arr = np.array(x_data), np.array(y_data)
            popt, pcov = curve_fit(func, x_arr, y_arr)
            elapsed = (time.time() - start) * 1000
            return ComputeResult(
                status="success", result_type="curve_fit",
                value={"parameters": popt.tolist(), "covariance_diagonal": np.diag(pcov).tolist()},
                engine="scipy", compute_time_ms=elapsed,
            )
        except ImportError:
            return ComputeResult(status="error", error="SciPy 未安装", engine="scipy")
        except Exception as e:
            return ComputeResult(status="error", error=str(e), engine="scipy")


SCIENCE_SKILL_DEFINITION = {
    "skill_id": "science_compute",
    "name": "科学计算工具链",
    "description": "爱因斯坦/达尔文/伽利略灵魂专用：数值计算+符号推导+数据可视化+天文/生物/金融专业计算",
    "soul_mapping": ["einstein", "galileo", "darwin", "strategy"],
    "tools": [
        "science_evaluate", "science_symbolic", "science_plot",
        "science_wolfram_query", "science_get_constant",
        "science_orbital_calc", "science_genetic_calc",
        "science_statistical_test", "science_curve_fit",
        "science_screenshot",
    ],
    "connectors": [
        {"name": "python_scipy", "type": "python", "packages": ["numpy", "scipy", "sympy", "matplotlib"]},
        {"name": "wolfram_alpha", "type": "rest_api", "base_url": "https://api.wolframalpha.com/v2"},
        {"name": "astropy", "type": "python", "packages": ["astropy"], "soul": "galileo"},
        {"name": "biopython", "type": "python", "packages": ["biopython"], "soul": "darwin"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
}

_adapter_instance: Optional[ScienceAdapter] = None


def get_adapter() -> ScienceAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ScienceAdapter()
    return _adapter_instance
