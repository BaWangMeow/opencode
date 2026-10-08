# fel-diagnostics

面向自由电子激光（FEL）诊断系统的**仿真束流诊断数据生成器**，是硕士论文课题
《面向 FEL 诊断系统的在线数据采集与束流异常检测》的**数据底座**。

它生成多通道、物理上有相关性的束流诊断时序（BPM、束流电荷、能量、RF、束损、定时），
并注入多类异常（束损尖峰、BPM 漂移、横向振荡、RF 阶跃、噪声增大），同时产出**逐点、
逐通道、逐类型**的 ground-truth 标签，供后续无监督异常检测模型（AE / LSTM-VAE /
Anomaly Transformer 等）训练与评估使用。

## 为什么这样做

- **真实束流数据拿不到**（SHINE 在建），所以用仿真保证实验**可复现、标签可控**；
- **多通道相关**：异常检测的"关联偏差"类方法依赖通道间相关性，这里用
  "潜在物理因子 → 线性混合 → 观测通道" 的方式显式建模了这种相关性；
- **标签分层**：`label`（全局二值）/ `type_label`（异常类型）/ `channel_label`（逐通道），
  便于做细粒度的评价指标（Precision/Recall/F1、逐类对比）。

## 目录结构

```
fel-diagnostics/
├── src/feldiag/           # 包本体
│   ├── config.py          # 配置 dataclass
│   ├── signals.py         # 潜在因子生成 + 通道混合
│   ├── anomalies.py       # 异常注入 + 标签
│   ├── generator.py       # 顶层 generate()
│   ├── export.py          # 导出 HDF5 / NPZ / CSV
│   └── plot.py            # 可视化
├── scripts/generate.py    # 命令行入口
└── tests/test_generator.py
```

## 安装

```bash
cd projects/fel-diagnostics
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
```

## 快速开始

### 命令行生成

```bash
python scripts/generate.py --n-samples 50000 --n-anomalies 30 --seed 42 --out data/train.h5
```

### Python API

```python
from feldiag import GeneratorConfig, AnomalyConfig, generate, to_hdf5

cfg = GeneratorConfig(n_samples=50_000, sampling_rate=10.0, seed=42)
cfg.anomalies = AnomalyConfig(n_anomalies=30)

ds = generate(cfg)

print(ds.data.shape)         # (50000, 12)  -> (样本数, 通道数)
print(ds.channel_names)      # ['BPM1.x', 'BPM1.y', ...]
print(ds.label.sum())        # 异常样本总数
print(ds.events[0])          # 单个异常事件元数据（类型/起止/通道/幅度）

to_hdf5(ds, "data/train.h5")
```

### 可视化

```python
from feldiag import generate, plot_dataset
import matplotlib.pyplot as plt

ds = generate(GeneratorConfig(n_samples=5000, seed=1))
plot_dataset(ds, start=0, end=3000)
plt.show()
```

## 数据约定

| 字段 | 形状 | 含义 |
| :--- | :--- | :--- |
| `data` | (T, C) | 多通道时序 |
| `label` | (T,) | 全局二值异常标签（1 = 该时刻任一通道异常） |
| `type_label` | (T,) | 异常类型码（见下） |
| `channel_label` | (T, C) | 逐通道二值标签 |
| `events` | - | 每个异常的元数据（类型/起止/通道/幅度） |

异常类型码：`1=point(束损尖峰)` `2=drift(BPM漂移)` `3=oscillation(横向振荡)`
`4=level_shift(RF阶跃)` `5=noise(噪声增大)`，`0=normal`。

## 运行测试

```bash
pytest
```

## 下一步

- 接入 EPICS 模拟 IOC，把生成器替换为真实/准实时的数据源；
- 在其上训练 AE / LSTM-VAE / Anomaly Transformer，用 `type_label` 做逐类评估。
