# 多模态高光谱植物表型获取系统

> **Multimodal Hyperspectral Plant Phenotyping**
>
> 对应揭榜挂帅题目 **SY-202605《多模态高光谱成像在植物表型中的应用探索》** 的工程实现。
>
> 当前阶段数据仍以 **公开 / 合成数据** 跑通全链路，重心在 **工程实现**；已内置真实数据摄取
> （ENVI / GeoTIFF + 点云 + 表型表），以及一套可部署的演示系统。

---

## 这是什么

这套系统把「采集 → 预处理 → 配准 → 反射率校正 → 特征提取 → 表型反演 → 表型输出」做成
**可插拔的 stage**，硬件无关。等真实数据到位，只需换 `loaders/` 里的适配器，算法链路不用重写。

仓库内包含：

- 命令行端到端流水线（`main.py` / `pipeline.py`）
- Streamlit 交互式演示系统（`app.py`，侧边栏可切换数据源 / 模型 / 校正项）
- 真实数据摄取（ENVI / GeoTIFF 高光谱 + 点云 + 表型标签表）
- 公开数据集适配器（GreenHyperSpectra）
- 表型反演模型（Ridge / RandomForest / GradientBoosting / PLS）
- 变量施肥处方图与建议报告生成器
- 评估指标（R² / RMSE / 配准误差）
- Docker 一键部署

---

## 如何获取

```bash
git clone https://github.com/15934110986pmq-debug/multimodal-hyperspectral-phenotyping.git
cd multimodal-hyperspectral-phenotyping
```

## 环境要求

- **Git**
- **Python 3.9+**（推荐 3.10–3.12；Docker 镜像基于 3.12）
- 其余依赖见 `requirements.txt`（网络、探地等数据处理相关系统库见 Dockerfile）

## 一键安装（推荐）

不想手动敲命令？直接跑引导脚本，它会自动：

1. 检查 Python 版本（需 3.9+）
2. 创建或复用虚拟环境 `.venv`
3. 逐项检测依赖，缺失 / 版本不符时自动安装
4. 自检（模块导入 + 端到端合成数据流水线）
5. 打印启动指引

**Windows：** 双击 `install.bat`

**Linux / macOS：**

```bash
./install.sh
```

**跨平台通用（手动调用引导器）：**

```bash
python bootstrap.py                  # 检测 + 补全依赖 + 自检 + 启动指引
python bootstrap.py --run            # 自检通过后直接启动演示
python bootstrap.py --with-optional  # 一并安装公开数据集可选依赖
python bootstrap.py --quick          # 自检只做导入检查，更快
```

> 机器上还没装 Python 的话，先去 https://www.python.org/ 装 3.9+，安装时勾选
> **Add Python to PATH**。

## 安装

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

> 想在联网环境下接入公开数据集 `GreenHyperSpectra`，可额外安装可选依赖：
> `pip install -r requirements-optional.txt`（需要 `datasets` / `huggingface_hub` / `pyarrow`）。

## 快速开始（无需真实数据）

**命令行流水线** —— 使用内置合成数据，离线即可运行：

```bash
python main.py --loader synthetic --model random_forest
```

**交互式演示** —— 打开 Streamlit 界面：

```bash
# Windows 一键脚本
scripts\run.ps1

# Linux / macOS
./scripts/run.sh

# 或直接
python -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`。侧边栏可切换数据源 / 模型 / 折数 / 倾角校正，页面包括
数据管理、实时采集、数据处理、配准校正、特征提取、表型分析、施肥建议、数据上传、系统设置。

## 接入自己的数据

`loaders/base.py` 定义了一致的加载器接口。实现一个 loader 返回统一的 `SampleSet`，即可接入任意
光谱-表型数据。

```bash
# 真实数据：ENVI/GeoTIFF + 点云 CSV/.npy + 表型 CSV
python main.py --loader real --data <你的数据目录> --model ridge

# 公开数据集 GreenHyperSpectra
python main.py --loader greenhyperspectra --data <本地数据集路径> --trait chlorophyll,LAI
```

> **注意**：本仓库**不包含**大体积真实数据（`data/uploads/` 等已在 `.gitignore` 中排除，`.venv/`、
> `output/` 也不入库）。克隆下来能用的是合成数据和 `data/` 下的示例文件。要复现真实场景实验，
> 请自行准备数据并按上面的 `--loader real` / `--data` 方式传入。

### `main.py` 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--loader` | `synthetic` / `real` / `greenhyperspectra` | `synthetic` |
| `--data` | `real` / `greenhyperspectra` 需要的数据路径 | — |
| `--model` | `ridge` / `random_forest` / `gradient_boosting` / `pls` | `random_forest` |
| `--trait` | 逗号分隔的表型列名 | — |
| `--n-samples` | 合成数据样本数 | `400` |
| `--cv` | 交叉验证折数 | `5` |
| `--n-features` | 特征数 | `8` |
| `--outdir` | 输出目录 | `output` |
| `--no-plots` | 关闭图件输出 | `False` |

## 变量施肥建议

『施肥建议』基于训练好的反演模型对场景逐像素预测叶绿素，结合冠层 NDVI 得到氮营养指数（NNI），
换算差异化施氮量与分区处方，输出处方图、分区处方表和自包含 HTML 报告（可下载）。无标签数据时
自动退化为「NDVI 指数法」。P/K 及微量元素需土壤速效养分化验，当前仅给氮处方。

```bash
python create_fertilizer_report.py --source synthetic --model pls --crop wheat
```

## 静态报告（离线备份）

生成与演示系统内容一致的 HTML 报告，图片内嵌、可离线打开：

```bash
python create_report.py --source synthetic --model ridge --out static_report.html
```

## 容器化部署

```bash
docker compose up --build
docker compose down
```

`docker-compose.yml` 把宿主机数据目录挂载为 `/data`。可用环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `MULTIMODE_SOURCE` | `synthetic` / `real` / `greenhyperspectra` | `real` |
| `MULTIMODE_DATA_ROOT` | 容器内数据目录 | `/data` |
| `MULTIMODE_MODEL` | `ridge` / `random_forest` / `gradient_boosting` / `pls` | `ridge` |
| `MULTIMODE_TRAITS` | 表型列，逗号分隔 | `chlorophyll,LAI,...` |
| `MULTIMODE_PORT` | 对外端口 | `8501` |
| `MULTIMODE_DATA_DIR` | 宿主机数据目录 | `./data` |

## 仓库结构

```text
data/schema.py                  统一内部数据模型
loaders/real.py                 真实数据摄取（ENVI/GeoTIFF + 点云 + 表型 CSV）
loaders/synthetic.py            合成数据（离线兜底，默认可用）
loaders/greenhyperspectra.py    公开数据集 GreenHyperSpectra 适配器
stages/base.py                  Stage 接口 + Pipeline
stages/preprocess.py            去噪 / 辐射校正 / 坏点校正
stages/co_registration.py       点云 <-> 高光谱配准（接口 + 合成验证）
stages/reflectance_correction.py 叶片倾角 / BRDF 反射率校正
stages/features.py              NDVI / 红边位置 / 光谱特征向量
stages/phenotyping.py           ML / DL 表型反演模型
stages/postprocess.py           表型输出 / 伪彩图 / 报表
stages/fertilizer.py            变量施肥处方
evaluate/metrics.py             指标核验（R² / RMSE / 配准误差等）
demo/plots.py                   可视化
demo/scene.py                   合成植物场景（点云 / RGB / 高程演示）
demo/fertilizer_report.py       施肥报告
app.py                          交互式演示系统（Streamlit）
create_report.py                静态报告生成器
create_fertilizer_report.py     变量施肥处方/建议报告生成器（离线）
tools/make_real_dataset.py      生成"真实格式"占位数据集
scripts/run.ps1 / run.sh        启动脚本
Dockerfile / docker-compose.yml  容器化部署
pipeline.py                     一键流水线
main.py                         端到端演示入口
```

## 常见问题

**Windows 控制台输出中文乱码**

```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
python main.py --loader synthetic --model random_forest
```

**数据为空 / 找不到数据** —— 用 `synthetic`（默认）验证链路；接入真实数据请用
`--loader real --data <目录>`，并确保目录内存在高光谱 + 点云 + 表型文件。

## 技术栈

Python · NumPy · SciPy · scikit-learn · pandas · tifffile · matplotlib · Streamlit · Plotly · Docker

## 说明

本仓库为课程设计 / 揭榜挂帅题目的开源工程实现，数据与模型仍在迭代。**目前暂未附加开源许可证**；
如需引用、二次开发或商用，请先联系作者。
