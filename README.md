# 多模态高光谱植物表型获取系统 (Multimodal Hyperspectral Plant Phenotyping)

对应揭榜挂帅题目 **SY-202605《多模态高光谱成像在植物表型中的应用探索》**。

> 当前阶段：数据尚在处理，先用**公开/合成数据**跑通流水线，重心在**工程实现**。
> 已提供**真实数据摄取链路**：能直接读 ENVI/GeoTIFF 高光谱 + 点云 + 表型表，
> 以及一套**可部署的演示系统**与**静态报告**备份。

## 定位

把硬件无关的软件骨架搭起来：定义统一内部数据模型，把「采集 → 预处理 → 配准 →
反射率校正 → 特征提取 → 表型反演 → 表型输出」做成可插拔的 stage，等真数据一到只需
换 `loaders/` 里的适配器，算法链路不需要重写。

## 仓库结构

```
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
evaluate/metrics.py             指标核验（R² / RMSE / 配准误差等）
demo/plots.py                   可视化
demo/scene.py                   合成植物场景（点云 / RGB / 高程演示）
app.py                          交互式演示系统（Streamlit）
create_report.py                静态报告生成器（与部署内容一致）
create_fertilizer_report.py     变量施肥处方/建议报告生成器（离线）
tools/make_real_dataset.py      生成"真实格式"占位数据集
scripts/run.ps1 / run.sh        启动脚本
deploy/Dockerfile / docker-compose.yml  容器化部署
pipeline.py                     一键流水线
main.py                         端到端演示入口
```

## 环境

```bash
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
```

## 快速运行

```bash
# 合成数据（无需联网）
.venv\\Scripts\\python main.py --loader synthetic --model random_forest

# 真实数据（ENVI/GeoTIFF + 点云 + 表型）
.venv\\Scripts\\python main.py --loader real --data D:\\multimode\\data --model ridge
```

## 交互式演示系统

```bash
scripts\\run.ps1        # Windows；Linux 用 scripts\\run.sh
# 或直接
.venv\\Scripts\\python -m streamlit run app.py
```

默认浏览器打开 `http://localhost:8501`。侧边栏可切换数据源 / 模型 / 折数 / 倾角校正，
九个页面对应技术图「软件界面」：数据管理、实时采集、数据处理、配准校正、特征提取、
表型分析、施肥建议、数据上传、系统设置。

### 数据上传

『数据上传』页支持：
  * **ZIP 数据包**：打包高光谱 + 点云 + 表型标签，系统自动解压并识别。
  * **多文件**：分别上传 ENVI(`.hdr`+`.img/.dat`) / GeoTIFF / `.npy` 高光谱、点云 `.csv`、
    表型标签 `.csv`，自动按列特征归类。
上传后加载进 `session_state`，其余页面（数据管理 / 施肥建议等）立即复用该真实数据。

### 施肥建议 / 变量施肥处方图

『施肥建议』页基于已训练的反演模型，对场景**逐像素**预测叶绿素，并结合冠层 NDVI 得到
氮营养指数（NNI），换算差异化施氮量（kg N/ha）与分区处方，输出：
  * 处方图（氮营养指数 / 施氮量 / 分区，三图）
  * 分区处方表（等级 / 施氮量 / 面积占比 / 需氮量）
  * 施肥建议文本 + 自包含 HTML 报告（可下载）

无标签数据时自动退化为"NDVI 指数法"。P/K 及微量元素需土壤速效养分化验，当前仅给氮处方。

```bash
# 命令行生成施肥报告（离线备份）
.venv\\Scripts\\python create_fertilizer_report.py --source real --data_root D:\\multimode\\data --model ridge --crop wheat
```

## 容器化部署（真正部署形态）

```bash
docker compose up --build
docker compose down
```

`docker-compose.yml` 把真实数据目录挂载为 `/data`，用环境变量指定数据源与模型。
可用变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `MULTIMODE_SOURCE` | `synthetic` / `real` / `greenhyperspectra` | `real` |
| `MULTIMODE_DATA_ROOT` | 数据目录 | `/data` |
| `MULTIMODE_MODEL` | `ridge` / `random_forest` / `gradient_boosting` / `pls` | `ridge` |
| `MULTIMODE_TRAITS` | 表型列，逗号分隔 | chlorophyll,LAI,... |
| `MULTIMODE_PORT` | 对外端口 | `8501` |
| `MULTIMODE_DATA_DIR` | 宿主机数据目录 | `./data` |

## 静态报告（离线备份）

生成与部署演示系统内容一致的 HTML 报告，图片内嵌、可离线打开：

```bash
.venv\\Scripts\\python create_report.py --source real --data_root D:\\multimode\\data --model ridge --out static_report.html
```

## 真实/公开数据接入

`loaders/base.py` 里定义了一致的加载器接口。实现一个 loader 返回统一的 `SampleSet`
就能接入任意光谱-表型数据。`loaders/real.py` 支持 ENVI(`.hdr`+`.img`/`.dat`)、
GeoTIFF、点云 CSV/`.npy`、表型 CSV；`loaders/greenhyperspectra.py` 接入公开数据集。

### 已上传的新真实数据

用户上传的真实高光谱数据已解压到 `data/uploads/`，演示系统默认使用其中的
`data/uploads/220728`（5 组 ENVI BIL 立方体：`1` / `DBZ1` / `DBZ1-1` / `DBZ2` / `TES1`，
200 波段、386–1022 nm、12-bit）。这些是原始反射率立方体，**不含** `points.csv` /
`labels.csv`，因此反演/施肥会自动退化为「冠层 NDVI 指数法」（见『表型分析』『施肥建议』页）。

在侧边栏「数据路径」填入下面的目录/文件即可切换：

* `data/uploads/220728` — 5 个真实场景（默认，推荐）
* `data/uploads/260316` — 大扫描条带 `2026_3_16_1_41_28.bil`（900×70330×150，约 19 GB）
  + 注意：该 `.bil` 目前读到的是近 0 / 稀疏数值（不像是反射率立方体），`_1_40_46.zip`
    内含其 `.hdr` / `.times` / `.bin`，字段与真实外观需进一步确认后才能用于建模。
* `data/uploads/0915pjj` — 早期上传的 `guanmu*` 场景残留（`.bil` + `.hdr`）

> 说明：当前沙箱仅能访问包镜像，无法直连 huggingface/zenodo 等外部仓库，
> 因此公开数据集需在本机联网下载后走 `--data <本地路径>`；真实硬件数据用 `real` 加载器。
