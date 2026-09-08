"""多模态高光谱植物表型获取系统 —— 交互式演示界面（控制面板风格）。
+
启动：.venv\\Scripts\\python -m streamlit run app.py
+
布局对齐"软件平台界面"参考图：
  左侧竖排导航  数据管理 / 实时采集 / 数据处理 / 配准校正 / 特征提取 / 表型分析 / 施肥建议 / 数据上传 / 系统设置
  上部四卡片    可见光图像 / 高光谱伪彩色 / 三维点云 / 反射率曲线
  下部左        测量结果      下部右        批量统计分析
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.schema import RawScene, SampleSet
from loaders import LOADERS
from loaders.synthetic import SyntheticLoader
from loaders.real import RealDatasetLoader
from pipeline import run_phenotyping
from demo.scene import make_synthetic_scene
from demo.fertilizer_report import make_fertilizer_report, build_recommendations
from stages.fertilizer import (
    FertilizationAdvisor,
    prescription_from_scene,
    scene_ndvi,
    predict_trait_map,
    build_prescription,
)


st.set_page_config(page_title="多模态高光谱植物表型获取系统", layout="wide", page_icon="🌱")


# --------------------------------------------------------------------------- theme / css
CSS = """
<style>
.stApp { background:#0b1220; }
[data-testid="stSidebar"] { background:#0e1830; border-right:1px solid #1e3350; }
[data-testid="stSidebar"] * { color:#cfe1f5; }
h1,h2,h3,h4 { color:#e6f2ff; }
.panel-title{ color:#7fd0ff; font-size:15px; font-weight:600; margin:2px 0 10px 0; letter-spacing:.5px;}
.metric-box{ background:linear-gradient(180deg,#0f2a4d,#0a1c38); border:1px solid #2a5a86; border-radius:10px;
             padding:16px 12px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,.25);}
.metric-label{ color:#8fb3d9; font-size:12px; margin-bottom:6px;}
.metric-val{ color:#3fe0a5; font-size:30px; font-weight:700; line-height:1;}
.metric-unit{ color:#79c7ff; font-size:12px; margin-left:2px;}
.rcard{ background:#0e1c38; border:1px solid #1e3350; border-radius:10px; padding:6px;}
.stButton>button{ background:#0f2a4d; color:#cfe1f5; border:1px solid #2a5a86;}
[data-testid="stMetricValue"]{ color:#3fe0a5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


UPLOAD_DIR = os.environ.get("MULTIMODE_UPLOAD_DIR", os.path.join("data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- helpers
def _nearest(wl: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wl - target)))


def _rgb(scene: RawScene, r=650.0, g=550.0, b=450.0,
         p_low: float = 2.0, p_high: float = 99.0) -> np.ndarray:
    """真彩色合成。对未定标 / 过曝的 DN 数据用逐通道稳健拉伸，避免整体发白。"""
    cube = np.nan_to_num(scene.cube, nan=0.0)
    rgb = np.stack([cube[:, :, _nearest(scene.wavelengths, r)],
                    cube[:, :, _nearest(scene.wavelengths, g)],
                    cube[:, :, _nearest(scene.wavelengths, b)]], axis=-1)
    lo = np.percentile(rgb, p_low, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, p_high, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb


def _false_color(scene: RawScene, index=800.0, low=660.0, high=550.0) -> np.ndarray:
    """冠层结构伪彩色。先把各波段做稳健拉伸，再去掉照明谱型的影响，
    使植被归一化差异（绿-青）能展示出来；对未定标 / 过曝数据更稳健。"""
    wl = scene.wavelengths
    cube = np.nan_to_num(scene.cube, nan=0.0)
    lo = np.percentile(cube, 2.0, axis=(0, 1), keepdims=True)
    hi = np.percentile(cube, 99.0, axis=(0, 1), keepdims=True)
    cn = np.clip((cube - lo) / (hi - lo + 1e-9), 0, 1)
    i_red = _nearest(wl, 660.0)
    i_nir = _nearest(wl, index)
    val = (cn[:, :, i_nir] - cn[:, :, i_red]) / (cn[:, :, i_nir] + cn[:, :, i_red] + 1e-8)
    val = np.clip(val, 0, 1)
    # 绿-青-蓝 伪彩映射（参考图的青绿色高亮）
    col = np.zeros((*val.shape, 3))
    col[:, :, 1] = val          # 绿
    col[:, :, 0] = val * 0.35   # 少量红 -> 偏青
    col[:, :, 2] = np.clip(1.0 - val * 0.35, 0, 1)
    return np.clip(col, 0, 1)


def _pointcloud_3d(scene: RawScene) -> go.Figure:
    pc = scene.pointcloud
    if pc.xyz.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="当前数据未提供点云（无 points.csv / .npy）",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#9fb8d6", size=13))
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=pc.xyz[:, 0], y=pc.xyz[:, 1], z=pc.xyz[:, 2], mode="markers",
            marker=dict(size=2.0, color=pc.xyz[:, 2], colorscale="Viridis",
                        showscale=False))])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb8d6"), margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"), height=260)
    return fig


def _reflectance_curve(scene: RawScene) -> go.Figure:
    cube = np.nan_to_num(scene.cube, nan=0.0)
    # 用全场景中位数谱，避免单点过曝导致反射率曲线被拉平失真。
    spec = np.median(cube.reshape(-1, cube.shape[2]), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scene.wavelengths, y=spec,
                             mode="lines", line=dict(color="#3fe0a5", width=2)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb8d6"), margin=dict(l=40, r=10, t=10, b=30),
                      xaxis=dict(title="波长 (nm)", gridcolor="#1e3350"),
                      yaxis=dict(title="反射率", gridcolor="#1e3350"), height=260)
    return fig


def _compute_ndvi(ss: SampleSet) -> float:
    i_red = _nearest(ss.wavelengths, 660.0)
    i_nir = _nearest(ss.wavelengths, 800.0)
    ndvi = (ss.X[:, i_nir] - ss.X[:, i_red]) / (ss.X[:, i_nir] + ss.X[:, i_red] + 1e-8)
    return float(np.mean(ndvi))


def _trait_mean(ss: SampleSet, name: str) -> Optional[float]:
    if ss.y is None or name not in (ss.trait_names or []):
        return None
    j = ss.trait_names.index(name)
    return float(np.mean(ss.y[:, j]))


def _heatmap(scene: RawScene, ss: SampleSet) -> go.Figure:
    """批量统计分析：将样本按五等分分组，统计各代表波长的平均反射率。"""
    wl = ss.wavelengths
    centers = [550, 650, 705, 750, 800, 850, 900, 950]
    centers = [c for c in centers if wl.min() <= c <= wl.max()]
    idx = [_nearest(wl, c) for c in centers]
    X = ss.X
    n_groups = 5
    order = np.argsort(X[:, _nearest(wl, 800)])  # 按近红外排序分组
    groups = np.array_split(order, n_groups)
    mat = np.array([[X[g, j].mean() for j in idx] for g in groups])
    fig = go.Figure(go.Heatmap(
        z=mat, x=[f"{c}nm" for c in centers], y=[f"组{i+1}" for i in range(n_groups)],
        colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in mat], texttemplate="%{text}",
        textfont=dict(color="white", size=11)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb8d6"), margin=dict(l=20, r=10, t=10, b=10),
                      coloraxis_showscale=False, height=250,
                      xaxis=dict(gridcolor="#1e3350"), yaxis=dict(gridcolor="#1e3350"))
    return fig


def _metric_box(label: str, value: str, unit: str = "", tag: str = "") -> None:
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}'
        f'{" <span style=\"color:#79c7ff;font-weight:600\">" + tag + "</span>" if tag else ""}</div>'
        f'<div class="metric-val">{value}<span class="metric-unit">{unit}</span></div></div>',
        unsafe_allow_html=True,
    )


def _card(title: str) -> None:
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- inversion (train -> predict)
@st.cache_resource(show_spinner=False)
def _build_predictor(train_root: str, trait_csv: str, model_type: str, cv_folds: int) -> Dict:
    """在带标签的训练数据上训练表型反演模型，返回可复用的 predictor。

    predictor 内含已 fitted 的 ``feature_extractor``（记录训练波长网格 / 光谱均值 /
    PCA 基）与回归模型，可对任意波段的重采样新场景做逐像素反演。
    """
    trait_list = [t.strip() for t in trait_csv.split(",") if t.strip()]
    if not train_root or not os.path.isdir(train_root):
        raise ValueError(f"训练数据路径不存在：{train_root!r}")
    loader = RealDatasetLoader(
        root=train_root, trait_columns=trait_list, max_pixels=400000, max_samples=20000,
    )
    ts = loader.load().samples
    if ts.y is None:
        raise ValueError("训练数据没有表型标签（未找到 labels.csv 或指定表型列）")
    res = run_phenotyping(ts, model_type=model_type, cv_folds=cv_folds)
    # 记录训练集的特征分布（均值/标准差）与表型实测范围，用于域一致性判断与结果约束
    feat_mat = res["data"].X
    y = ts.y
    return {
        "model": res["model"],
        "feature_extractor": res["feature_extractor"],
        "trait_names": list(ts.trait_names or []),
        "metrics": res["metrics"],
        "feature_names": list(res["feature_names"]),
        "train_feat_mean": np.asarray(feat_mat.mean(axis=0), dtype=float),
        "train_feat_std": np.asarray(feat_mat.std(axis=0), dtype=float),
        "label_range": {
            name: (float(y[:, j].min()), float(y[:, j].max()))
            for j, name in enumerate(ts.trait_names or [])
        },
        "train_root": train_root,
        "train_wavelengths": np.asarray(ts.wavelengths, float),
        "n_bands": ts.n_bands,
    }


def _trait_clamp(predictor: Dict, name: str, value: float) -> float:
    """把反演值约束到训练集实测范围，兼顾物理范围，避免跨域外推出离谱数。"""
    lo, hi = predictor["label_range"].get(name, (None, None))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return _physical(name, value)


def _domain_ok(predictor: Dict, ss: SampleSet, z_limit: float = 3.0) -> bool:
    """检查场景的统计特征是否落在训练特征分布内（±z_limit 个标准差）。"""
    feats = predictor["feature_names"]
    keys = ["NDVI", "NDRE", "GNDVI", "MeanRefl"]
    idx = [feats.index(k) for k in keys if k in feats]
    if not idx:
        return True
    F = predictor["feature_extractor"].transform(
        ss.X.astype(float), np.asarray(ss.wavelengths, float))
    scene_mean = F[:, idx].mean(axis=0)
    z = (scene_mean - predictor["train_feat_mean"][idx]) / (
        predictor["train_feat_std"][idx] + 1e-9)
    return bool(np.all(np.abs(z) < z_limit))


def _predict_trait_means(predictor: Dict, ss: SampleSet) -> Dict[str, float]:
    """用训练好的 predictor 对样本集逐样本反演，返回各表型均值。"""
    feat = predictor["feature_extractor"].transform(
        ss.X.astype(float), np.asarray(ss.wavelengths, float)
    )
    pred = np.asarray(predictor["model"].predict(feat), dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return {
        name: _trait_clamp(predictor, name, float(pred[:, j].mean()))
        for j, name in enumerate(predictor["trait_names"])
    }


def _bands_match(predictor: Dict, ss: SampleSet, atol: float = 0.6) -> bool:
    """训练波段与当前场景波段是否一致（一致则无需重采样，反演更可靠）。"""
    tw = predictor["train_wavelengths"]
    sw = np.asarray(ss.wavelengths, float)
    return tw.size == sw.size and bool(np.allclose(tw, sw, atol=atol))


def _physical(name: str, value: float) -> float:
    """把反演值约束到物理合理范围（仅用于展示，真实精度看 R^2 与分布）。"""
    if name in ("LAI", "biomass", "chlorophyll", "soluble_solids"):
        return max(0.0, value)
    if name in ("water_content",):
        return float(np.clip(value, 0.0, 100.0))
    return float(value)


def _trait_map_fig(mp: np.ndarray, title: str, unit: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=mp, colorscale="Viridis", coloraxis="coloraxis"))
    fig.update_layout(coloraxis=dict(colorscale="Viridis"), height=280, title=f"{title} {unit}".strip(),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb8d6"), margin=dict(l=0, r=0, t=36, b=0))
    return fig


def _measure_value(ss: SampleSet, name: str, pred_means: Optional[Dict[str, float]],
                   domain_ok: bool = True) -> Tuple[Optional[float], str]:
    """优先读实测标签；无标签时用反演模型预测值。域不一致时不给值、只给标注。"""
    if ss.y is not None and name in (ss.trait_names or []):
        j = ss.trait_names.index(name)
        return float(np.mean(ss.y[:, j])), ""
    if pred_means is not None and name in pred_means:
        if not domain_ok:
            return None, "跨域·不可靠"
        return pred_means[name], "反演"
    return None, ""


def _render_inversion_details(predictor: Dict, ss: SampleSet) -> None:
    """反演预测详情：模型来源、波段适配、质量指标与预测分布/图。"""
    from stages.fertilizer import predict_trait_map

    metrics = predictor["metrics"]
    domain_ok = _domain_ok(predictor, ss)
    st.write(f"**训练数据**：{predictor['train_root']}　**模型**：{metrics.get('model_type')}　"
             f"**交叉验证**：{metrics.get('cv_folds')}-fold")
    st.write(f"**训练波段**：{predictor['n_bands']} 个（{predictor['train_wavelengths'][0]:.0f}–"
             f"{predictor['train_wavelengths'][-1]:.0f} nm）　**当前场景波段**：{ss.n_bands} 个")

    bands_ok = _bands_match(predictor, ss)
    if not bands_ok:
        st.warning("训练波段与当前场景不一致，已把新场景光谱按训练波长重采样后再反演。"
                   "跨波段/跨传感器域反演可能存在较大误差，仅供趋势参考。")

    if not domain_ok:
        feats = predictor["feature_names"]
        keys = ["NDVI", "NDRE", "GNDVI", "MeanRefl"]
        idx = [feats.index(k) for k in keys if k in feats]
        F = predictor["feature_extractor"].transform(
            ss.X.astype(float), np.asarray(ss.wavelengths, float))
        rows = []
        for i in idx:
            z = (F[:, i].mean() - predictor["train_feat_mean"][i]) / (
                predictor["train_feat_std"][i] + 1e-9)
            rows.append({
                "特征": feats[i],
                "训练集均值": f"{predictor['train_feat_mean'][i]:.3f}",
                "当前场景均值": f"{F[:, i].mean():.3f}",
                "偏离(σ)": f"{z:+.1f}",
            })
        st.error("⚠️ 域不一致：当前场景的统计特征已明显偏离训练集分布（见下表）。"
                 "此时模型只能外推，结果不可信，因此「测量结果」不给反演数值。"
                 "要得到可靠反演，请用与当前场景同一传感器、相近作物/生育期的带标签数据重新训练。")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.success("域一致性通过：当前场景光谱处于训练集分布内，反演结果可作为参考。")

    # 训练集内 CV 质量指标（域内精度，不等于跨域精度）
    rows = []
    for name, r2, rmse in zip(metrics.get("trait_names", []), metrics.get("r2", []), metrics.get("rmse", [])):
        rows.append({"表型": name, "R²(训练CV)": f"{r2:.3f}", "RMSE": f"{rmse:.3f}"})
    if rows:
        st.markdown("**训练集内交叉验证精度（跨域仅供参考）**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if domain_ok:
        # 当前场景预测分布（仅在域一致时展示，避免给出误导性数值）
        feat = predictor["feature_extractor"].transform(
            ss.X.astype(float), np.asarray(ss.wavelengths, float))
        pred = np.asarray(predictor["model"].predict(feat), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        body = []
        for j, name in enumerate(predictor["trait_names"]):
            if name in ("water_content", "LAI", "biomass"):
                body.append([name, f"{_trait_clamp(predictor, name, float(pred[:, j].mean())):.2f}",
                             f"{float(pred[:, j].min()):.1f}", f"{float(pred[:, j].max()):.1f}"])
        if body:
            st.markdown("**当前场景反演结果（均值/最小/最大，已约束到训练实测范围）**")
            st.dataframe(pd.DataFrame(body, columns=["表型", "均值", "最小值", "最大值"]),
                         width="stretch", hide_index=True)

        # 逐像素预测图（使用与训练一致的特征基）
        st.markdown("**逐像素反演分布图**")
        sc = scene
        c1, c2 = st.columns(2)
        for col, name in ((c1, "biomass"), (c2, "LAI")):
            if name not in predictor["trait_names"]:
                continue
            j = predictor["trait_names"].index(name)
            try:
                mp = predict_trait_map(sc, predictor["model"], predictor["feature_extractor"], j)
            except Exception as exc:  # noqa: BLE001
                col.info(f"{name} 图生成失败：{exc}")
                continue
            unit = "(g)" if name == "biomass" else "(m²/m²)"
            with col:
                st.plotly_chart(_trait_map_fig(mp, f"预测 {name}", unit), width="stretch")


# --------------------------------------------------------------------------- data source
@st.cache_resource
def load_source(source: str, data_path: str, trait: str, n_samples: int):
    trait_list = [t.strip() for t in trait.split(",") if t.strip()]
    if source == "synthetic":
        return make_synthetic_scene(), SyntheticLoader(n_samples=n_samples).load()
    if source == "real":
        rs = LOADERS["real"](root=data_path, trait_columns=trait_list,
                             max_pixels=300000, max_samples=8000).load()
        return rs.scene, rs.samples
    if source == "greenhyperspectra":
        ss = LOADERS["greenhyperspectra"](path=data_path, trait_columns=trait_list).load()
        return make_synthetic_scene(), ss
    raise ValueError(source)


def _load_uploaded(cube_path, points_path, labels_path, trait, max_pixels, max_samples):
    """把上传到磁盘的真实数据经 RealDatasetLoader 转成 scene + ss。"""
    trait_list = [t.strip() for t in trait.split(",") if t.strip()]
    loader = RealDatasetLoader(
        cube_path=cube_path, points_path=points_path, labels_path=labels_path,
        trait_columns=trait_list, max_pixels=max_pixels, max_samples=max_samples,
    )
    rs = loader.load()
    return rs.scene, rs.samples


def _guess_csv_role(name: str) -> str:
    """根据文件名与列断定 CSV 是点云还是表型标签。"""
    import pandas as pd
    low = name.lower()
    if "point" in low or "pc" in low or "cloud" in low:
        return "points"
    if "label" in low or "trait" in low or "pheno" in low:
        return "labels"
    return "unknown"


def _save_upload(file_obj, dest_dir: str) -> str:
    dest = os.path.join(dest_dir, file_obj.name)
    with open(dest, "wb") as fh:
        fh.write(file_obj.getbuffer())
    return dest


def _default_data_path() -> str:
    """默认真实数据目录：优先用户新上传的真实高光谱数据，其次旧占位数据集。"""
    for cand in ("data/uploads/0915pjj", "data/uploads/220728", "data/uploads/260316", "data"):
        if os.path.isdir(cand):
            return cand
    return ""


# --------------------------------------------------------------------------- sidebar
NAV = ["数据管理", "实时采集", "数据处理", "配准校正", "特征提取",
       "表型分析", "施肥建议", "数据上传", "系统设置"]
st.sidebar.title("⚙️ 软件平台")
page = st.sidebar.radio("导航", NAV, index=0, label_visibility="collapsed")
st.sidebar.markdown("---")

SOURCES = ["synthetic", "real", "greenhyperspectra"]
MODELS = ["ridge", "random_forest", "gradient_boosting", "pls"]
source = st.sidebar.selectbox("数据源", SOURCES,
                              index=SOURCES.index(_env("MULTIMODE_SOURCE", "real"))
                              if _env("MULTIMODE_SOURCE", "real") in SOURCES else 0)
data_path = _env("MULTIMODE_DATA_ROOT") or _default_data_path()
trait = _env("MULTIMODE_TRAITS", "chlorophyll,LAI,water_content,biomass,soluble_solids")
if source != "synthetic":
    data_path = st.sidebar.text_input("数据路径（目录/文件）", data_path)
    trait = st.sidebar.text_input("表型列 (逗号分隔)", trait)
n_samples = st.sidebar.slider("样本数", 100, 1500, 600, step=100)
model_type = st.sidebar.selectbox("反演模型", MODELS,
                                  index=MODELS.index(_env("MULTIMODE_MODEL", "ridge"))
                                  if _env("MULTIMODE_MODEL", "ridge") in MODELS else 0)
cv_folds = st.sidebar.slider("交叉验证折数", 3, 10, 5)
train_root = _env("MULTIMODE_TRAIN_ROOT") or os.path.join("data")
if source != "synthetic":
    train_root = st.sidebar.text_input("反演训练数据路径（含 labels.csv）", train_root)
use_prediction = st.sidebar.checkbox(
    "无标签数据用模型反演预测（推荐）", value=True,
    help="当前场景缺少表型标签时，用『反演训练数据路径』里的带标签数据训练模型，"
         "对当前场景逐像素反演 LAI / 含水量 / 生物量等。",
)

source_actual = source
if "upload_scene" in st.session_state and "upload_ss" in st.session_state:
    scene = st.session_state["upload_scene"]
    ss = st.session_state["upload_ss"]
    source_actual = "upload"
else:
    if source != "synthetic" and not data_path:
        st.sidebar.error("该数据源需要提供数据路径，或前往『数据上传』页上传数据")
        st.stop()
    try:
        scene, ss = load_source(source, data_path, trait, n_samples)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"数据加载失败：{exc}")
        st.stop()
try:
    ss.validate()
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"数据校验失败：{exc}")
    st.stop()

# 无标签场景 -> 用带标签训练数据训练反演模型并预测（缓存，仅在需要时训练一次）
predictor: Optional[Dict] = None
if use_prediction and ss.y is None:
    try:
        with st.spinner("正在用训练数据构建表型反演模型并预测（首次稍慢，之后用缓存）..."):
            predictor = _build_predictor(train_root, trait, model_type, cv_folds)
    except Exception as exc:  # noqa: BLE001
        predictor = None
        st.sidebar.warning(f"未能构建反演预测模型：{exc}")


# --------------------------------------------------------------------------- pages
st.markdown("## 🌱 多模态高光谱植物表型获取系统", unsafe_allow_html=False)
st.caption("三维结构 + 高光谱同步成像 · 高精度配准 · 反射率校正 · 表型智能解析")


def page_overview():
    """『数据管理』总览：四卡片 + 测量结果 + 批量统计分析。"""
    cols = st.columns(4)
    with cols[0]:
        _card("可见光图像")
        st.image(_rgb(scene), width="stretch")
    with cols[1]:
        _card("高光谱伪彩色")
        st.image(_false_color(scene), width="stretch")
    with cols[2]:
        _card("三维点云")
        st.plotly_chart(_pointcloud_3d(scene), width="stretch")
    with cols[3]:
        _card("反射率曲线")
        st.plotly_chart(_reflectance_curve(scene), width="stretch")

    st.write("")
    left, right = st.columns([1, 1.3])
    with left:
        _card("测量结果")
        pred_means = None
        domain_ok = True
        if predictor is not None and ss.y is None:
            pred_means = _predict_trait_means(predictor, ss)
            domain_ok = _domain_ok(predictor, ss)
        m = st.columns(4)
        with m[0]:
            v, tag = _measure_value(ss, "LAI", pred_means, domain_ok)
            _metric_box("叶面积指数 (m²/m²)", ("—" if v is None else f"{v:.2f}"), tag=tag)
        with m[1]:
            v, tag = _measure_value(ss, "water_content", pred_means, domain_ok)
            _metric_box("含水量 (%)", ("—" if v is None else f"{v:.2f}"), tag=tag)
        with m[2]:
            v, tag = _measure_value(ss, "biomass", pred_means, domain_ok)
            _metric_box("生物量 (g)", ("—" if v is None else f"{v:.2f}"), tag=tag)
        with m[3]:
            _metric_box("NDVI", f"{_compute_ndvi(ss):.2f}")
        if pred_means is not None:
            if not domain_ok:
                st.info("当前场景与训练域不一致，反演结果不可信，故未给出数值；"
                        "原因与对策见下方「反演预测详情」。")
            else:
                st.caption("带「反演」标注的为模型预测值；未标注的为实测标签。")
    with right:
        _card("批量统计分析")
        st.plotly_chart(_heatmap(scene, ss), width="stretch")

    if predictor is not None and ss.y is None:
        with st.expander("🔬 反演预测详情（模型来源 / 波段适配 / 精度 / 分布图）", expanded=False):
            _render_inversion_details(predictor, ss)


def page_acquisition():
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", ss.n_samples)
    c2.metric("光谱波段", ss.n_bands)
    c3.metric("表型数", ss.y.shape[1] if ss.y is not None else 0)
    with st.container(border=True):
        _card("反射率光谱")
        idx = np.linspace(0, ss.n_samples - 1, min(8, ss.n_samples)).astype(int)
        fig = go.Figure([go.Scatter(x=ss.wavelengths, y=ss.X[i], mode="lines", name=str(i))
                         for i in idx])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9fb8d6"), height=380, template=None)
        st.plotly_chart(fig, width="stretch")


def page_processing():
    from stages.preprocess import BadPixelCorrection, Denoise, ReflectanceCorrection
    sc = scene
    sc = Denoise(window=9).run(BadPixelCorrection().run(sc))
    sc = ReflectanceCorrection().run(sc)
    c1, c2 = st.columns(2)
    with c1:
        _card("校正后 RGB")
        st.image(_rgb(sc), width="stretch")
        st.metric("坏点校正数量", sc.meta.get("bad_pixel_corrected", 0))
    with c2:
        pre = np.nan_to_num(scene.cube).mean(axis=(0, 1))
        post = np.nan_to_num(sc.cube).mean(axis=(0, 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scene.wavelengths, y=pre, mode="lines", name="校正前"))
        fig.add_trace(go.Scatter(x=scene.wavelengths, y=post, mode="lines", name="校正后"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9fb8d6"), height=360)
        st.plotly_chart(fig, width="stretch")


def page_registration():
    from stages.reflectance_correction import cosine_correction
    st.markdown("**配准 / 反射率校正**")
    st.info("当前为演示兜底实现。接入真实硬件数据后，注入共光路内参与视角几何进行 BRDF 校正。")
    with st.container(border=True):
        _card("三维点云（按高程着色）")
        st.plotly_chart(_pointcloud_3d(scene), width="stretch")


def page_features():
    from stages.features import SpectrumFeatures
    feat = SpectrumFeatures(n_components=8)
    feat_set = feat.run(ss)
    st.markdown("**工程光谱特征**（" + str(len(feat.feature_names)) + " 个）")
    st.write(", ".join(feat.feature_names))
    m = feat_set.X[: min(30, feat_set.n_samples)]
    fig = go.Figure(go.Heatmap(z=m, x=list(range(m.shape[1])), colorscale="Viridis",
                               coloraxis="coloraxis"))
    fig.update_layout(coloraxis=dict(colorscale="Viridis"), height=360,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb8d6"))
    st.plotly_chart(fig, width="stretch")


def page_phenotyping():
    if ss.y is None:
        st.info("当前数据没有表型标签，无法进行反演建模。请通过『数据上传』页上传带标签的 CSV。")
        st.write("**已加载数据**")
        c1, c2, c3 = st.columns(3)
        c1.metric("样本数", ss.n_samples)
        c2.metric("光谱波段", ss.n_bands)
        c3.metric("表型数", 0)
        return
    res = run_phenotyping(ss, model_type=model_type, cv_folds=cv_folds)
    metrics = res["metrics"]
    st.code(res["report"])
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=metrics["trait_names"], y=metrics["r2"], marker_color="#3fe0a5"))
        fig.add_hline(y=0.9, line=dict(color="red", dash="dash"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9fb8d6"), yaxis_range=[0, 1.05], height=360)
        st.plotly_chart(fig, width="stretch")
    with c2:
        sel = st.selectbox("选择表型", metrics["trait_names"])
        j = metrics["trait_names"].index(sel)
        y = ss.y[:, j]; pred = res["y_pred"][:, j]
        fig = go.Figure(go.Scatter(x=y, y=pred, mode="markers",
                                   marker=dict(size=6, opacity=0.6, color="#3fe0a5")))
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="white", dash="dash")))
        r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        fig.update_layout(title=f"{sel}  R^2={r2:.3f}",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9fb8d6"), height=360)
        st.plotly_chart(fig, width="stretch")


def page_upload():
    """『数据上传』：上传真实数据（ZIP 或 多文件）并加载到系统。"""
    st.markdown("## 📤 数据上传")
    st.caption("支持上传 ENVI(.hdr+.img/.dat) / GeoTIFF / .npy 高光谱，点云 CSV，表型标签 CSV；"
               "也可上传整个数据目录的 ZIP 压缩包。")

    mode = st.radio("上传方式", ["ZIP 数据包", "多文件"], horizontal=True)
    trait_input = st.text_input("表型列 (逗号分隔，可选)", "chlorophyll,LAI,water_content,biomass,soluble_solids")
    max_pixels = st.slider("场景像素上限 (用于显示/分析)", 20000, 500000, 200000, 10000)
    max_samples = st.slider("样本抽样上限", 1000, 20000, 6000, 500)

    if mode == "ZIP 数据包":
        st.info("ZIP 内建议结构：高层光谱文件、points.csv、labels.csv 平铺或放在子目录均可，"
                "系统会自动定位。")
        zip_file = st.file_uploader("选择 ZIP 压缩包", type=["zip"])
        if zip_file is not None:
            out_dir = _upload_workdir()
            zip_path = os.path.join(out_dir, zip_file.name)
            with open(zip_path, "wb") as fh:
                fh.write(zip_file.getbuffer())
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            root = out_dir
        else:
            root = out_dir = None
        cube_path = _find_cube(root)
        points_path = _find_points(root)
        labels_path = _find_labels(root, skip=points_path)
        ready = root is not None and cube_path is not None

    else:
        st.info("逐个文件上传：高光谱数据（.hdr/.img/.tif/.npy，ENVI 需同时上传 .hdr 与对应二进制文件）、"
                "点云（points.csv）、标签（labels.csv）。")
        files = st.file_uploader("选择文件", accept_multiple_files=True,
                                 type=["hdr", "bil", "img", "dat", "bsq", "bip",
                                       "tif", "tiff", "npy", "csv"])
        if files:
            out_dir = _upload_workdir()
            for f in files:
                _save_upload(f, out_dir)
            cube_path = _find_cube(out_dir)
            points_path = _find_points(out_dir)
            labels_path = _find_labels(out_dir, skip=points_path)
        else:
            cube_path = points_path = labels_path = out_dir = None
        ready = cube_path is not None

    if not ready:
        st.info("请先上传数据。系统将自动识别高光谱立方体与可选的点云、标签文件。")
        return

    st.write("**已识别文件**")
    st.write({"高光谱": cube_path, "点云": points_path, "标签": labels_path})

    if st.button("加载上传数据", type="primary"):
        try:
            up_scene, up_ss = _load_uploaded(
                cube_path, points_path, labels_path, trait_input, max_pixels, max_samples)
        except Exception as exc:  # noqa: BLE001
            st.error(f"上传数据加载失败：{exc}")
            st.stop()
        up_ss.validate()
        st.session_state["upload_scene"] = up_scene
        st.session_state["upload_ss"] = up_ss
        st.session_state["upload_root"] = out_dir
        st.success(f"加载成功：{up_ss.n_samples} 样本 / {up_ss.n_bands} 波段 / "
                   f"场景 {up_scene.cube.shape[0]}x{up_scene.cube.shape[1]} / "
                   f"{len(up_ss.trait_names or [])} 表型。已切换到『数据管理』页。")
        st.rerun()

    if "upload_scene" in st.session_state and st.button("🗑️ 清除当前上传数据并恢复侧边栏数据源"):
        for k in ("upload_scene", "upload_ss", "upload_root", "upload_session_dir"):
            st.session_state.pop(k, None)
        st.rerun()


def _upload_workdir() -> str:
    """返回本次上传会话的写入目录（按会话稳定，避免每次 rerun 新建）。"""
    key = st.session_state.get("upload_session_dir")
    if key is None:
        sid = uuid.uuid4().hex[:10]
        d = os.path.join(UPLOAD_DIR, sid)
        os.makedirs(d, exist_ok=True)
        st.session_state["upload_session_dir"] = d
        key = d
    return key


def page_fertilizer():
    """『施肥建议』：训练反演模型，逐像素预测氮状态，生成变量施肥处方图与建议报告。"""
    st.markdown("## 🧪 施肥建议 / 变量施肥处方图")
    st.caption("以叶绿素(氮状态)为主、冠层长势为辅，逐像素给出差异化施氮量（kg N/ha），"
               "并生成处方图与分区建议报告。")

    crop = st.text_input("作物类型", "field_crop")
    with st.expander("施肥参数", expanded=False):
        n_max = st.number_input("最高施氮量 (kg N/ha)", 20.0, 500.0, 200.0, 10.0)
        n_min = st.number_input("最低施氮量 (kg N/ha)", 0.0, n_max, 0.0, 5.0)
        chl_low = st.number_input("叶绿素缺乏阈值", 5.0, 50.0, 15.0, 1.0)
        chl_high = st.number_input("叶绿素充足阈值", chl_low, 100.0, 55.0, 1.0)
        w_chl = st.slider("叶绿素权重", 0.0, 1.0, 0.7, 0.05)
        w_ndvi = round(1.0 - w_chl, 2)
        pixel_mm = st.slider("像素地面尺寸 (mm)", 1, 200, 50, 1)
        n_zones = st.slider("处方等级数", 2, 8, 5)

    advisor = FertilizationAdvisor(
        crop=crop, chl_low=chl_low, chl_high=chl_high,
        n_min=n_min, n_max=n_max,
        chl_weight=w_chl, ndvi_weight=w_ndvi,
        pixel_size_m=pixel_mm / 1000.0, n_zones=n_zones,
    )

    # 场景与训练样本波长/波段一致性：不一致时逐像素 ML 预测的 PCA 基会错位，
    # 稳妥做法是退回 NDVI 指数法（只依赖场景立方体本身）。
    bands_match = (len(scene.wavelengths) == ss.n_bands
                   and np.allclose(np.asarray(scene.wavelengths, float),
                                   np.asarray(ss.wavelengths, float), atol=0.6))

    # 训练存在性检查：无标签时退回纯 NDVI 法
    if (not bands_match or ss.y is None
            or (ss.trait_names and "chlorophyll" not in ss.trait_names
                and "LAI" not in ss.trait_names)):
        if not bands_match:
            st.warning("场景与训练样本的波长/波段不一致，改用冠层 NDVI 直接推定氮状态（纯指数法）。")
        else:
            st.warning("当前数据无表型标签或缺少叶绿素/LAI，改用冠层 NDVI 直接推定氮状态（纯指数法）。")
        ndvi_map = scene_ndvi(scene)
        presc = build_prescription(ndvi_map, ndvi_map, advisor)
        presc["pixel_size_m"] = advisor.pixel_size_m
    else:
        with st.spinner("正在训练反演模型并逐像素预测（首次可能需数十秒）..."):
            try:
                res = run_phenotyping(ss, model_type=model_type, cv_folds=cv_folds)
            except Exception as exc:  # noqa: BLE001
                st.error(f"建模失败：{exc}")
                return
            model = res["model"]
            feat_extractor = res["feature_extractor"]
            trait_names = ss.trait_names or []
            proxy = "chlorophyll" if "chlorophyll" in trait_names else trait_names[0]
            trait_index = trait_names.index(proxy)
            presc = prescription_from_scene(scene, model, feat_extractor, advisor, trait_index)
            presc["pixel_size_m"] = advisor.pixel_size_m

    # 先把处方图 PNG 落盘（供下载 & 报告内嵌）
    from pathlib import Path
    out_dir = os.path.join("output", "fertilizer")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "fertilizer_prescription.png")
    html_path = os.path.join(out_dir, "fertilizer_report.html")
    csv_path = os.path.join(out_dir, "fertilizer_zones.csv")
    from demo.plots import plot_prescription_map
    plot_prescription_map(presc, Path(png_path), advisor.pixel_size_m)

    st.write("")
    left, right = st.columns([1.4, 1])
    with left:
        _card("变量施肥处方图")
        st.image(png_path, width="stretch")
    with right:
        _card("处方总览")
        _metric_box("平均施氮量 (kg N/ha)", f"{presc['mean_rate']:.1f}")
        st.write("")
        _metric_box("合计需氮 (kg N)", f"{presc['total_n_kg']:.1f}")
        st.write("")
        _metric_box("处方面积 (m²)", f"{presc['total_area_m2']:.1f}")
        st.write("")
        _metric_box("平均氮营养指数", f"{presc['mean_status']:.2f}")

    st.write("")
    _card("分区处方表")
    import pandas as pd
    zone_df = pd.DataFrame(presc["zone_table"])
    if zone_df.empty:
        st.info("当前参数下没有产生有效分区。")
    else:
        st.dataframe(zone_df, width="stretch", hide_index=True)

    st.write("")
    _card("施肥建议")
    rec_text = build_recommendations(presc, crop)
    st.markdown(rec_text)

    # 文件下载
    zone_df.to_csv(csv_path, index=False)
    st.download_button("⬇️ 下载处方图 PNG",
                       data=open(png_path, "rb").read(),
                       file_name="fertilizer_prescription.png", mime="image/png")
    st.download_button("⬇️ 下载施肥建议报告 (HTML)",
                       data=_render_fertilizer_html(presc, crop, png_path),
                       file_name="fertilizer_report.html", mime="text/html")
    st.download_button("⬇️ 下载分区处方表 CSV",
                       data=zone_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="fertilizer_zones.csv", mime="text/csv")

    if st.button("💾 保存处方图到 output", type="secondary"):
        from demo.fertilizer_report import make_fertilizer_report
        out = make_fertilizer_report(
            source=source_actual, crop=crop, prescription=presc,
            out_png=__import__("pathlib").Path(png_path),
            out_html=__import__("pathlib").Path(html_path),
        )
        st.success(f"已保存：{out['png']} / {out['html']}")


def _find_cube(root: Optional[str]) -> Optional[str]:
    if not root:
        return None
    import glob
    for pat in ("*.hdr", "*.bil", "*.bsq", "*.bip", "*.tif", "*.tiff", "*.npy",
                "*.img", "*.dat"):
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None


def _find_points(root: Optional[str]) -> Optional[str]:
    if not root:
        return None
    import glob
    hits = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    for h in hits:
        if _guess_csv_role(h) == "points":
            return h
    return None


def _find_labels(root: Optional[str], skip: Optional[str] = None) -> Optional[str]:
    if not root:
        return None
    import glob
    hits = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    for h in hits:
        if skip and os.path.abspath(h) == os.path.abspath(skip):
            continue
        if _guess_csv_role(h) == "labels":
            return h
    # 兜底：没有特征列名的 csv（含 sample_id 或 numeric 首列）
    for h in hits:
        if skip and os.path.abspath(h) == os.path.abspath(skip):
            continue
        import pandas as pd
        try:
            cols = pd.read_csv(h, nrows=0).columns.tolist()
        except Exception:  # noqa: BLE001
            continue
        low = [c.lower() for c in cols]
        # 跳过三列 x,y,z 的点云表；仅当含样本/像素 id 或明显的表型列时判为标签
        if {"x", "y", "z"}.issubset(set(low[:3])) and all(
            c in ("x", "y", "z", "reflectance", "intensity") for c in low
        ):
            continue
        if any("sample" in c or "pixel" in c for c in low[:3]) or len(low) >= 2:
            return h
    return None


def _render_fertilizer_html(presc: Dict, crop: str, png_path: str) -> str:
    """内存里渲染一个可下载的自包含 HTML 报告。"""
    import base64
    rec_text = build_recommendations(presc, crop)
    b64 = ""
    if os.path.exists(png_path):
        b64 = base64.b64encode(open(png_path, "rb").read()).decode("ascii")
    rows = "".join(
        f"<tr><td>{r['zone'] + 1} 级</td><td>{r['n_rate']:.0f}</td>"
        f"<td>{r['area_pct']:.1f}%</td><td>{r['area_m2']:.1f}</td><td>{r['n_kg']:.1f}</td></tr>"
        for r in presc["zone_table"]
    )
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>
<title>{crop} 施肥建议报告</title><style>
body{{font-family:'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#1f2933;margin:0;padding:24px;}}
h1{{color:#16324f;border-bottom:3px solid #4c72b0;padding-bottom:8px;}}
table{{border-collapse:collapse;width:100%;margin-top:12px;}} th,td{{border:1px solid #ccc;padding:8px;}}
th{{background:#eef2f7;}} img{{max-width:100%;border-radius:6px;margin-top:8px;}}
pre{{white-space:pre-wrap;font-family:inherit;background:#fff;padding:12px;border-radius:8px;}}
</style></head><body>
<h1>🌾 {crop} 变量施肥建议报告</h1>
<p>平均氮营养指数 {presc['mean_status']:.2f} · 平均施氮量 {presc['mean_rate']:.1f} kg N/ha · 合计需氮 {presc['total_n_kg']:.1f} kg</p>
<pre>{rec_text}</pre>
<h2>分区处方</h2>
<table><tr><th>等级</th><th>施氮量</th><th>面积占比</th><th>面积 m²</th><th>需氮 kg</th></tr>{rows}</table>
{"<h2>处方图</h2><img src='data:image/png;base64," + b64 + "'/>" if b64 else ""}
</body></html>"""


def page_settings():
    st.markdown("**系统配置**")
    st.write({"数据源": source_actual, "模型": model_type, "交叉验证折数": cv_folds,
              "波段": int(ss.n_bands), "样本": int(ss.n_samples),
              "表型": ss.trait_names or "无（仅光谱）"})
    st.write("**数据源元信息**")
    st.json(ss.meta)
    st.download_button("下载表型指标 CSV",
                       data=pd.DataFrame({"trait": ss.trait_names or ["(none)"]}).to_csv(index=False),
                       file_name="phenotype_summary.csv", mime="text/csv")


page_map = {
    "数据管理": page_overview,
    "实时采集": page_acquisition,
    "数据处理": page_processing,
    "配准校正": page_registration,
    "特征提取": page_features,
    "表型分析": page_phenotyping,
    "施肥建议": page_fertilizer,
    "数据上传": page_upload,
    "系统设置": page_settings,
}

page_map[page]()
