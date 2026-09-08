"""施肥建议报告生成器：把处方图统计转成可读的中文建议（文本 + HTML）。

说明：
  * 这里只做**氮素**的变量施肥建议（光谱直接可反演的是冠层/叶片氮状态）。
  * 磷、钾及微量元素需土壤速效养分化验，当前报告作为"待补化验"边界给出，
    不影响氮处方图落地。
  * ``recommendations`` 分区文案与实际软件口径一致，可直接用于汇报。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from demo.plots import plot_prescription_map


ZONE_LABELS = ["充足", "较充足", "适中", "不足", "严重缺乏"]


def _zone_default(zone: int) -> Dict[str, Any]:
    """按等级返回默认分区解释与建议文案。

    处方索引升序 = 施氮量升序 = 氮营养指数降序 = 缺氮程度加重：
    zone 0（施氮最少）对应"充足"，zone n-1（施氮最多）对应"严重缺乏"。
    """
    base = {
        0: ("氮素充足，可少施或不施", "建议少施或不施氮肥，节省成本并减少环境负荷。"),
        1: ("氮素较充足，适度控施", "建议控施氮肥，重点关注是否出现贪青/倒伏。"),
        2: ("氮素中等，维持常规用量", "建议维持常规氮肥用量，按目标产量精细调控。"),
        3: ("氮素不足，建议适当增施", "建议增施中高氮配方，分次施用降低淋溶风险。"),
        4: ("氮素严重缺乏，需追施高氮", "建议立即追施高氮配方，并配合水分管理提高利用率。"),
    }
    return base.get(zone, ("待补充", "需结合农艺师现场判断。"))


def build_recommendations(prescription: Dict[str, Any], crop: str = "field_crop") -> str:
    """生成分区建议的纯文本（含总体判断）。"""
    status = prescription.get("mean_status", 0)
    if status >= 0.75:
        overall = "整体氮素供应充足，建议以控施/少施肥为主，侧重节本与品质。"
    elif status >= 0.5:
        overall = "整体氮素水平适中，建议按目标产量维持常规用量并做好分期调控。"
    elif status >= 0.25:
        overall = "整体氮素偏不足，建议适当增施氮肥，优先覆盖低氮区。"
    else:
        overall = "整体氮素严重不足，建议尽快追施高氮并核查灌排与土壤状况。"

    lines = [f"# {crop} 变量施肥建议", "", overall, ""]
    lines.append("## 分区处方")
    for row in prescription.get("zone_table", []):
        z = int(row["zone"])
        headline, detail = _zone_default(z)
        lines.append(
            f"- **{headline}**（{z + 1} 级）：建议 {row['n_rate']:.0f} kg N/ha，"
            f"覆盖 {row['area_pct']:.1f}% 面积。{detail}"
        )
    lines.append("")
    lines.append(
        f"合计需氮：{prescription.get('total_n_kg', 0):.1f} kg N；"
        f"平均 {prescription.get('mean_rate', 0):.1f} kg N/ha；"
        f"处方区 {len(prescription.get('zone_table', []))} 个。"
    )
    lines.append(
        "> 注：P/K 及微量元素需土壤速效养分化验，本报告仅给出氮素处方；"
        "面积按像素地面尺寸估算，实际请以地块边界为准。"
    )
    return "\n".join(lines)


def build_fertilizer_html(
    source: str,
    crop: str,
    prescription: Dict[str, Any],
    prescription_png: str,
    recommendations: str,
) -> str:
    """生成自包含 HTML 报告（处方图以 base64 内嵌，可离线打开）。"""
    rows = []
    for row in prescription.get("zone_table", []):
        z = int(row["zone"])
        headline, _ = _zone_default(z)
        rows.append(
            f"<tr><td>{z + 1} 级 · {headline}</td>"
            f"<td>{row['n_rate']:.0f}</td>"
            f"<td>{row['area_pct']:.1f}%</td>"
            f"<td>{row['area_m2']:.1f} m²</td>"
            f"<td>{row['n_kg']:.1f}</td></tr>"
        )
    table = "".join(rows)
    total_n = prescription.get("total_n_kg", 0)
    mean_rate = prescription.get("mean_rate", 0)
    mean_status = prescription.get("mean_status", 0)
    total_area = prescription.get("total_area_m2", 0)

    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>
<title>{crop} 变量施肥建议报告</title>
<style>
 body{{font-family:'Microsoft YaHei',sans-serif;margin:0;background:#f5f7fa;color:#1f2933;}}
 .wrap{{max-width:960px;margin:0 auto;padding:28px 24px;}}
 h1{{color:#16324f;border-bottom:3px solid #4c72b0;padding-bottom:10px;}}
 h2{{color:#16324f;margin-top:28px;}}
 .card{{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-top:16px;}}
 table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #e2e6ea;padding:8px 10px;text-align:left;}}
 th{{background:#eef2f7;}}
 .fig img{{width:100%;border-radius:6px;}} .cap{{text-align:center;color:#5c6b7a;font-size:13px;}}
 .metrics{{display:flex;gap:14px;flex-wrap:wrap;}} .m{{flex:1;min-width:140px;background:#fff;border-radius:8px;padding:14px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06);}}
 .m b{{font-size:26px;color:#4c72b0;}}
 pre{{white-space:pre-wrap;font-family:inherit;}}
 footer{{margin-top:40px;color:#8a97a6;font-size:12px;}}
</style></head><body><div class="wrap">
<h1>🌾 {crop} 变量施肥建议报告</h1>
<p>基于多模态高光谱成像 + 表型反演的差异化氮素管理建议</p>
<div class="card"><b>数据源：</b>{source}　<b>生成时间：</b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<h2>总体结论</h2>
<div class="metrics">
 <div class="m"><b>{mean_status:.2f}</b><br/>平均氮营养指数</div>
 <div class="m"><b>{mean_rate:.1f}</b><br/>平均施氮量 kg N/ha</div>
 <div class="m"><b>{total_n:.1f}</b><br/>合计需氮 kg</div>
 <div class="m"><b>{total_area/10000:.2f}</b><br/>处方面积 ha</div>
</div>
<div class="card"><pre>{recommendations}</pre></div>

<h2>分区处方</h2>
<div class="card"><table><tr><th>处方等级</th><th>施氮量 (kg N/ha)</th><th>面积占比</th><th>面积</th><th>需氮 (kg)</th></tr>
{table}</table></div>

<h2>处方图</h2>
<div class="card"><div class="fig"><img src="data:image/png;base64,{prescription_png}"/>
<div class="cap">变量施肥处方图（氮营养指数 / 施氮量 / 分区）</div></div></div>

<footer>本报告由演示系统生成；P/K 及微量元素需土壤化验补充。图片内嵌，可离线打开。</footer>
</div></body></html>"""


def make_fertilizer_report(
    *,
    source: str,
    crop: str,
    prescription: Dict[str, Any],
    out_png: Any,
    out_html: Any,
) -> Dict[str, Any]:
    """一站式：生成处方图 PNG、建议文本与 HTML 报告，返回写出的路径与文本。"""
    import base64

    plot_prescription_map(prescription, out_png, prescription.get("pixel_size_m", 0.05))
    recommendations = build_recommendations(prescription, crop)
    b64 = base64.b64encode(out_png.read_bytes()).decode("ascii")
    html_str = build_fertilizer_html(source, crop, prescription, b64, recommendations)
    out_html.write_text(html_str, encoding="utf-8")
    return {
        "recommendations": recommendations,
        "png": str(out_png),
        "html": str(out_html),
        "prescription": prescription,
    }
