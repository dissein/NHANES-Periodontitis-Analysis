import pathlib, csv, json, math, base64

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
WEB.mkdir(exist_ok=True)
RES = ROOT / 'results'
FIGS = ROOT / 'figures'
DOCS = ROOT / 'docs'

# Load data tables
files = ['performance', 'paired_comparisons', 'calibration_bins', 'operating_points', 'subgroups', 'prevalence', 'refit_uncertainty', 'associations_nonlab']
data = {}

for f in files:
    csv_path = RES / f'{f}.csv'
    if csv_path.exists():
        rows = []
        for r in csv.DictReader(csv_path.open(encoding='utf-8-sig')):
            row = {}
            for k, v in r.items():
                try:
                    num = float(v)
                    row[k] = num if math.isfinite(num) else None
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
        data[f] = rows

paper_path = DOCS / '完整研究稿_方法结果讨论.md'
data['paper_markdown'] = paper_path.read_text(encoding='utf-8') if paper_path.exists() else ""

# Load 6 high-res figures as base64
fig_b64 = {}
fig_meta = {
    '01_panel_performance': {
        'title': '图1：各面板在开发集与验证集的区分能力森林图',
        'desc': '展示 5 组信息面板在 3 种算法、2 个周期下的加权 AUC 及 95% 置信区间。问卷 7 项在外部验证集 G 达到 AUC 0.801，与 47 项扩展面板表现相当。'
    },
    '02_laboratory_increment': {
        'title': '图2：47项扩展对比11项的化验增量检验 (ΔAUC 与 95% CI)',
        'desc': '在控制血压混杂的前提下，比较扩展 47 项与非化验含血压 11 项。三种算法的 ΔAUC 95% 置信区间全部跨越 0 轴，未建立统计学显著增量。'
    },
    '03_calibration_shift': {
        'title': '图3：跨周期校准曲线与预测概率偏移 (H内部 vs G外部)',
        'desc': 'H 周期内部验证校准良好（截距 -0.014，斜率 0.963）；迁移至 G 周期后出现截距正向漂移（截距 0.404），整体预测概率系统性偏低。'
    },
    '04_threshold_workload': {
        'title': '图4：不同敏感度目标下的假阳性与检查工作量分布',
        'desc': '展示以 80% 与 90% 敏感度目标为阈值时，每 1,000 人中标记数、真阳性数与假阳性数分布。敏感度 80% 时，假阳性达 198/1,000 人。'
    },
    '05_external_contributions': {
        'title': '图5：特征归因贡献度与删除重训可替代性分析',
        'desc': 'TreeSHAP 归因重要性 vs. 特征删除后重新训练的 AUC 损失对比。自报吸烟与可替宁单删损失微弱，联合删除损失显著，证实信息高度重叠。'
    },
    '06_subgroup_discrimination': {
        'title': '图6：年龄亚组区分能力与高龄人群衰减',
        'desc': '30–49 岁人群 AUC 为 0.834，50–64 岁为 0.765，而 65 岁及以上高龄人群急剧下降至 0.646，揭示了整体模型无法代表老年人群的重要边界。'
    }
}

for p in sorted(FIGS.glob('*.png')):
    b64_str = base64.b64encode(p.read_bytes()).decode('ascii')
    fig_b64[p.stem] = f"data:image/png;base64,{b64_str}"

data_json = json.dumps(data, ensure_ascii=False)
fig_json = json.dumps(fig_b64, ensure_ascii=False)
fig_meta_json = json.dumps(fig_meta, ensure_ascii=False)

html_template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHANES 中重度牙周炎识别：化验信息与模型复杂度的增量价值研究</title>
  <meta name="description" content="基于 NHANES 跨周期调查加权数据的中重度牙周炎风险预测模型实证研究，探讨化验信息增量与算法复杂度。">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Noto+Serif+SC:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #090d16;
      --bg-surface: #0f172a;
      --bg-card: rgba(15, 23, 42, 0.78);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --bg-card-alt: rgba(30, 41, 59, 0.5);
      --border-subtle: rgba(148, 163, 184, 0.14);
      --border-focus: rgba(20, 184, 166, 0.5);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-teal: #14b8a6;
      --accent-cyan: #06b6d4;
      --accent-indigo: #6366f1;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --glow-teal: rgba(20, 184, 166, 0.15);
      --glow-indigo: rgba(99, 102, 241, 0.15);
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif;
      --font-serif: 'Noto Serif SC', Georgia, Cambria, serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-card: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    [data-theme="light"] {
      --bg-base: #f8fafc;
      --bg-surface: #ffffff;
      --bg-card: rgba(255, 255, 255, 0.88);
      --bg-card-hover: rgba(241, 245, 249, 0.95);
      --bg-card-alt: rgba(241, 245, 249, 0.7);
      --border-subtle: rgba(203, 213, 225, 0.8);
      --border-focus: rgba(13, 148, 136, 0.5);
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #94a3b8;
      --accent-teal: #0d9488;
      --accent-cyan: #0284c7;
      --accent-indigo: #4f46e5;
      --accent-amber: #d97706;
      --accent-rose: #e11d48;
      --glow-teal: rgba(13, 148, 136, 0.08);
      --glow-indigo: rgba(79, 70, 229, 0.08);
      --shadow-card: 0 10px 25px -5px rgba(15, 23, 42, 0.06);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; font-size: 15px; }
    body {
      background: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.65;
      overflow-x: hidden;
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Ambient dynamic glow background */
    .bg-mesh {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      pointer-events: none;
      z-index: 0;
      background: radial-gradient(circle at 15% 15%, var(--glow-teal) 0%, transparent 45%),
                  radial-gradient(circle at 85% 20%, var(--glow-indigo) 0%, transparent 45%),
                  radial-gradient(circle at 50% 85%, rgba(245, 158, 11, 0.05) 0%, transparent 50%);
    }

    .container {
      max-width: 1240px;
      margin: 0 auto;
      padding: 0 28px;
      position: relative;
      z-index: 1;
    }

    /* Top Sticky Navigation */
    .navbar {
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      background: rgba(9, 13, 22, 0.82);
      border-bottom: 1px solid var(--border-subtle);
      transition: all 0.3s ease;
    }
    [data-theme="light"] .navbar {
      background: rgba(255, 255, 255, 0.85);
    }
    .nav-inner {
      height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: var(--text-main);
    }
    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-teal), var(--accent-indigo));
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #fff;
      font-size: 16px;
      box-shadow: 0 4px 12px rgba(20, 184, 166, 0.35);
    }
    .brand-title {
      font-weight: 700;
      font-size: 16px;
      letter-spacing: -0.3px;
    }
    .brand-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .nav-links {
      display: flex;
      align-items: center;
      gap: 20px;
      list-style: none;
    }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 13.5px;
      font-weight: 500;
      transition: color 0.2s;
    }
    .nav-links a:hover {
      color: var(--accent-teal);
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 600;
      border-radius: var(--radius-sm);
      cursor: pointer;
      border: 1px solid transparent;
      text-decoration: none;
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-teal), #0ea5e9);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(20, 184, 166, 0.3);
    }
    .btn-primary:hover {
      opacity: 0.92;
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: var(--bg-card-alt);
      color: var(--text-main);
      border-color: var(--border-subtle);
    }
    .btn-secondary:hover {
      background: var(--bg-card-hover);
      border-color: var(--accent-teal);
    }
    .theme-toggle-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 16px;
      transition: all 0.2s;
    }
    .theme-toggle-btn:hover {
      border-color: var(--accent-amber);
    }

    /* Hero Section */
    .hero {
      padding: 64px 0 48px;
      text-align: center;
      position: relative;
    }
    .badge-row {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }
    .pill {
      font-size: 11.5px;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 20px;
      background: rgba(20, 184, 166, 0.12);
      color: var(--accent-teal);
      border: 1px solid rgba(20, 184, 166, 0.25);
      letter-spacing: 0.3px;
    }
    .pill.pill-indigo {
      background: rgba(99, 102, 241, 0.12);
      color: var(--accent-indigo);
      border-color: rgba(99, 102, 241, 0.25);
    }
    .pill.pill-amber {
      background: rgba(245, 158, 11, 0.12);
      color: var(--accent-amber);
      border-color: rgba(245, 158, 11, 0.25);
    }
    .hero-title {
      font-family: var(--font-serif);
      font-size: 40px;
      line-height: 1.25;
      font-weight: 700;
      letter-spacing: -0.8px;
      margin-bottom: 20px;
      background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, var(--accent-teal) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    [data-theme="light"] .hero-title {
      background: linear-gradient(135deg, #0f172a 30%, #334155 70%, var(--accent-teal) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
      max-width: 820px;
      margin: 0 auto 36px;
      font-size: 16px;
      color: var(--text-muted);
      line-height: 1.7;
    }
    .author-bar {
      display: inline-flex;
      align-items: center;
      gap: 16px;
      padding: 8px 20px;
      border-radius: 30px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 44px;
    }
    .author-name {
      font-weight: 600;
      color: var(--text-main);
    }

    /* Hero 4 Metric Cards */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-bottom: 56px;
    }
    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 24px 20px;
      text-align: left;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-card);
      transition: all 0.3s ease;
    }
    .metric-card:hover {
      transform: translateY(-3px);
      border-color: var(--accent-teal);
    }
    .metric-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--accent-teal), var(--accent-indigo));
      opacity: 0.8;
    }
    .metric-card:nth-child(2)::before {
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
    }
    .metric-card:nth-child(3)::before {
      background: linear-gradient(90deg, var(--accent-amber), var(--accent-rose));
    }
    .metric-card:nth-child(4)::before {
      background: linear-gradient(90deg, var(--accent-rose), var(--accent-indigo));
    }
    .metric-val {
      font-family: var(--font-mono);
      font-size: 28px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 4px;
      letter-spacing: -0.5px;
    }
    .metric-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 6px;
    }
    .metric-sub {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
    }

    /* Core Findings Cards */
    .section-title-wrap {
      margin-bottom: 28px;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 14px;
    }
    .section-title {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .section-desc {
      font-size: 13px;
      color: var(--text-muted);
    }
    .findings-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      margin-bottom: 60px;
    }
    .finding-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 26px;
      box-shadow: var(--shadow-card);
      position: relative;
      transition: all 0.3s;
    }
    .finding-card:hover {
      border-color: var(--border-focus);
    }
    .finding-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    .finding-badge {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 14px;
      background: rgba(20, 184, 166, 0.15);
      color: var(--accent-teal);
    }
    .finding-title {
      font-size: 17px;
      font-weight: 700;
      color: var(--text-main);
    }
    .finding-body {
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.7;
    }
    .finding-highlight {
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      background: var(--bg-card-alt);
      border-left: 3px solid var(--accent-teal);
      font-size: 12.5px;
      font-family: var(--font-mono);
      color: var(--text-main);
    }

    /* Interactive Explorer Box */
    .explorer-box {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 32px;
      margin-bottom: 64px;
      box-shadow: 0 20px 40px -15px rgba(0,0,0,0.6);
    }
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 28px;
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .control-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
    }
    .btn-group {
      display: inline-flex;
      background: var(--bg-card-alt);
      border-radius: var(--radius-sm);
      padding: 3px;
      border: 1px solid var(--border-subtle);
    }
    .btn-toggle {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-toggle.active {
      background: var(--accent-teal);
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(20, 184, 166, 0.4);
    }

    /* Forest Plot Render Area */
    .forest-plot-container {
      margin-bottom: 28px;
    }
    .forest-header {
      display: grid;
      grid-template-columns: 220px 1fr 140px;
      padding: 10px 16px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      border-bottom: 1px solid var(--border-subtle);
    }
    .forest-row {
      display: grid;
      grid-template-columns: 220px 1fr 140px;
      align-items: center;
      padding: 16px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      transition: background 0.2s;
    }
    .forest-row:hover {
      background: var(--bg-card-hover);
    }
    .forest-label-title {
      font-weight: 600;
      font-size: 14px;
      color: var(--text-main);
    }
    .forest-label-desc {
      font-size: 11.5px;
      color: var(--text-muted);
    }
    .forest-axis-wrap {
      position: relative;
      height: 32px;
      display: flex;
      align-items: center;
    }
    .forest-baseline {
      position: absolute;
      top: 0; bottom: 0;
      width: 1px;
      background: rgba(148, 163, 184, 0.3);
      border-left: 1px dashed var(--accent-amber);
    }
    .forest-bar-range {
      position: absolute;
      height: 6px;
      border-radius: 3px;
      background: rgba(20, 184, 166, 0.35);
    }
    .forest-point {
      position: absolute;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--accent-teal);
      border: 2px solid #ffffff;
      transform: translate(-50%, -4px);
      box-shadow: 0 0 10px rgba(20, 184, 166, 0.8);
    }
    .forest-score {
      text-align: right;
      font-family: var(--font-mono);
    }
    .forest-score-val {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-main);
    }
    .forest-score-ci {
      font-size: 11px;
      color: var(--text-muted);
    }

    .explorer-summary {
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 18px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    .explorer-summary-text {
      font-size: 13.5px;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .explorer-summary-text strong {
      color: var(--accent-teal);
    }

    /* Two Columns Section (Increment & Substitution) */
    .two-col-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 60px;
    }
    .detail-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 26px;
      box-shadow: var(--shadow-card);
    }
    .detail-card h3 {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .detail-card p {
      font-size: 13.5px;
      color: var(--text-muted);
      margin-bottom: 18px;
      line-height: 1.6;
    }
    .table-clean {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .table-clean th {
      text-align: left;
      padding: 10px 12px;
      color: var(--text-dim);
      font-weight: 600;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 11.5px;
      text-transform: uppercase;
    }
    .table-clean td {
      padding: 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      color: var(--text-main);
    }
    .table-clean tr:hover td {
      background: var(--bg-card-hover);
    }

    /* Figure Gallery Showcase */
    .gallery-section {
      margin-bottom: 64px;
    }
    .gallery-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }
    .gallery-tab-btn {
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 600;
      border-radius: var(--radius-sm);
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
    }
    .gallery-tab-btn.active {
      background: var(--accent-indigo);
      color: #fff;
      border-color: var(--accent-indigo);
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .gallery-viewport {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
      text-align: center;
      box-shadow: var(--shadow-card);
    }
    .gallery-img-wrap {
      background: #ffffff;
      border-radius: var(--radius-md);
      padding: 16px;
      display: inline-block;
      max-width: 100%;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
      cursor: zoom-in;
    }
    .gallery-img {
      max-width: 100%;
      max-height: 540px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
    }
    .gallery-caption {
      margin-top: 18px;
      text-align: left;
      padding: 0 10px;
    }
    .gallery-caption-title {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 6px;
    }
    .gallery-caption-desc {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Q&A Accordion */
    .qa-grid {
      display: grid;
      gap: 14px;
      margin-bottom: 64px;
    }
    .qa-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
      transition: all 0.2s;
    }
    .qa-header {
      padding: 18px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      font-weight: 600;
      font-size: 15px;
      color: var(--text-main);
      user-select: none;
    }
    .qa-header:hover {
      background: var(--bg-card-hover);
    }
    .qa-icon {
      font-size: 18px;
      color: var(--accent-teal);
      transition: transform 0.2s;
    }
    .qa-card.open .qa-icon {
      transform: rotate(180deg);
    }
    .qa-content {
      padding: 0 22px 20px;
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.7;
      display: none;
      border-top: 1px solid rgba(148, 163, 184, 0.08);
      padding-top: 14px;
    }
    .qa-card.open .qa-content {
      display: block;
    }

    /* Modal for Image Zoom */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0, 0, 0, 0.88);
      backdrop-filter: blur(8px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 30px;
    }
    .modal-overlay.active {
      display: flex;
    }
    .modal-img {
      max-width: 95%;
      max-height: 92%;
      background: #ffffff;
      border-radius: 8px;
      padding: 10px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.8);
    }
    .modal-close {
      position: absolute;
      top: 24px; right: 28px;
      background: rgba(255,255,255,0.15);
      border: none;
      color: #fff;
      font-size: 24px;
      width: 44px; height: 44px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    /* Footer */
    footer {
      border-top: 1px solid var(--border-subtle);
      padding: 48px 0;
      background: var(--bg-surface);
      font-size: 13px;
      color: var(--text-dim);
    }
    .footer-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 20px;
    }

    @media (max-width: 980px) {
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
      .findings-grid { grid-template-columns: 1fr; }
      .two-col-grid { grid-template-columns: 1fr; }
      .forest-header, .forest-row { grid-template-columns: 160px 1fr 100px; }
      .nav-links { display: none; }
      .hero-title { font-size: 30px; }
    }
  </style>
</head>
<body>
  <div class="bg-mesh"></div>

  <!-- Navbar -->
  <header class="navbar">
    <div class="container nav-inner">
      <a href="#" class="brand">
        <div class="brand-icon">N</div>
        <div>
          <div class="brand-title">NHANES Periodontitis AI</div>
          <div class="brand-subtitle">跨周期实证与增量价值检验</div>
        </div>
      </a>
      <ul class="nav-links">
        <li><a href="#findings">核心结论</a></li>
        <li><a href="#explorer">交互探索器</a></li>
        <li><a href="#increment">化验增量</a></li>
        <li><a href="#substitution">信息重叠</a></li>
        <li><a href="#gallery">论文图表</a></li>
        <li><a href="#qa">学术答辩</a></li>
      </ul>
      <div class="nav-actions">
        <button class="theme-toggle-btn" id="themeToggle" title="切换暗色/明亮模式">🌙</button>
        <button class="btn btn-secondary" id="btnDownloadPaper">下载完整论文</button>
        <button class="btn btn-primary" id="btnDownloadCSV">导出数据 CSV</button>
      </div>
    </div>
  </header>

  <main class="container">
    <!-- Hero Section -->
    <section class="hero">
      <div class="badge-row">
        <span class="pill">CDC/AAP 2012 监测标准</span>
        <span class="pill pill-indigo">复杂抽样调查加权 (WTMEC2YR)</span>
        <span class="pill pill-amber">TRIPOD+AI 规范流程</span>
      </div>
      <h1 class="hero-title">
        额外化验信息与模型复杂度<br>能否带来可迁移的预测收益？
      </h1>
      <p class="hero-subtitle">
        一项基于 NHANES 跨周期调查加权（2013–2014 开发集 n=3,617 与 2011–2012 验证集 n=3,323，共 6,940 名成人）的中重度牙周炎现患识别与信息替代性实证研究。
      </p>
      <div class="author-bar">
        <span>研究者：<span class="author-name">周程</span></span>
        <span>•</span>
        <span>北京大学口腔医学院 2023级五年制本科生</span>
        <span>•</span>
        <span>完成日期：2026-10-11</span>
      </div>

      <!-- 4 High Impact Stat Cards -->
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-val">6,940</div>
          <div class="metric-label">两周期无重叠受试者</div>
          <div class="metric-sub">H周期 3,617 人内部开发；G周期 3,323 人跨周期外部验证。</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">0.801</div>
          <div class="metric-label">问卷 7 项跨周期 AUC</div>
          <div class="metric-sub">仅凭人口学与问卷信息，无需抽血即可达到 0.801 跨周期区分度。</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">+0.0009</div>
          <div class="metric-label">化验增量 ΔAUC (95% CI 跨0)</div>
          <div class="metric-sub">47项对比11项，区间 [-0.0079, 0.0095]，36项化验无稳定增益。</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">0.646</div>
          <div class="metric-label">65+ 岁高龄亚组 AUC</div>
          <div class="metric-sub">整体高 AUC 无法外推至老年人群，揭示了关键的适用范围边界。</div>
        </div>
      </div>
    </section>

    <!-- Core Findings Section -->
    <section id="findings">
      <div class="section-title-wrap">
        <h2 class="section-title">💡 四大核心实证结论</h2>
        <div class="section-desc">基于 1,000 次 Rao–Wu 设计重抽样及 200 次独立全流程 Bootstrap 验证</div>
      </div>
      <div class="findings-grid">
        <div class="finding-card">
          <div class="finding-header">
            <div class="finding-badge">1</div>
            <div class="finding-title">额外化验与算法复杂度未建立稳定增量</div>
          </div>
          <div class="finding-body">
            在控制血压的前提下，47 项扩展面板对比 11 项非化验面板，Logistic、EBM 与 XGBoost 的 ΔAUC 95% 置信区间全部跨越 0。在 200 次全流程重新插补、调参及训练 Bootstrap 中，ΔAUC 区间为 [-0.0168, 0.0084]。复杂算法同样未展现出超越年龄非线性处理后 Logistic 的优势。
          </div>
          <div class="finding-highlight">
            ΔAUC = +0.0009 [-0.0079, 0.0095] ｜ ΔBrier = -0.0005 [-0.0041, 0.0031]
          </div>
        </div>

        <div class="finding-card">
          <div class="finding-header">
            <div class="finding-badge">2</div>
            <div class="finding-title">归因贡献排名 ≠ 不可替代的预测收益</div>
          </div>
          <div class="finding-body">
            单凭 SHAP 排名选择特征存在盲区：家庭收入在 XGBoost SHAP 中高居第 4，但单删收入重新调参后 AUC 反而微升 0.0087；只有在收入与教育同时删除时才出现损失。自报吸烟与血清可替宁同样存在高度信息替代性，单删均无显著损失，双删则导致 AUC 显著下跌 0.0132。
          </div>
          <div class="finding-highlight">
            双删吸烟与可替宁：ΔAUC = 0.0132 [0.0078, 0.0188]，证实信息源互补与替代。
          </div>
        </div>

        <div class="finding-card">
          <div class="finding-header">
            <div class="finding-badge">3</div>
            <div class="finding-title">跨周期概率存在漂移，校准截距达 0.404</div>
          </div>
          <div class="finding-body">
            在 H 周期内部验证中，校准截距为 -0.014、斜率为 0.963；但当冻结模型迁移至较早的 G 周期时，校准截距显著正漂移至 0.404（斜率 1.051），表明模型在外部周期系统性低估了患者实际患病风险。区分能力优良不等于预测概率准确，临床应用前必须重新标定。
          </div>
          <div class="finding-highlight">
            外部校准截距：0.404 ｜ 校准斜率：1.051 ｜ 提示概率不可直接用于绝对风险评估。
          </div>
        </div>

        <div class="finding-card">
          <div class="finding-header">
            <div class="finding-badge">4</div>
            <div class="finding-title">年龄分层异质性显著，老年亚组降至 0.646</div>
          </div>
          <div class="finding-body">
            问卷 Logistic 在 30–49 岁人群中表现极佳（AUC 0.834 [0.812, 0.857]），50–64 岁为 0.765，但在 65 岁及以上人群中跌至 0.646 [0.587, 0.713]。高龄人群中牙周病现患率高且累积危险因素泛化，基于常规问卷指标难以进行敏锐二次区分。
          </div>
          <div class="finding-highlight">
            30–49岁: 0.834 ｜ 50–64岁: 0.765 ｜ 65+岁: 0.646（整体 AUC 无法代表高龄表现）
          </div>
        </div>
      </div>
    </section>

    <!-- Interactive Explorer Section -->
    <section id="explorer">
      <div class="section-title-wrap">
        <h2 class="section-title">📊 交互模型探索器 (Interactive Model Explorer)</h2>
        <div class="section-desc">自由切换周期阶段、算法及信息面板，实时观察性能分布与 95% 置信区间</div>
      </div>

      <div class="explorer-box">
        <div class="controls-bar">
          <div class="control-group">
            <span class="control-label">验证阶段：</span>
            <div class="btn-group" id="stageToggle">
              <button class="btn-toggle active" data-val="External">G 周期 外部验证 (n=3,323)</button>
              <button class="btn-toggle" data-val="Internal">H 周期 内部开发 (n=3,617)</button>
            </div>
          </div>
          <div class="control-group">
            <span class="control-label">算法选择：</span>
            <div class="btn-group" id="algoToggle">
              <button class="btn-toggle active" data-val="Logistic">样条岭回归 (Logistic)</button>
              <button class="btn-toggle" data-val="EBM">可解释增强机 (EBM)</button>
              <button class="btn-toggle" data-val="XGBoost">梯度提升树 (XGBoost)</button>
            </div>
          </div>
        </div>

        <!-- Forest Plot -->
        <div class="forest-plot-container">
          <div class="forest-header">
            <div>信息面板与层级</div>
            <div style="text-align: center;">加权 AUC 及 95% 置信区间 [0.74 ~ 0.84]</div>
            <div style="text-align: right;">AUC [95% CI]</div>
          </div>
          <div id="forestRows">
            <!-- Rows injected by JavaScript -->
          </div>
        </div>

        <div class="explorer-summary">
          <div class="explorer-summary-text" id="explorerNote">
            正在载入探索器数据...
          </div>
          <div style="display:flex;gap:16px;text-align:right;">
            <div><span style="font-size:11px;color:var(--text-dim);">样本量 n</span><br><strong id="summaryN">3,323</strong></div>
            <div><span style="font-size:11px;color:var(--text-dim);">加权患病率</span><br><strong id="summaryPrev">41.1%</strong></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Two-column: Lab Increment & Feature Substitution -->
    <section id="increment">
      <div class="two-col-grid">
        <!-- Col 1: Lab Increment -->
        <div class="detail-card">
          <h3>🔬 化验增量严谨检验 (47项 vs 11项)</h3>
          <p>
            为隔离体检指标与血压对化验分析的混杂干扰，我们将主要化验比较严格锚定在<b>扩展 47 项</b>对比<b>非化验含血压 11 项</b>上。
          </p>
          <table class="table-clean">
            <thead>
              <tr>
                <th>模型算法</th>
                <th>ΔAUC [95% CI]</th>
                <th>ΔBrier [95% CI]</th>
                <th>显著性</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>Logistic</b></td>
                <td style="font-family:var(--font-mono);color:var(--accent-teal);">+0.0009 [-0.0079, 0.0095]</td>
                <td style="font-family:var(--font-mono);">-0.0005 [-0.0041, 0.0031]</td>
                <td><span style="color:var(--text-dim);">不显著 (跨0)</span></td>
              </tr>
              <tr>
                <td><b>EBM</b></td>
                <td style="font-family:var(--font-mono);">-0.0002 [-0.0087, 0.0080]</td>
                <td style="font-family:var(--font-mono);">-0.0005 [-0.0042, 0.0032]</td>
                <td><span style="color:var(--text-dim);">不显著 (跨0)</span></td>
              </tr>
              <tr>
                <td><b>XGBoost</b></td>
                <td style="font-family:var(--font-mono);">+0.0066 [-0.0029, 0.0147]</td>
                <td style="font-family:var(--font-mono);">-0.0009 [-0.0047, 0.0029]</td>
                <td><span style="color:var(--text-dim);">不显著 (跨0)</span></td>
              </tr>
              <tr style="background:var(--bg-card-alt);">
                <td><b>200次全流程Bootstrap</b></td>
                <td style="font-family:var(--font-mono);color:var(--accent-amber);" colspan="2">[-0.0168, 0.0084] (涵盖重新调参/插补全流程)</td>
                <td><span style="color:var(--accent-amber);">高度稳健</span></td>
              </tr>
            </tbody>
          </table>
          <p style="margin-top:14px;font-size:12px;color:var(--text-dim);">
            注：ΔAUC > 0 或 ΔBrier < 0 表示 47 项扩展模型表现改善。全部三项置信区间均覆盖 0 轴。
          </p>
        </div>

        <!-- Col 2: Feature Substitution -->
        <div class="detail-card" id="substitution">
          <h3>🧩 特征归因与信息可替代性</h3>
          <p>
            对自报行为问卷与实验室生化标志物进行<b>删除重新训练</b>（而非单看 SHAP 排名），揭示信息之间的重叠与互补机制：
          </p>
          <table class="table-clean">
            <thead>
              <tr>
                <th>删除测试方案</th>
                <th>AUC 损失 [95% CI]</th>
                <th>机制解释</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>单删自报吸烟</b></td>
                <td style="font-family:var(--font-mono);">+0.0020 [-0.0010, 0.0060]</td>
                <td>损失微弱，可替宁维持了大部分预测信号</td>
              </tr>
              <tr>
                <td><b>单删血清可替宁</b></td>
                <td style="font-family:var(--font-mono);">+0.0011 [-0.0007, 0.0029]</td>
                <td>损失微弱，自报吸烟弥补了生化指标缺失</td>
              </tr>
              <tr style="background:rgba(244, 63, 94, 0.08);">
                <td><b>双删吸烟 + 可替宁</b></td>
                <td style="font-family:var(--font-mono);color:var(--accent-rose);font-weight:700;">+0.0132 [0.0078, 0.0188]</td>
                <td><b>显著损失！</b>证实两类指标具有高度信息替代性</td>
              </tr>
              <tr>
                <td><b>单删家庭收入</b></td>
                <td style="font-family:var(--font-mono);">-0.0087 (XGBoost)</td>
                <td>SHAP 排名居前，但单删后性能反而微升</td>
              </tr>
              <tr>
                <td><b>双删收入 + 教育</b></td>
                <td style="font-family:var(--font-mono);color:var(--accent-amber);">显著损失区间 > 0</td>
                <td>社会经济地位 (SES) 维度内部互补</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Figure Gallery Section -->
    <section class="gallery-section" id="gallery">
      <div class="section-title-wrap">
        <h2 class="section-title">🖼️ 论文高分辨率图表展厅</h2>
        <div class="section-desc">包含完整研究报告中的 6 张高精科研图表（点击可查看全屏高清放大图）</div>
      </div>

      <div class="gallery-tabs" id="galleryTabs">
        <!-- Generated by JS -->
      </div>

      <div class="gallery-viewport">
        <div class="gallery-img-wrap" id="galleryImgWrap" title="点击放大查看">
          <img class="gallery-img" id="activeGalleryImg" src="" alt="科研图表展示">
        </div>
        <div class="gallery-caption">
          <div class="gallery-caption-title" id="activeGalleryTitle">载入中...</div>
          <div class="gallery-caption-desc" id="activeGalleryDesc">载入中...</div>
        </div>
      </div>
    </section>

    <!-- Q&A Section -->
    <section id="qa">
      <div class="section-title-wrap">
        <h2 class="section-title">🎓 学术答辩与导师汇报高频问答</h2>
        <div class="section-desc">针对研究设计边界、临床定位及统计审稿要点的详尽回应</div>
      </div>

      <div class="qa-grid">
        <div class="qa-card open">
          <div class="qa-header">
            <span>问：为什么不把无牙颌或医疗排除者当作健康对照纳入？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：CDC/AAP 2012 牙周炎监测定义严格依赖全口天然牙邻面探诊深度 (PD) 与附着丧失 (CAL)。无牙颌患者大多因长期牙周炎严重破坏或重度龋病丧失全部牙齿，若强行将其归为“无牙周炎对照”，会严重颠倒疾病危险因素与真实暴露关系；而医疗排除者（因心血管疾病、抗生素预防等无法接受探诊）缺乏金标准结局。我们的结论严格限定于<b>可观察牙周检查的目标人群域</b>，避免产生选择偏倚。
          </div>
        </div>

        <div class="qa-card">
          <div class="qa-header">
            <span>问：为什么主化验增量比较是 47 项对 11 项，而不是直接对 7 项问卷？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：因为从 7 项问卷到完整 47 项扩展面板，中间包含了体格检查（BMI、腰围）和血压（收缩压、舒张压）。若直接比较 47 项与 7 项，一旦发现微弱增益，将无法厘清该收益是来自诊室快速测量的血压，还是来自高成本抽血化验。通过引入 11 项“非化验含血压”面板作为严格对照，我们纯化了 36 项血液与尿液实验室化验的净增量价值。
          </div>
        </div>

        <div class="qa-card">
          <div class="qa-header">
            <span>问：为什么说“增量证据不足”不能等同于“7项问卷等效于47项”？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：在统计学方法论中，“未能拒绝无增量零假设”并不等价于“证明两者等效”。我们没有预设生物等效性或临床非劣效性边界（Equivalence Margins），置信区间虽然跨越 0，但也包含了一定微弱效应范围。科学严谨的表述是“在现有样本及跨周期验证体系下，未观察到化验信息带来稳健、可迁移的预测改善”，而非宣称二者绝对等效。
          </div>
        </div>

        <div class="qa-card">
          <div class="qa-header">
            <span>问：SHAP 排名很高的变量（如收入），为什么删除后模型表现几乎不变甚至微升？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：这揭示了<b>特征使用度（Feature Attribution）</b>与<b>边际增量价值（Marginal Value）</b>的本质区别。SHAP 衡量的是在给定当前特征集下，模型对该特征的权重依赖程度；但当特征之间存在共线性（如收入与教育程度均反映社会经济地位 SES），模型会习惯性使用收入，一旦强制剔除收入并重新调参，教育变量完全可以接替其预测能力，因此单删损失为零。这警示我们：绝不能单凭 SHAP 排名直接挑选“核心生物标志物”。
          </div>
        </div>

        <div class="qa-card">
          <div class="qa-header">
            <span>问：整体 AUC 达到了 0.801，为什么还不能直接推广到临床应用？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：第一，模型存在显著的跨周期校准截距漂移（0.404），预测概率系统性偏低，直接使用会导致临床决策低估风险；第二，在疾病高发的 65 岁以上老年人群中，AUC 急剧衰减至 0.646，无法对老年患者提供可靠的危险度分层；第三，NHANES 是美国一般人群调查数据，不能直接替代中国临床就诊人群的真实筛查环境。
          </div>
        </div>

        <div class="qa-card">
          <div class="qa-header">
            <span>问：在本项目中，AI Agent 的角色是什么？你自己做了哪些实质性核验？</span>
            <span class="qa-icon">▼</span>
          </div>
          <div class="qa-content">
            答：AI 扮演了高效的编程与计算辅助工具（辅助编写数据处理脚本、搭建复杂调查加权与交叉验证循环、渲染图表）。我自己独立主导了：① CDC/AAP 2012 判定逻辑与 SAS 特殊缺失编码的逐行核验；② 发现并修正了既往分析中 UACR 尺度不当（mg/g 官方比值重建）及早期代码轻度判定括号优先级失误；③ 提出将 11 项非化验作为化验增量对照的控制变量逻辑；④ 对全部 6,940 条原始样本及 90 个最终模型的预测复现进行边界测试验证。
          </div>
        </div>
      </div>
    </section>

    <!-- Modal for Zoom -->
    <div class="modal-overlay" id="zoomModal">
      <button class="modal-close" id="modalClose">✕</button>
      <img class="modal-img" id="modalImg" src="" alt="放大图表">
    </div>
  </main>

  <!-- Footer -->
  <footer>
    <div class="container footer-content">
      <div>
        <strong style="color:var(--text-main);">NHANES 中重度牙周炎风险预测模型跨周期实证研究</strong><br>
        代码与数据基于美国 CDC/NCHS NHANES 公开数据集，遵循 TRIPOD+AI 及 PROBAST+AI 临床预测模型透明报告指南。
      </div>
      <div>
        北京大学口腔医学院 2023级五年制本科生 周程 · 2026年10月
      </div>
    </div>
  </footer>

  <script>
    const D = __DATA__;
    const FIGS = __FIGS__;
    const FIG_META = __FIG_META__;

    // Names mapping
    const panelNames = {
      'Questionnaire': ['问卷信息 (7项)', '年龄、性别、种族、教育、收入比、牙科就诊、自报吸烟'],
      'NonLab': ['非化验体测 (9项)', '问卷 7 项 + BMI、腰围'],
      'NonLabBP': ['非化验含血压 (11项)', '非化验 9 项 + 首次收缩压、舒张压'],
      'CBC': ['血常规 (19项)', '非化验 9 项 + 10 个外周血常规化验'],
      'Expanded': ['扩展全量 (47项)', '非化验 11 项 + 36 项血液与尿液化验']
    };

    let currentStage = 'External';
    let currentAlgo = 'Logistic';

    // Theme toggle
    const themeBtn = document.getElementById('themeToggle');
    themeBtn.addEventListener('click', () => {
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      document.documentElement.setAttribute('data-theme', isLight ? 'dark' : 'light');
      themeBtn.textContent = isLight ? '🌙' : '☀️';
    });

    // Download handlers
    document.getElementById('btnDownloadPaper').addEventListener('click', () => {
      const blob = new Blob([D.paper_markdown], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'NHANES_牙周炎模型增量价值研究_完整研究稿.md';
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('btnDownloadCSV').addEventListener('click', () => {
      const rows = D.performance.filter(r => r.variant === 'raw');
      const cols = ['stage', 'panel', 'algorithm', 'n', 'n_features', 'auc', 'auc_low', 'auc_high', 'brier', 'calibration_intercept', 'calibration_slope'];
      const csvContent = '\ufeff' + [cols.join(','), ...rows.map(r => cols.map(c => r[c]).join(','))].join('\r\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'NHANES_模型全指标跨周期汇总.csv';
      a.click();
      URL.revokeObjectURL(url);
    });

    // Forest plot rendering
    function renderForest() {
      const panels = ['Questionnaire', 'NonLab', 'NonLabBP', 'CBC', 'Expanded'];
      const rows = panels.map(p => {
        const item = D.performance.find(r => r.stage === currentStage && r.panel === p && r.algorithm === currentAlgo && r.variant === 'raw');
        return { panel: p, data: item };
      });

      // Axis range: 0.74 to 0.84 (span 0.10)
      const minA = 0.74, maxA = 0.84;
      const getPct = v => Math.max(0, Math.min(100, ((v - minA) / (maxA - minA)) * 100));

      const rowsHtml = rows.map(r => {
        const d = r.data;
        const [title, desc] = panelNames[r.panel];
        const auc = d ? d.auc : 0;
        const low = d ? d.auc_low : 0;
        const high = d ? d.auc_high : 0;

        const leftPct = getPct(low);
        const widthPct = Math.max(2, getPct(high) - leftPct);
        const dotPct = getPct(auc);

        return `
          <div class="forest-row">
            <div>
              <div class="forest-label-title">${title}</div>
              <div class="forest-label-desc">${desc}</div>
            </div>
            <div class="forest-axis-wrap">
              <div class="forest-baseline" style="left: ${getPct(0.80)}%;"></div>
              <div class="forest-bar-range" style="left: ${leftPct}%; width: ${widthPct}%;"></div>
              <div class="forest-point" style="left: ${dotPct}%;"></div>
            </div>
            <div class="forest-score">
              <div class="forest-score-val">${auc.toFixed(4)}</div>
              <div class="forest-score-ci">[${low.toFixed(3)}, ${high.toFixed(3)}]</div>
            </div>
          </div>
        `;
      }).join('');

      document.getElementById('forestRows').innerHTML = rowsHtml;

      // Update Summary Note
      const qItem = rows[0].data;
      const expItem = rows[4].data;
      const delta = expItem.auc - qItem.auc;

      const stageText = currentStage === 'External' ? 'G周期 (2011–2012) 外部验证' : 'H周期 (2013–2014) 内部开发';
      document.getElementById('explorerNote').innerHTML = `
        当前查看：<strong>${stageText}</strong> ｜ 算法：<strong>${currentAlgo}</strong><br>
        问卷 7 项 AUC 为 <strong>${qItem.auc.toFixed(4)}</strong>；加入 36 项化验后 47 项扩展 AUC 为 <strong>${expItem.auc.toFixed(4)}</strong>（净变化：<strong>${(delta >= 0 ? '+' : '') + delta.toFixed(4)}</strong>，95% CI 跨0）。
      `;
      document.getElementById('summaryN').textContent = currentStage === 'External' ? '3,323 人' : '3,617 人';
      document.getElementById('summaryPrev').textContent = currentStage === 'External' ? '41.1%' : '34.3%';
    }

    // Toggle events
    document.querySelectorAll('#stageToggle .btn-toggle').forEach(b => {
      b.addEventListener('click', e => {
        document.querySelectorAll('#stageToggle .btn-toggle').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        currentStage = b.dataset.val;
        renderForest();
      });
    });

    document.querySelectorAll('#algoToggle .btn-toggle').forEach(b => {
      b.addEventListener('click', e => {
        document.querySelectorAll('#algoToggle .btn-toggle').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        currentAlgo = b.dataset.val;
        renderForest();
      });
    });

    // Gallery Tabs
    const figKeys = Object.keys(FIGS);
    let activeFigKey = figKeys[0];

    const galleryTabsContainer = document.getElementById('galleryTabs');
    galleryTabsContainer.innerHTML = figKeys.map((k, i) => `
      <button class="gallery-tab-btn ${i === 0 ? 'active' : ''}" data-key="${k}">
        ${FIG_META[k].title.split('：')[0]}
      </button>
    `).join('');

    function updateGallery() {
      const img = document.getElementById('activeGalleryImg');
      const title = document.getElementById('activeGalleryTitle');
      const desc = document.getElementById('activeGalleryDesc');

      img.src = FIGS[activeFigKey];
      title.textContent = FIG_META[activeFigKey].title;
      desc.textContent = FIG_META[activeFigKey].desc;
    }

    galleryTabsContainer.addEventListener('click', e => {
      const btn = e.target.closest('.gallery-tab-btn');
      if (!btn) return;
      document.querySelectorAll('.gallery-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFigKey = btn.dataset.key;
      updateGallery();
    });

    // Modal Zoom
    const zoomModal = document.getElementById('zoomModal');
    const modalImg = document.getElementById('modalImg');
    document.getElementById('galleryImgWrap').addEventListener('click', () => {
      modalImg.src = FIGS[activeFigKey];
      zoomModal.classList.add('active');
    });
    document.getElementById('modalClose').addEventListener('click', () => {
      zoomModal.classList.remove('active');
    });
    zoomModal.addEventListener('click', e => {
      if (e.target === zoomModal) zoomModal.classList.remove('active');
    });

    // Q&A Accordion
    document.querySelectorAll('.qa-header').forEach(h => {
      h.addEventListener('click', () => {
        const card = h.closest('.qa-card');
        card.classList.toggle('open');
      });
    });

    // Init
    renderForest();
    updateGallery();
  </script>
</body>
</html>
'''

full_html = html_template.replace('__DATA__', data_json).replace('__FIGS__', fig_json).replace('__FIG_META__', fig_meta_json)

out_file = WEB / 'NHANES_研究展示.html'
out_file.write_text(full_html, encoding='utf-8')

print(f"Successfully generated showcase HTML: {out_file} (Size: {len(full_html)} bytes)")
