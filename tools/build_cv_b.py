#!/usr/bin/env python3
"""Generate LHC_CV_B.pdf — resume tailored for embodied physics-engine roles."""

from pathlib import Path

from reportlab.lib.colors import Color, black, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)

FONT = "WQY"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))

PAGE_W, PAGE_H = A4
MARGIN_X = 1.35 * cm
MARGIN_Y = 1.0 * cm
ACCENT = HexColor("#1a365d")
LINE = HexColor("#2c5282")
MUTED = HexColor("#2d3748")


def styles():
    s = {}
    s["name"] = ParagraphStyle(
        "name", fontName=FONT, fontSize=16, leading=19, textColor=ACCENT, alignment=1, spaceAfter=1
    )
    s["intent"] = ParagraphStyle(
        "intent", fontName=FONT, fontSize=9.5, leading=12, textColor=LINE, alignment=1, spaceAfter=3
    )
    s["sec"] = ParagraphStyle(
        "sec", fontName=FONT, fontSize=10.5, leading=13, textColor=ACCENT, spaceBefore=4, spaceAfter=1
    )
    s["body"] = ParagraphStyle(
        "body", fontName=FONT, fontSize=8.5, leading=11.4, textColor=MUTED, alignment=0
    )
    s["bullet"] = ParagraphStyle(
        "bullet", fontName=FONT, fontSize=8.5, leading=11.2, textColor=MUTED,
        leftIndent=9, firstLineIndent=-9, spaceBefore=0.3, spaceAfter=0.3
    )
    s["proj"] = ParagraphStyle(
        "proj", fontName=FONT, fontSize=8.7, leading=11.4, textColor=black, spaceBefore=2, spaceAfter=0.3
    )
    s["meta"] = ParagraphStyle(
        "meta", fontName=FONT, fontSize=8.2, leading=10.8, textColor=MUTED, alignment=1
    )
    s["small"] = ParagraphStyle(
        "small", fontName=FONT, fontSize=8.0, leading=10.6, textColor=MUTED,
        leftIndent=9, firstLineIndent=-9, spaceBefore=0.2, spaceAfter=0.2
    )
    return s


def section(title, st):
    return KeepTogether([
        Paragraph(title, st["sec"]),
        HRFlowable(width="100%", thickness=0.8, color=LINE, spaceBefore=0, spaceAfter=4),
    ])


def bullet(text, st, style="bullet"):
    return Paragraph(f"• {text}", st[style])


def build(out: Path):
    st = styles()
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_Y,
        bottomMargin=MARGIN_Y,
        title="刘昊琛 - 具身物理引擎工程师简历",
        author="刘昊琛",
    )
    story = []

    story.append(Paragraph("刘昊琛", st["name"]))
    story.append(Paragraph("求职意向：具身物理引擎工程师 / Embodied Physics Engine Engineer", st["intent"]))
    story.append(Paragraph(
        "男 | 1997.11 | 中共党员 | 山东潍坊 | 18340091052<br/>"
        "Email：yc37477@um.edu.mo；l18340091052@gmail.com<br/>"
        "GitHub：LHappyCureall（Isaac Sim Fluid Lab / 数值求解器）",
        st["meta"],
    ))

    # 概述
    story.append(section("职业概述", st))
    story.append(Paragraph(
        "澳门大学数学博士在读，计算数学背景，具备 C++ / Python 双栈研发能力。"
        "长期从事流体动力学、多物理场有限元与大规模非线性求解器（Newton-Krylov / Jacobian / 预条件）研发；"
        "并基于 NVIDIA Isaac Sim + Newton + Warp 构建具身仿真实验：GPU 隐式 MPM 液滴/软体型 continuum、"
        "GPU PBF 流体、Franka 刚体-运动学交互与目标水量闭环控制。"
        "熟悉线性代数、微积分、数值方法、计算几何与可微/可导数值框架，"
        "希望从事具身智能物理引擎、仿真内核与二次开发相关工作。",
        st["body"],
    ))

    # 教育
    story.append(section("教育经历", st))
    edu = [
        ["2023.08 – 2027.08（预计）", "澳门大学", "数学 博士研究生"],
        ["2020.09 – 2023.06", "新疆大学", "计算数学 硕士研究生　GPA 3.61/4.0"],
        ["2016.09 – 2020.06", "山东师范大学", "数学与应用数学 本科　GPA 3.48/4.0"],
    ]
    t = Table(edu, colWidths=[4.2 * cm, 3.2 * cm, 11.2 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), MUTED),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8),
    ]))
    story.append(t)

    # 技能 — ATS keywords front and center
    story.append(section("专业技能（匹配物理引擎 / 具身仿真）", st))
    skills = [
        "<b>编程语言：</b>精通 C++、Python；能快速原型验证与脚本化实验流水线（批量仿真、日志、复现与验收）。",
        "<b>具身仿真引擎：</b>熟练使用 NVIDIA <b>Isaac Sim</b>（6.x）及配套 <b>Newton / Warp</b> GPU 物理栈；"
        "完成液滴隐式 MPM、GPU PBF 流体、Franka URDF/IK 场景搭建与二次开发；对比过 PhysX PBD 参数语义；"
        "掌握刚体/软体/流体约束类引擎共性，可迁移至 MuJoCo、Isaac Gym、PyBullet、Drake 等主流物理引擎二次开发。",
        "<b>物理建模：</b>深入理解<b>流体动力学</b>（Navier–Stokes / 相场 / PBF / MPM）、"
        "<b>刚体动力学</b>与机器人运动学交互、<b>软体动力学</b>/连续介质（隐式 MPM）；"
        "熟悉计算几何中的碰撞 SDF、粒子邻域、压力约束、质量守恒与边界条件。",
        "<b>数值与可微分：</b>扎实线性代数、微积分、数值方法、计算几何与数值分析；"
        "熟悉残量/<b>Jacobian（可微）</b>组装、Newton-Krylov、GMRES、预条件（两网格/AMG）及收敛诊断。",
        "<b>高性能计算：</b>C++ + PETSc/libMesh + MPI；千万自由度三维算例，强扩展效率 90%+；"
        "Linux 编译调试与超算部署（澳门大学 / 广州天河）。",
        "<b>工具与工程：</b>USD/Omniverse、GPU 表面重建、可复现实验协议；PyTorch 小规模实验；了解 Blender/Maya 资产管线。",
    ]
    for x in skills:
        story.append(bullet(x, st))

    # 项目 — KeepTogether 避免标题孤行
    story.append(section("项目与研究经历", st))

    p1 = [
        Paragraph(
            "<b>Isaac Sim 具身流体物理实验室（开源）</b>　|　独立研发　·　2025.11 – 至今",
            st["proj"],
        ),
        Paragraph(
            "仓库：github.com/LHappyCureall/isaac-sim-fluid-lab　|　Isaac Sim / Newton / Warp / Python / CUDA Graph",
            st["small"],
        ),
    ]
    for x in [
        "基于 Isaac Sim + Newton <b>SolverImplicitMPM</b> 实现液滴撞击（milkcrown）隐式 MPM："
        "密度/黏度（Pa·s）等物理参数、杯壁碰撞与穿透检查，Surface/Particles 可视化。",
        "自研 GPU <b>PBF</b> 液体求解器（邻域压力约束、黏性、容器 SDF 碰撞），支撑 Franka 试管倒水实时仿真"
        "（约 5k+ 粒子、质量守恒），解决细长容器中 MPM 过度压缩问题。",
        "搭建 <b>Franka 刚体</b>具身场景：URDF、RobotPoser IK、预握→倾倒→扶正；"
        "目标水量 20/30/40 mL 反馈控制与精密补水；CEM 策略搜索 + 留出测试；精密控制 MAE≈0.034 mL（9/9 ±0.1 mL）。",
    ]:
        p1.append(bullet(x, st))
    story.append(KeepTogether(p1))

    p2 = [
        Paragraph(
            "<b>质量守恒相场两相流全耦合并行求解</b>　|　博士课题　·　2023 – 至今",
            st["proj"],
        ),
    ]
    for x in [
        "质量守恒 Allen–Cahn–Navier–Stokes + 移动接触线；全耦合隐式有限元 → 大规模稀疏非线性系统。",
        "C++ / PETSc / libMesh / MPI 实现 Newton-Krylov-Schwarz：残量与 <b>Jacobian</b>、线搜索、GMRES/预条件；"
        "千万自由度三维测试，强扩展 90%+；覆盖润湿、滑移/渗透边界与含表面活性剂多相流。",
    ]:
        p2.append(bullet(x, st))
    story.append(KeepTogether(p2))

    p3 = [
        Paragraph(
            "<b>高并行预条件非精确 Newton 方法</b>　|　项目参与　·　2023 – 2025",
            st["proj"],
        ),
        bullet(
            "非线性消去（NE）/ NEPIN；代数·几何两网格、AMG、两水平及线性消去预条件，"
            "对比迭代与三维并行扩展——对应物理引擎中刚性系统与大规模求解内核能力。",
            st,
        ),
    ]
    story.append(KeepTogether(p3))

    p4 = [
        Paragraph(
            "<b>闭环地热流-热耦合有限元方法</b>　|　硕士科研　·　2020 – 2023",
            st["proj"],
        ),
        bullet(
            "适定性、罚方法与两网格解耦；误差估计与收敛实验，系统训练数值方法与多物理场建模。",
            st,
        ),
    ]
    story.append(KeepTogether(p4))

    # 教学
    story.append(section("教学 / 论文 / 关键词", st))
    story.append(bullet(
        "教学：有限元方法（理论+程序），B 站播放 8 万+；公开 Isaac Sim 液滴/倒水演示与实验文档。",
        st,
    ))
    story.append(bullet(
        "论文：质量守恒 AC-NS（Submitted JCP）；广义 Navier 滑移两相流（JCP 2025）；"
        "地热有限元适定性/两网格/罚方法（DCDS-B / JAAC / IJNAM 2023）；"
        "在研 NKS 接触线与 CH-NS surfactant 能量稳定格式。",
        st,
    ))
    story.append(Paragraph(
        "核心技术词：Isaac Sim · Newton · Warp · MuJoCo · Isaac Gym · PyBullet · Drake · PhysX · "
        "C++ · Python · 刚体动力学 · 流体动力学 · 软体动力学 · MPM · PBF · 计算几何 · "
        "数值方法 · Jacobian · 可微分 · 线性代数 · Franka · USD",
        st["meta"],
    ))

    doc.build(story)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "LHC_CV_B.pdf"
    build(out)
    # Also drop a copy under artifacts if present
    art = Path("/opt/cursor/artifacts")
    if art.is_dir():
        import shutil
        shutil.copy2(out, art / "LHC_CV_B.pdf")
        print(f"Copied to {art / 'LHC_CV_B.pdf'}")
