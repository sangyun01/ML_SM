from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "TCAD_ML_성능평가_요약.pdf"
FONT = ROOT / "reports" / "assets" / "Pretendard-M.ttf"

PAGE_W, PAGE_H = A4
BLUE = colors.HexColor("#2563EB")
NAVY = colors.HexColor("#172554")
LIGHT_BLUE = colors.HexColor("#EFF6FF")
LIGHT_GRAY = colors.HexColor("#F8FAFC")
MID_GRAY = colors.HexColor("#64748B")
GRID = colors.HexColor("#CBD5E1")
GREEN = colors.HexColor("#047857")
RED = colors.HexColor("#B91C1C")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Pretendard", str(FONT)))


def p(text, style):
    return Paragraph(text, style)


def table(data, widths, *, header=True, aligns=None, font_size=8.0):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Pretendard"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ]
    if aligns:
        for col, align in enumerate(aligns):
            style.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    t.setStyle(TableStyle(style))
    return t


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - 7 * mm, PAGE_W, 7 * mm, stroke=0, fill=1)
    canvas.setFont("Pretendard", 8)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(18 * mm, 10 * mm, "TCAD 기반 ML surrogate 및 inverse process design")
    canvas.drawRightString(PAGE_W - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="TCAD-ML 성능평가 요약",
        author="TCAD-ML Project Team",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=page_decor))

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleKR", parent=styles["Title"], fontName="Pretendard", fontSize=23,
        leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=6 * mm,
    )
    subtitle = ParagraphStyle(
        "SubtitleKR", parent=styles["Normal"], fontName="Pretendard", fontSize=10.5,
        leading=16, textColor=MID_GRAY, spaceAfter=8 * mm,
    )
    h1 = ParagraphStyle(
        "H1KR", parent=styles["Heading1"], fontName="Pretendard", fontSize=15,
        leading=21, textColor=BLUE, spaceBefore=2 * mm, spaceAfter=4 * mm,
    )
    h2 = ParagraphStyle(
        "H2KR", parent=styles["Heading2"], fontName="Pretendard", fontSize=11.5,
        leading=17, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "BodyKR", parent=styles["BodyText"], fontName="Pretendard", fontSize=9.3,
        leading=15, textColor=colors.HexColor("#1E293B"), spaceAfter=2.2 * mm,
    )
    small = ParagraphStyle(
        "SmallKR", parent=body, fontSize=8.2, leading=12.5, textColor=MID_GRAY,
    )
    box = ParagraphStyle(
        "BoxKR", parent=body, backColor=LIGHT_BLUE, borderColor=colors.HexColor("#BFDBFE"),
        borderWidth=0.7, borderPadding=8, borderRadius=4, spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
    eq = ParagraphStyle(
        "EqKR", parent=body, alignment=TA_CENTER, fontSize=10.2, leading=17,
        backColor=LIGHT_GRAY, borderPadding=6, spaceBefore=1.5 * mm, spaceAfter=2.5 * mm,
    )
    bullet = ParagraphStyle(
        "BulletKR", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm,
        bulletIndent=0, spaceAfter=1.2 * mm,
    )

    story = []
    story += [
        Spacer(1, 8 * mm),
        p("TCAD-ML 성능평가 요약", title),
        p(
            "Planar NMOS 공정조건 기반 forward prediction과 목표 소자특성 기반 "
            "inverse process recommendation의 검증 결과", subtitle,
        ),
        p(
            "<b>평가 원칙</b><br/>Forward 성능은 학습에 사용하지 않은 20% TCAD 데이터에서 평가하고, "
            "inverse 성능은 ML이 추천한 Top 1 공정조건을 실제 TCAD에 다시 입력하여 평가하였다.", box,
        ),
        p("1. Forward prediction 성능", h1),
        p(
            "전체 TCAD 데이터를 80% 학습 데이터와 20% 테스트 데이터로 분할하였다. "
            "Valid/Invalid 분류에는 전체 레시피를 사용하고, Vth·SS·Ion·Ioff 회귀에는 Valid 레시피만 사용하였다.", body,
        ),
        table(
            [
                ["모델", "전체 레시피", "Valid 레시피", "Valid/Invalid 정확도"],
                ["Short", "5,072", "3,317", "97.93%"],
                ["Long", "2,750", "2,332", "99.45%"],
            ],
            [27 * mm, 38 * mm, 38 * mm, 54 * mm],
            aligns=["CENTER"] * 4,
            font_size=8.4,
        ),
        Spacer(1, 4 * mm),
        table(
            [
                ["모델", "Vth", "SS", "Ion", "Ioff"],
                [
                    "Short",
                    "R² 0.960\nMAE 28.0 mV",
                    "R² 0.831\nlog RMSE 0.207 dec",
                    "R² 0.987\nlog RMSE 0.609 dec",
                    "R² 0.996\nlog RMSE 0.605 dec",
                ],
                [
                    "Long",
                    "R² 0.986\nMAE 6.20 mV",
                    "R² 0.994\nMAE 2.66 mV/dec",
                    "R² 0.994\nlog RMSE 0.010 dec",
                    "R² 0.794\nlog RMSE 0.267 dec",
                ],
            ],
            [20 * mm, 34 * mm, 37 * mm, 37 * mm, 37 * mm],
            aligns=["CENTER"] * 5,
            font_size=7.7,
        ),
        Spacer(1, 3 * mm),
        p(
            "Vth·SS는 Vd=0.05 V, Ion·Ioff는 Vd=1.0 V 결과를 사용하였다. "
            "Short SS의 원 단위 MAE는 극단적으로 큰 SS 값에 민감하므로 log RMSE를 함께 제시하였다.", small,
        ),
        p("2. Inverse recommendation 성능", h1),
        p(
            "극단적인 비정상 스펙은 제외하고, 기존 TCAD 데이터 분포에서 실제 소자로 사용할 가능성이 높은 영역을 "
            "기준으로 목표 스펙 테스트셋을 구성하였다. 아래 범위는 산업 인증규격이 아닌 프로젝트 내부 목표 생성범위이다.", body,
        ),
        table(
            [
                ["Vth", "SS", "Ion", "Ioff", "On/Off"],
                ["0.3-1.2 V", "≤200 mV/dec", "≥1×10⁻⁴ A", "≤1×10⁻⁶ A", "판정 제외"],
            ],
            [33 * mm] * 5,
            aligns=["CENTER"] * 5,
            font_size=8.1,
        ),
        PageBreak(),
        p("Inverse 테스트 구성", h1),
        p("• 각 Lg에서 500,000개의 후보 공정조건 생성", bullet),
        p("• 목표 스펙과 가장 가까운 Top 1 공정조건을 실제 TCAD로 재검증", bullet),
        p("• Short: 65/90 nm 각각 50건, 총 100건", bullet),
        p("• Long: 180/360/720/1000 nm 각각 50건, 총 200건", bullet),
        p("• On/Off ratio는 목표 및 평가에서 제외", bullet),
        Spacer(1, 2 * mm),
        p("ML 예상값과 실제 TCAD 결과의 일치도", h2),
        table(
            [
                ["모델·지표", "R²", "MAE", "RMSE"],
                ["Short Vth", "0.473", "85.1 mV", "142.0 mV"],
                ["Short SS", "0.021", "23.2 mV/dec", "58.1 mV/dec"],
                ["Short Ion (log)", "0.535", "0.125 dec", "0.162 dec"],
                ["Short Ioff (log)", "0.463", "1.785 dec", "2.431 dec"],
                ["Long Vth", "0.935", "10.2 mV", "18.6 mV"],
                ["Long SS", "0.685", "1.47 mV/dec", "2.50 mV/dec"],
                ["Long Ion (log)", "0.982", "0.017 dec", "0.023 dec"],
                ["Long Ioff (log)", "0.920", "0.282 dec", "0.559 dec"],
            ],
            [50 * mm, 30 * mm, 42 * mm, 42 * mm],
            aligns=["LEFT", "CENTER", "CENTER", "CENTER"],
            font_size=7.7,
        ),
        Spacer(1, 3 * mm),
        p("요청한 Min-Max 목표범위 달성률", h2),
        table(
            [
                ["목표 지표", "Short", "Long"],
                ["Vth", "53/100 (53%)", "197/200 (98.5%)"],
                ["SS", "84/100 (84%)", "200/200 (100%)"],
                ["Ion", "90/100 (90%)", "200/200 (100%)"],
                ["Ioff", "17/100 (17%)", "152/200 (76%)"],
                ["네 지표 동시 만족", "9/100 (9%)", "152/200 (76%)"],
            ],
            [82 * mm, 41 * mm, 41 * mm],
            aligns=["LEFT", "CENTER", "CENTER"],
            font_size=8.0,
        ),
        Spacer(1, 2 * mm),
        p(
            "각 테스트 사례에서 사용자가 요청한 Min-Max 범위 안에 실제 TCAD 결과가 들어간 비율이다. "
            "네 지표 동시 만족률을 inverse recommendation의 최종 성공률로 사용하였다.", small,
        ),
        PageBreak(),
        p("3. 평가지표 해석", h1),
        p("R² (결정계수)", h2),
        p("R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²", eq),
        p(
            "실제 데이터 변화 중 모델이 설명하는 정도를 나타낸다. 1에 가까울수록 좋고, 0은 평균값 예측과 비슷하며, "
            "음수는 평균값 예측보다 낮은 성능을 의미한다. 실제 오차의 단위와 크기는 알려주지 않으므로 MAE·RMSE와 함께 사용한다.", body,
        ),
        p("MAE (평균절대오차)", h2),
        p("MAE = (1/N) Σ |y - ŷ|", eq),
        p(
            "예측값과 TCAD 값의 절대 차이를 평균한다. Vth와 SS는 각각 mV와 mV/dec로 직접 해석할 수 있어 "
            "외부 검증의 주요 지표로 사용하였다.", body,
        ),
        p("RMSE (평균제곱근오차)", h2),
        p("RMSE = √[(1/N) Σ(y - ŷ)²]", eq),
        p(
            "큰 오차에 더 큰 페널티를 주므로 일부 공정조건에서 발생하는 심각한 예측 실패를 확인하는 데 적합하다.", body,
        ),
        p("Log MAE와 Log RMSE", h2),
        p("Elog = |log₁₀(y) - log₁₀(ŷ)|", eq),
        p(
            "Ion과 Ioff는 여러 자릿수에 걸쳐 변하므로 로그오차를 사용하였다. 0.3 decade는 약 2배, "
            "1 decade는 10배, 2 decades는 100배 오차를 의미한다. log MAE는 전형적인 배수오차를, "
            "log RMSE는 일부 사례에서 발생한 큰 decade 오차를 강조한다.", body,
        ),
        PageBreak(),
        p("4. SS와 Ioff 예측오차의 연관성", h1),
        p(
            "Short inverse 검증 100건에서 SS 절대오차와 Ioff 절대 로그오차의 Spearman 순위상관계수를 계산하였다.", body,
        ),
        p("ESS = |SS(TCAD) - SS(ML)|", eq),
        p("EIoff = |log₁₀(Ioff,TCAD) - log₁₀(Ioff,ML)|", eq),
        p("Spearman rho = 0.586,  p < 0.001", box),
        p(
            "SS 오차가 큰 공정조건일수록 Ioff 로그오차도 커지는 경향이 확인되었다. 이는 인과관계를 의미하지는 않지만, "
            "동일한 short-channel electrostatics 불안정 영역에서 두 모델이 함께 어려움을 겪고 있음을 보여준다.", body,
        ),
        table(
            [
                ["SS 오차 그룹", "건수", "SS 오차 중앙값", "Ioff 평균 로그오차", "배수 환산"],
                ["가장 작은 25건", "25", "1.43 mV/dec", "0.900 dec", "약 8배"],
                ["두 번째 25건", "25", "4.30 mV/dec", "0.983 dec", "약 9.6배"],
                ["세 번째 25건", "25", "9.38 mV/dec", "1.783 dec", "약 61배"],
                ["가장 큰 25건", "25", "30.26 mV/dec", "3.473 dec", "약 3,000배"],
            ],
            [39 * mm, 20 * mm, 37 * mm, 40 * mm, 29 * mm],
            aligns=["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"],
            font_size=8.0,
        ),
        PageBreak(),
        p("5. Short Ioff 오차가 큰 이유", h1),
        p(
            "Subthreshold 전류는 Vth와 SS에 대해 선형이 아니라 지수적으로 변한다. Short-channel에서는 DIBL이 유효 문턱전압을 "
            "낮춰 Vd=1 V의 off-state leakage를 추가로 증가시킬 수 있다.", body,
        ),
        p("ID ~ exp[(VG - Vth,eff) / (nVT)]", eq),
        p("SS = ln(10)nVT", eq),
        p("Vth,eff = Vth,0 - DIBL·VD", eq),
        p("Ioff ~ exp[-Vth,eff / (nVT)]", eq),
        p(
            "따라서 Vth, SS 또는 DIBL에 해당하는 electrostatics를 조금만 잘못 추정해도 Ioff는 수십 배에서 수백만 배까지 달라질 수 있다.", body,
        ),
        p("단일 이상치 여부", h2),
        table(
            [
                ["분석 조건", "Short Ioff log MAE"],
                ["전체 100건", "1.785 decades"],
                ["최악의 1건 제외", "1.733 decades"],
                ["최악의 10건 제외", "1.375 decades"],
                ["예측 방향", "과대 46건 / 과소 54건"],
            ],
            [70 * mm, 80 * mm],
            aligns=["LEFT", "CENTER"],
            font_size=8.3,
        ),
        Spacer(1, 4 * mm),
        p(
            "Ioff 오차는 한 개의 이상치나 한 방향의 편향으로 설명되지 않는다. 제한된 8차원 학습 데이터, "
            "급격한 SCE/DIBL 전이, Random Forest의 구간 평균 특성, 500,000개 후보 중 낙관적인 Top 1이 선택될 가능성이 "
            "복합적으로 작용한 결과다.", box,
        ),
        KeepTogether([
            p("6. 최종 결론", h1),
            p("• Long 모델은 forward 및 inverse 검증 모두에서 높은 신뢰도를 보였다.", bullet),
            p("• Short 모델은 Vth·SS·Ion은 비교적 양호하지만 Ioff와 정확한 다중 목표 동시 만족 성능이 제한적이었다.", bullet),
            p("• Short Ioff 오차는 SS 오차와 함께 증가하며, short-channel electrostatics가 불안정한 영역에 집중되었다.", bullet),
            p("• ML은 넓은 공정공간의 후보 탐색에 사용하고, 최종 후보는 TCAD로 재검증하는 hybrid framework가 적절하다.", bullet),
        ]),
    ]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
