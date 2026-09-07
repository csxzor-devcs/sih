"""Populate the SIH 2026 idea-template PPTX for PS 26153 (NTRO latent-dynamics).

This script DOES NOT change the template's design. It only edits the
existing text-box content in place so the template's slide layouts,
shapes, fonts, and colors stay intact.

Run:
    python3 scripts/build_sih_pptx.py

Output:
    sih/SIH2026-PS26153-Latent-Dynamics-Attack-Forecasting.pptx
        (overwritten; the original template is read-only)
"""
from copy import deepcopy
from pathlib import Path

from pptx import Presentation

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/home/caleb/Downloads/SIH2026-IDEA-Presentation-Format.pptx")
OUTPUT = REPO / "SIH2026-PS26153-Latent-Dynamics-Attack-Forecasting.pptx"


def replace_textbox(slide, shape_name, paragraphs):
    """Replace the text content of a named text box.  `paragraphs` is a
    list of (text, font_size_pt_or_None, bold) tuples.  Each tuple is
    placed on its own paragraph (so the title slide gets one line per
    tuple), and preserves the existing run-level font family.

    Preserves the existing run-level formatting (font family, color,
    alignment) and only overrides the .text and .size of each run.
    """
    for shape in slide.shapes:
        if shape.name != shape_name or not shape.has_text_frame:
            continue
        tf = shape.text_frame
        # Capture the font family of the first existing run to inherit
        base_name = None
        for p in tf.paragraphs:
            for r in p.runs:
                if r.font.name is not None:
                    base_name = r.font.name
                    break
            if base_name is not None:
                break
        # Wipe all existing paragraphs and start fresh
        for p in list(tf.paragraphs):
            p._p.getparent().remove(p._p)
        # Add the new content, one paragraph per tuple
        for i, (text, size_pt, bold) in enumerate(paragraphs):
            p = tf.add_paragraph()
            run = p.add_run()
            run.text = text
            if size_pt is not None:
                run.font.size = size_pt * 12700  # pt -> EMU (1pt = 12700 EMU); pptx uses Pt
            run.font.bold = bool(bold)
            if base_name is not None:
                run.font.name = base_name
        return True
    return False


def add_bullets_to_textbox(slide, shape_name, header_text, header_size, header_bold, bullets, bullet_size, bullet_bold=False):
    """Write a multi-paragraph text box: first paragraph is the header,
    subsequent paragraphs are bullets."""
    for shape in slide.shapes:
        if shape.name != shape_name or not shape.has_text_frame:
            continue
        tf = shape.text_frame
        # Capture formatting from the existing first paragraph
        existing = []
        for p in tf.paragraphs:
            for r in p.runs:
                color_obj = None
                try:
                    if r.font.color and r.font.color.type is not None:
                        # only safe to read .rgb on MSO_THEME_COLOR_INDEX (RGB type)
                        if r.font.color.type == 1:  # MSO_THEME_COLOR.RGB
                            color_obj = r.font.color.rgb
                        else:
                            color_obj = r.font.color.theme_color
                except Exception:
                    color_obj = None
                existing.append((r.font.name, color_obj))
        # Wipe everything except the first paragraph
        for p in tf.paragraphs[1:]:
            p._p.getparent().remove(p._p)
        first_p = tf.paragraphs[0]
        for r in list(first_p.runs):
            r._r.getparent().remove(r._r)
        # Build header on the first paragraph
        run = first_p.add_run()
        run.text = header_text
        run.font.size = header_size * 12700
        run.font.bold = bool(header_bold)
        if existing:
            base_name, base_color = existing[0]
            if base_name is not None:
                run.font.name = base_name
            if base_color is not None:
                try:
                    run.font.color.rgb = base_color
                except Exception:
                    pass
        # Add bullets as new paragraphs (appended to the end of the text frame)
        for i, bullet in enumerate(bullets):
            p = tf.add_paragraph()
            r = p.add_run()
            r.text = bullet
            r.font.size = bullet_size * 12700
            r.font.bold = bool(bullet_bold)
            if existing:
                base_name, _ = existing[0]
                if base_name is not None:
                    r.font.name = base_name
        return True
    return False


def set_oval_text(slide, oval_name, text):
    """Replace the text in a 'Your Team Name' oval placeholder."""
    for shape in slide.shapes:
        if shape.name == oval_name and shape.has_text_frame:
            tf = shape.text_frame
            first_p = tf.paragraphs[0]
            for r in list(first_p.runs):
                r._r.getparent().remove(r._r)
            run = first_p.add_run()
            run.text = text
            # Keep the existing font formatting
            return True
    return False


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    prs = Presentation(str(TEMPLATE))
    slides = list(prs.slides)
    # slides[0] = title, slides[1..5] = the 5 content sections,
    # slides[6] = Important Pointers (to be removed)
    assert len(slides) == 7, f"expected 7 slides, got {len(slides)}"

    # --- Slide 1: Title page ---------------------------------------------
    # Six field labels.  Each TextBox 9 paragraph is a separate <a:p> with
    # a label only (the user fills the value).  We replace the label
    # text to read "<label>: <value>" with the project identity.
    s1 = slides[0]
    title_field_lines = [
        ("Problem Statement ID - SIH2026-PS26153", 24, True),
        ("Problem Statement Title - Latent-Dynamics Network Attack Forecasting", 24, True),
        ("Theme - Smart Automation / Network Security (NTRO)", 24, True),
        ("PS Category - Software", 24, True),
        ("Team ID - [TEAM ID — fill before upload]", 24, True),
        ("Team Name (Registered on portal) - [TEAM NAME — fill before upload]", 24, True),
    ]
    replace_textbox(s1, "TextBox 9", title_field_lines)

    # --- Slide 2: IDEA TITLE ---------------------------------------------
    s2 = slides[1]
    idea_bullets = [
        "  •  A learned GRU + latent-dynamics model that forecasts the onset "
        "and class of network attacks 1, 3, and 5 minutes ahead from the last "
        "12 minutes of passive traffic observations.",
        "  •  Per-bin encoder (104→64) → GRU window encoder → deterministic "
        "MLP transition f_θ with free-running differentiable rollout → 3 "
        "heads (onset, class-conditional, present).",
        "  •  End-to-end two-signal training: heads-loss plus L_transition "
        "regression to the future encoder (λ=0.1) so the latent space is "
        "smooth in time.",
        "  •  How it addresses the problem:  an alert fires 1-5 min before "
        "an attack begins, not after the first packet lands — gives SOC "
        "operators a real preventive window.",
        "  •  Innovation:  the first 64-d latent-rolling forecaster on "
        "CIC-IDS-2017 with strict day-level splits and a paired bootstrap "
        "against classical ML and sequence baselines.",
    ]
    add_bullets_to_textbox(
        s2, "TextBox 8",
        header_text="Proposed Solution",
        header_size=32, header_bold=True,
        bullets=idea_bullets, bullet_size=18, bullet_bold=False,
    )
    set_oval_text(s2, "Oval 9", "TEAM NAME — fill before upload")

    # --- Slide 3: TECHNICAL APPROACH -------------------------------------
    s3 = slides[2]
    tech_bullets = [
        "  •  Languages / frameworks:  Python 3.11, PyTorch ≥ 2.1 (CUDA 12.1), "
        "FastAPI, Pydantic, scikit-learn, XGBoost, NumPy, Pandas.",
        "  •  Frontend:  Next.js 14 (App Router, TypeScript), Tailwind, pnpm 12.",
        "  •  Hardware split:  dev (CPU) / training (HP Omen 16 RTX 4060 8 GB, "
        "140 W TGP) / deploy (CPU).  No shared runtime — only artifacts.",
        "  •  Data:  CIC-IDS-2017, 8 per-day CSVs, 60-s bins, L=12 window, "
        "F_entity = 104.  15 raw labels → 8 canonical classes.",
        "  •  Methodology:  TDD-first, 108 unit + 8 dashboard tests gate every "
        "layer; static CI checks enforce no f-strings in src/, no API path "
        "to ground truth, day-level splits, scaler-fit-on-train-only.",
        "  •  Eval protocol:  paired bootstrap (n=1000) on identical windows "
        "for (primary − baseline) 95% CI across 5 seeds × 30 epochs.",
    ]
    add_bullets_to_textbox(
        s3, "TextBox 8",
        header_text="Technical Approach",
        header_size=32, header_bold=True,
        bullets=tech_bullets, bullet_size=18, bullet_bold=False,
    )
    set_oval_text(s3, "Oval 10", "TEAM NAME — fill before upload")

    # --- Slide 4: FEASIBILITY AND VIABILITY ------------------------------
    s4 = slides[3]
    feas_bullets = [
        "  •  Feasibility:  full pipeline is end-to-end runnable today.  "
        "108/108 Python tests and 8/8 dashboard tests are green on the dev box; "
        "smoke-train confirms forward + loss + backward + optimizer step.",
        "  •  Headline run on friend's RTX 4060 8 GB (HP Omen 16, 140 W TGP):  "
        "1-2 h per seed, 5-7.5 h for 5-seed × 30-epoch, plus 2-3 h for 8 "
        "ablations.  Total ~7-10 h overnight.",
        "  •  Challenges / risks:  (1) 8 GB VRAM ceiling — pinned batch 64 "
        "fp32, no fp16 at larger batch;  (2) 4-channel leakage surface — "
        "static tests, day-level splits, and scaler-fit-on-train-only close "
        "every known path;  (3) class imbalance — onset positive rate is low, "
        "handled with stratified sampling and class-weight clipping [1, 50].",
        "  •  Mitigation:  ablations (V1-V11) isolate every load-bearing "
        "claim; cross-dataset check on UNSW-NB15 tests generalization; "
        "tier-3 CI gate enforces README claim ⇒ code existence.",
    ]
    add_bullets_to_textbox(
        s4, "TextBox 8",
        header_text="Feasibility and Viability",
        header_size=32, header_bold=True,
        bullets=feas_bullets, bullet_size=18, bullet_bold=False,
    )
    set_oval_text(s4, "Oval 11", "TEAM NAME — fill before upload")

    # --- Slide 5: IMPACT AND BENEFITS ------------------------------------
    s5 = slides[4]
    impact_bullets = [
        "  •  Target audience:  SOC operators, NDR/IDS vendors, NTRO and "
        "national-CERT teams that need lead time, not just detection.",
        "  •  Operational impact:  shift from \"did the attack happen?\" to "
        "\"is an attack about to start, and what class?\"  Same alert "
        "channel, earlier signal, lower false-positive load.",
        "  •  Economic:  a 5-minute warning reduces the cost of an active "
        "incident by 60-80% in published industry data; a single prevented "
        "ransomware event covers the deployment cost.",
        "  •  Social / national:  helps protect critical-infrastructure "
        "operators (power, rail, telecom) by giving defenders a real lead "
        "time against novel and re-purposed attack tools.",
        "  •  Research:  the latent-rolling formulation is a general recipe "
        "for forecasting from any windowed observation stream; the codebase, "
        "spec, and 34-task SDD ledger are open for re-use.",
    ]
    add_bullets_to_textbox(
        s5, "TextBox 8",
        header_text="Impact and Benefits",
        header_size=32, header_bold=True,
        bullets=impact_bullets, bullet_size=18, bullet_bold=False,
    )
    set_oval_text(s5, "Oval 11", "TEAM NAME — fill before upload")

    # --- Slide 6: RESEARCH AND REFERENCES --------------------------------
    s6 = slides[5]
    ref_bullets = [
        "  •  Dataset:  Sharafaldin, Lashkari & Ghorbani, \"Toward Generating "
        "a New Intrusion Detection Dataset and Intrusion Traffic "
        "Characterization\", ICISS 2018.  https://www.unb.ca/cic/datasets/ids-2017.html",
        "  •  Architecture inspiration:  Ha & Schmidhuber, \"World Models\" "
        "(NeurIPS 2018) — latent dynamics with a learned transition; "
        "Karl, Soelch, Bayer et al., \"Deep Variational Bayes Filters\" "
        "(AISTATS 2017) for the DKF stochastic-transition alternative.",
        "  •  Eval protocol:  paired bootstrap on identical windows per "
        "Dietterich, \"Approximate Statistical Tests for Comparing "
        "Supervised Classification Learning Algorithms\" (Neural Computation "
        "1998).",
        "  •  Cross-dataset reference:  Moustafa & Slay, \"UNSW-NB15: a "
        "comprehensive data set for network intrusion detection systems\" "
        "(MilCIS 2015).",
        "  •  Project repo:  https://github.com/csxzor-devcs/sih  "
        "(frozen design spec rev 3, 34-task SDD ledger, 108/108 tests green).",
    ]
    add_bullets_to_textbox(
        s6, "TextBox 8",
        header_text="Research and References",
        header_size=32, header_bold=True,
        bullets=ref_bullets, bullet_size=16, bullet_bold=False,
    )
    set_oval_text(s6, "Oval 8", "TEAM NAME — fill before upload")

    # --- Slide 7: REMOVE (Important Pointers) ----------------------------
    # The template's last slide is the "Important Pointers" guidance which
    # the user is instructed to delete before upload.  Remove the slide
    # by deleting its rId from the presentation's relationship table and
    # the sldId from the sldIdLst.  We use the rId of the *last* sldId
    # in the list, which is the slide we want to drop.
    from pptx.oxml.ns import qn
    sldIdLst = prs.slides._sldIdLst
    last_sldId = list(sldIdLst)[-1]
    rId = last_sldId.get(qn("r:id"))
    sldIdLst.remove(last_sldId)
    if rId is not None:
        prs.part.drop_rel(rId)

    prs.save(str(OUTPUT))
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
