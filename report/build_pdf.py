#!/usr/bin/env python3
"""Build report.pdf from report.md in the same directory.

Uses only fpdf2 (2.9.0) plus the standard library. Handles:
  # headings, ## headings, paragraphs, bullet lists (- ), Markdown tables.
Run:  python3 build_pdf.py
Output: report.pdf in this directory, plus the page count on stdout.
"""

import os
import re
import sys

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "report.md")
PDF_PATH = os.path.join(HERE, "report.pdf")

PAGE_W, PAGE_H = 210.0, 297.0          # A4 in mm
MARGIN = 20.0                          # left/right margin
TEXT_W = PAGE_W - 2 * MARGIN           # 170 mm
BODY_PT = 11
BODY_LH = 5.8                          # body line height in mm
TABLE_PT = 10
TABLE_LH = 5.0

# Characters fpdf2 core fonts cannot render, mapped to plain ASCII.
SANITIZE = {
    "\u2014": "-", "\u2013": "-", "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u2265": ">=", "\u2264": "<=",
    "\u2192": "->", "\u20b9": "Rs ", "\u00d7": "x", "\u2022": "-",
    "\u2026": "...", "\u00a0": " ",
}


def clean(text):
    for src, dst in SANITIZE.items():
        text = text.replace(src, dst)
    # Final guard: anything still outside cp1252 becomes "?".
    return text.encode("cp1252", "replace").decode("cp1252")


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, "Page %d/{nb}" % self.page_no(), align="C")


def wrap_lines(pdf, text, width):
    """Split text into lines that fit width (mm) at the current font."""
    lines = []
    for word in text.split(" "):
        if not word:
            continue
        if pdf.get_string_width(word) > width:
            # Long unbroken word (e.g. a sha256): break it by character.
            chunk = ""
            for ch in word:
                if pdf.get_string_width(chunk + ch) > width and chunk:
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                lines.append(chunk)
            continue
        if not lines:
            lines.append(word)
        elif pdf.get_string_width(lines[-1] + " " + word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines or [""]


def parse_markdown(path):
    """Return a list of blocks: ('title', text), ('h1', text),
    ('h2', text), ('para', text), ('bullets', [texts]), ('table', rows)."""
    with open(path, encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    blocks = []
    buf = []          # paragraph lines
    bullets = []
    table = []

    def flush_para():
        if buf:
            blocks.append(("para", clean(" ".join(s.strip() for s in buf))))
            buf.clear()

    def flush_bullets():
        if bullets:
            blocks.append(("bullets", [clean(b) for b in bullets]))
            bullets.clear()

    def flush_table():
        if table:
            blocks.append(("table", list(table)))
            table.clear()

    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para()
            flush_bullets()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # separator row
            table.append([clean(c) for c in cells])
        elif stripped.startswith("- "):
            flush_para()
            flush_table()
            bullets.append(stripped[2:].strip())
        elif stripped.startswith("# "):
            flush_para()
            flush_bullets()
            flush_table()
            blocks.append(("title", clean(stripped[2:].strip())))
        elif stripped.startswith("## "):
            flush_para()
            flush_bullets()
            flush_table()
            blocks.append(("h2", clean(stripped[3:].strip())))
        elif stripped == "":
            flush_para()
            flush_bullets()
            flush_table()
        else:
            flush_table()
            buf.append(stripped)
    flush_para()
    flush_bullets()
    flush_table()
    return blocks


def column_widths(pdf, rows):
    """Column widths in mm that fit TEXT_W. Narrow columns keep their
    full width; only the widest column (whose text wraps anyway) is
    shrunk when the total would overflow."""
    pdf.set_font("Helvetica", "", TABLE_PT)
    ncol = max(len(r) for r in rows)
    widths = []
    for col in range(ncol):
        need = 0.0
        for row in rows:
            if col < len(row):
                need = max(need, pdf.get_string_width(row[col]))
        widths.append(max(need + 4.0, 14.0))
    total = sum(widths)
    if total <= TEXT_W:
        extra = (TEXT_W - total) / ncol
        return [w + extra for w in widths]
    while total > TEXT_W:
        i = widths.index(max(widths))
        excess = total - TEXT_W
        cut = min(excess, max(0.0, widths[i] - 30.0))
        if cut <= 0:
            break
        widths[i] -= cut
        total -= cut
    return widths


def render_table(pdf, rows):
    pdf.set_font("Helvetica", "B", TABLE_PT)
    widths = column_widths(pdf, rows)
    header = rows[0]
    body = rows[1:]

    def draw_row(row, bold, fill):
        pdf.set_font("Helvetica", "B" if bold else "", TABLE_PT)
        pdf.set_fill_color(232, 232, 232)
        counts = []
        for i, cell in enumerate(row):
            w = widths[i] if i < len(widths) else widths[-1]
            counts.append(max(1, len(wrap_lines(pdf, cell, w - 3.0))))
        height = max(counts) * TABLE_LH + 1.0
        if pdf.get_y() + height > pdf.page_break_trigger:
            pdf.add_page()
            draw_row(header, True, True)
            if row is header:
                return  # already drawn by the recursive call
        y0 = pdf.get_y()
        x0 = pdf.l_margin
        for i, cell in enumerate(row):
            w = widths[i] if i < len(widths) else widths[-1]
            pdf.set_xy(x0, y0)
            pdf.multi_cell(w, TABLE_LH, cell, border=1, align="L",
                           fill=fill, new_x="RIGHT", new_y="TOP")
            x0 += w
        pdf.set_xy(pdf.l_margin, y0 + height)

    draw_row(header, True, True)
    for row in body:
        draw_row(row, False, False)
    pdf.ln(3)


def build():
    blocks = parse_markdown(MD_PATH)
    pdf = ReportPDF("P", "mm", "A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(MARGIN, 18, MARGIN)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "", BODY_PT)

    for kind, value in blocks:
        if kind == "title":
            pdf.set_font("Helvetica", "B", 17)
            pdf.multi_cell(0, 8, value, align="C")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", BODY_PT)
        elif kind == "h2":
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12.5)
            pdf.multi_cell(0, 6.5, value)
            pdf.ln(1)
            pdf.set_font("Helvetica", "", BODY_PT)
        elif kind == "para":
            pdf.multi_cell(0, BODY_LH, value, align="J")
            pdf.ln(2)
        elif kind == "bullets":
            for item in value:
                pdf.cell(6, BODY_LH, "-")
                pdf.multi_cell(TEXT_W - 6, BODY_LH, item, align="J",
                               new_x="LMARGIN", new_y="NEXT")
                pdf.ln(0.8)
            pdf.ln(2)
        elif kind == "table":
            render_table(pdf, value)

    pdf.output(PDF_PATH)
    print("Wrote %s" % PDF_PATH)
    print("Pages: %d" % pdf.pages_count)


if __name__ == "__main__":
    if not os.path.exists(MD_PATH):
        sys.exit("report.md not found next to build_pdf.py")
    build()
