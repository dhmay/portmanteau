"""Build a slide deck from an executed Jupyter notebook.

The notebook is the source. Its heading structure is the slide format, so a number on a slide is
produced by code sitting next to the prose that describes it, and a slide is edited by editing a
cell rather than a builder script:

    # Deck title              -> the opening section slide (that line only)
    ## Slide title            -> one slide
    ### prep                  -> code whose output is never rendered: set-up, intermediate frames
    ### text                  -> markdown, which becomes the slide's bullets
    ### images                -> code whose outputs become pictures and tables
    ### notes                 -> markdown, which becomes the speaker notes

A `###` heading opens a section that runs to the next heading, so a markdown section may put its
body in the heading cell or in the cells after it. Anything before the first `##` is a preamble -
imports, loaders - and is ignored.

**Stored outputs are used as they are; nothing is re-executed.** A build takes about a second, at
the cost of trusting the last kernel run, so `read_notebook` refuses a notebook that cannot have
been run cleanly end to end: any error output, any unexecuted code cell, any `### images` cell that
produced nothing. Re-run before building::

    jupyter nbconvert --to notebook --execute --inplace deck.ipynb

**Layout is fixed by design.** The point is that slides cannot drift from the data, not that they
can look like anything:

- bullets take the left `text_fraction` of the content area, figures the right;
- one figure is centred vertically in that column, two are stacked;
- a **table** instead puts the bullets on top and the table full width underneath, because a table
  of numbers is wide and a column of bullets is not;
- three or more figures, or a table and a figure on one slide, raise `DeckError` - split the slide.

Text is written at an explicit point size and checked against its box. A template's
``<a:normAutofit/>`` is not honoured in a generated file - python-pptx writes no ``fontScale``, so
the stored state says 100% and text renders full size over whatever is beneath it. `build_deck`
estimates the rendered height and raises, naming the slide and section, so the error says which cell
to shorten. Tables are checked against the bottom of the slide for the same reason.

Typical use::

    from portmanteau.slides.notebook_deck import build_deck

    build_deck("chunk1.ipynb", out="chunk1.pptx", template="house_style.pptx")

Needs the ``slides`` extra: ``pip install portmanteau[slides]``.
"""

from __future__ import annotations

import base64
import math
import re
import tempfile
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Iterable

__all__ = ["DeckError", "DeckStyle", "Slide", "build_deck", "read_notebook", "inline_runs",
           "parse_bullets"]

SECTIONS = ("prep", "text", "images", "notes")
EMU_PER_INCH = 914400


class DeckError(Exception):
    """A notebook the builder will not render, or a slide that will not fit.

    Always raised with the notebook, slide and section in the message: the point of the check is to
    say which cell to change.
    """


@dataclass
class DeckStyle:
    """Template-dependent knobs. The defaults suit python-pptx's built-in template.

    The content area is read from the content layout's body placeholder rather than hardcoded, so a
    house template with different margins needs no configuration here.
    """

    content_layout: str = "Title and Content"
    section_layout: str = "Section Header"
    body_idx: int = 1
    body_pt: int = 17
    #: fraction of the content width given to bullets when figures sit beside them
    text_fraction: float = 0.46
    column_gap: float = 0.3
    #: how close to the bottom of the slide a figure or table may come. The layout's body
    #: placeholder is the box for *text*; a table sits below it and may legitimately run further,
    #: so the limit for figures and tables is the slide itself less this margin.
    bottom_margin: float = 0.45
    table_row_height: float = 0.32
    table_pt: int = 13
    table_gap: float = 0.3
    max_figures: int = 2
    slide_numbers: bool = True


@dataclass
class Slide:
    """One `##` heading and the cells beneath it, grouped by section."""

    title: str
    cells: dict = field(default_factory=lambda: {name: [] for name in SECTIONS})


# --------------------------------------------------------------------------- reading

def read_notebook(path) -> tuple[str, list[Slide]]:
    """Parse an executed notebook into (deck title, slides), validating as it goes.

    The deck title is the notebook's own `# ` heading - that one line, not the cell, whose remaining
    paragraphs usually explain the format to whoever edits the notebook and have no place on a
    slide.

    Raises:
        DeckError: on an unknown `###` section, a slide with neither text nor images, a cell that
            errored or was never executed, or an `### images` cell that produced no output.
    """
    import nbformat

    path = Path(path)
    notebook = nbformat.read(str(path), as_version=4)
    slides: list[Slide] = []
    current: Slide | None = None
    section: str | None = None
    deck_title: str | None = None

    for cell in notebook.cells:
        source = cell.source.strip()
        if cell.cell_type == "markdown":
            heading = source.split("\n", 1)[0].strip()
            body = source.split("\n", 1)[1].strip() if "\n" in source else ""
            if heading.startswith("# ") and deck_title is None:
                deck_title = heading[2:].strip()
                continue
            if heading.startswith("## ") and not heading.startswith("### "):
                current = Slide(title=heading[3:].strip())
                slides.append(current)
                section = None
                continue
            if heading.startswith("### "):
                name = heading[4:].strip().lower()
                if current is None:
                    raise DeckError(f"{path.name}: '### {name}' appears before any '## ' slide")
                if name not in SECTIONS:
                    raise DeckError(f"{path.name} / '{current.title}': unknown section "
                                    f"'### {name}'; expected one of {list(SECTIONS)}")
                section = name
                if body:
                    current.cells[name].append(("markdown", body, cell))
                continue
        if current is None or section is None:
            continue                            # preamble, or a stray cell before any section
        current.cells[section].append((cell.cell_type, cell.source, cell))

    if deck_title is None:
        raise DeckError(f"{path.name}: no '# ' heading to take the deck title from")
    for slide in slides:
        _validate(path.name, slide)
    return deck_title, slides


def _validate(source: str, slide: Slide) -> None:
    where = f"{source} / '{slide.title}'"
    for name in SECTIONS:
        for kind, _, cell in slide.cells[name]:
            if kind != "code":
                continue
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    raise DeckError(f"{where} / {name}: cell raised {output.get('ename')}. "
                                    "Re-run the notebook before building.")
            if cell.get("execution_count") is None:
                raise DeckError(f"{where} / {name}: cell was never executed. "
                                "Re-run the notebook before building.")
    if not slide.cells["text"] and not slide.cells["images"]:
        raise DeckError(f"{where}: slide has neither a '### text' nor an '### images' section")
    for kind, _, cell in slide.cells["images"]:
        if kind == "code" and not cell.get("outputs"):
            raise DeckError(f"{where} / images: cell produced no output. An '### images' cell must "
                            "display a figure or a frame; move set-up into '### prep'.")


# --------------------------------------------------------------------------- markdown

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def inline_runs(text: str) -> list[tuple[str, bool, bool]]:
    """Split a line into (text, bold, italic) runs. Backticked code is kept verbatim.

    python-pptx writes runs verbatim, so `**bold**` would otherwise arrive on the slide as four
    asterisks.
    """
    runs = []
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            runs.append((piece[2:-2], True, False))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            runs.append((piece[1:-1], False, True))
        elif piece.startswith("`") and piece.endswith("`"):
            runs.append((piece[1:-1], False, False))
        else:
            runs.append((piece, False, False))
    return runs


def parse_bullets(blocks: Iterable[tuple[str, str, object]]) -> list[tuple[list, int, bool]]:
    """Markdown to (runs, level, is_bullet). Two spaces of indent is one level.

    Two rules that are easy to get wrong and both have bitten:

    - **A blank line ends whatever is open.** Without it, a bold lead-in following a list is read as
      a continuation of the last bullet and silently merged into it.
    - **Continuations are joined before inline markup is parsed.** Parsing line by line misses
      emphasis that wraps: `*close` on one line and `the cut*` on the next is one span in markdown
      and two unmatched asterisks to a line-at-a-time parser.

    A line that is not a list item becomes a paragraph with no bullet glyph, which is how a bold
    lead-in sits above its own bullets.
    """
    items: list[tuple[list, int, bool]] = []
    pending: tuple[str, int, bool] | None = None

    def flush():
        nonlocal pending
        if pending:
            text, level, is_bullet = pending
            items.append((inline_runs(text), level, is_bullet))
            pending = None

    for kind, source, _ in blocks:
        if kind != "markdown":
            continue
        for line in source.split("\n"):
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            match = re.match(r"^(\s*)[-*]\s+(.*)$", line)
            if match:
                flush()
                pending = (match.group(2), min(len(match.group(1)) // 2, 4), True)
            elif pending:
                text, level, is_bullet = pending
                pending = (f"{text} {stripped}", level, is_bullet)
            else:
                pending = (stripped, 0, False)
        flush()
    return items


def _notes_text(blocks) -> str:
    return "\n\n".join(source for kind, source, _ in blocks if kind == "markdown").strip()


# --------------------------------------------------------------------------- outputs

def _header(column) -> str:
    """A readable column name from whatever `read_html` produced.

    A frame with both a named index and named columns comes back with a MultiIndex whose filler
    levels are "Unnamed: 0_level_0" and the like, which would otherwise be printed on the slide.
    """
    levels = column if isinstance(column, tuple) else (column,)
    parts = [str(level) for level in levels if not str(level).startswith("Unnamed")]
    return " ".join(dict.fromkeys(parts))


def _collect_outputs(blocks, figure_dir: Path, stem: str):
    """Pictures and frames from an `### images` section, in the order they were displayed."""
    import pandas as pd

    pictures, tables = [], []
    for kind, _, cell in blocks:
        if kind != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                path = figure_dir / f"{stem}_{len(pictures)}.png"
                path.write_bytes(base64.b64decode(data["image/png"]))
                pictures.append(path)
            elif "text/html" in data:
                # thousands=None: the notebook already formatted its numbers as strings, and the
                # default would parse "1,110,640" back to an int and render it unseparated. The
                # slide should show exactly what the cell showed.
                frame = pd.read_html(StringIO(data["text/html"]), thousands=None)[0]
                frame.columns = [_header(column) for column in frame.columns]
                tables.append(frame.fillna(""))
    return pictures, tables


# --------------------------------------------------------------------------- geometry

def estimate_height(bullets, width_inches: float, size: int) -> float:
    """Estimated rendered height of a bullet list, in inches.

    Deliberately crude and slightly conservative: 0.50 em average character width, 1.22 line
    spacing, 0.12" between paragraphs, 0.42" of indent per level. It guards against the silent
    overflow a generated deck is prone to; it is not a typesetter.
    """
    needed = 0.0
    for position, item in enumerate(bullets):
        runs, level = item[0], item[1]
        points = size if level == 0 else size - 2
        usable = max(0.5, width_inches - 0.42 * level)
        per_line = max(10, usable * 72 / (0.50 * points))
        text = "".join(run[0] for run in runs)
        needed += max(1, math.ceil(len(text) / per_line)) * 1.22 * points / 72
        if position:
            needed += 0.12
    return needed


def _suppress_bullet(para) -> None:
    """Remove the list glyph from one paragraph, for lead-in lines that are not list items."""
    from pptx.oxml.ns import qn

    properties = para._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for existing in properties.findall(qn(tag)):
            properties.remove(existing)
    properties.append(properties.makeelement(qn("a:buNone"), {}))


def _write_bullets(placeholder, bullets, size: int) -> None:
    from pptx.util import Pt

    frame = placeholder.text_frame
    frame.clear()
    for index, (runs, level, is_bullet) in enumerate(bullets):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.level = level
        if not is_bullet:
            _suppress_bullet(para)
        for text, bold, italic in runs:
            run = para.add_run()
            run.text = text
            run.font.bold, run.font.italic = bold, italic
            run.font.size = Pt(size if level == 0 else size - 2)


def _place_picture(slide_obj, path: Path, left, top, width, height) -> None:
    """Scale to fit the given box, centred in it."""
    from PIL import Image
    from pptx.util import Inches

    with Image.open(path) as image:
        aspect = image.height / image.width
    draw_width = min(width, height / aspect)
    draw_height = draw_width * aspect
    slide_obj.shapes.add_picture(
        str(path), Inches(left + (width - draw_width) / 2),
        Inches(top + (height - draw_height) / 2), width=Inches(draw_width))


def _place_table(slide_obj, frame, left, top, width, style: DeckStyle) -> float:
    """A real pptx table. Returns its bottom edge, which the caller checks against the slide."""
    from pptx.util import Inches, Pt

    rows, columns = len(frame) + 1, len(frame.columns)
    height = style.table_row_height * rows
    table = slide_obj.shapes.add_table(rows, columns, Inches(left), Inches(top),
                                       Inches(width), Inches(height)).table
    for column, name in enumerate(frame.columns):
        run = table.cell(0, column).text_frame.paragraphs[0].add_run()
        run.text = str(name)
        run.font.size, run.font.bold = Pt(style.table_pt), True
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        for column, value in enumerate(row):
            run = table.cell(row_index, column).text_frame.paragraphs[0].add_run()
            run.text = str(value)
            run.font.size = Pt(style.table_pt)
    return top + height


def _slide_number(slide_obj) -> None:
    """Fill the template's slide-number placeholder; python-pptx creates it empty.

    Found by placeholder *type* rather than index, which differs between templates.
    """
    from lxml import etree
    from pptx.enum.shapes import PP_PLACEHOLDER

    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for placeholder in slide_obj.placeholders:
        if placeholder.placeholder_format.type != PP_PLACEHOLDER.SLIDE_NUMBER:
            continue
        body = placeholder._element.find(f"{a}txBody")
        if body is None:
            continue
        for paragraph in body.findall(f"{a}p"):
            body.remove(paragraph)
        paragraph = etree.SubElement(body, f"{a}p")
        field = etree.SubElement(paragraph, f"{a}fld")
        field.set("id", "{B6F15528-21DE-4FA0-A08B-E82E22E4F0C2}")
        field.set("type", "slidenum")
        etree.SubElement(field, f"{a}t").text = " "


def _strip_sample_slides(prs) -> None:
    """Remove any slides the template ships with, leaving its layouts and theme."""
    ids = prs.slides._sldIdLst
    for element in list(ids):
        rid = element.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rid)
        ids.remove(element)


def _content_box(layout, style: DeckStyle) -> tuple[float, float, float, float]:
    """(left, top, width, height) of the content layout's body placeholder, in inches."""
    for placeholder in layout.placeholders:
        if placeholder.placeholder_format.idx == style.body_idx:
            return (placeholder.left / EMU_PER_INCH, placeholder.top / EMU_PER_INCH,
                    placeholder.width / EMU_PER_INCH, placeholder.height / EMU_PER_INCH)
    raise DeckError(f"layout '{layout.name}' has no placeholder at idx {style.body_idx}; "
                    "set DeckStyle.body_idx to the body placeholder of your template")


# --------------------------------------------------------------------------- building

def _build_slide(prs, layouts, slide: Slide, figure_dir: Path, stem: str, source: str,
                 style: DeckStyle) -> None:
    from pptx.util import Inches

    pictures, tables = _collect_outputs(slide.cells["images"], figure_dir, stem)
    where = f"{source} / '{slide.title}'"
    if tables and pictures:
        raise DeckError(f"{where}: a table and a figure on one slide is not a layout this builds - "
                        "split the slide.")
    if len(pictures) > style.max_figures:
        raise DeckError(f"{where}: {len(pictures)} figures on one slide; at most "
                        f"{style.max_figures} - split the slide.")

    layout = layouts[style.content_layout]
    slide_obj = prs.slides.add_slide(layout)
    slide_obj.shapes.title.text_frame.paragraphs[0].add_run().text = slide.title
    body = slide_obj.placeholders[style.body_idx]
    bullets = parse_bullets(slide.cells["text"])

    left, top, width, height = _content_box(layout, style)
    text_bottom = top + height
    media_bottom = prs.slide_height / EMU_PER_INCH - style.bottom_margin

    if pictures:
        text_width = width * style.text_fraction
        body.left, body.top = Inches(left), Inches(top)
        body.width, body.height = Inches(text_width), Inches(text_bottom - top)
        column_left = left + text_width + style.column_gap
        column_width = width - text_width - style.column_gap
        column_height = media_bottom - top
        if len(pictures) == 1:
            _place_picture(slide_obj, pictures[0], column_left, top, column_width, column_height)
        else:
            each = (column_height - 0.2) / len(pictures)
            for index, picture in enumerate(pictures):
                _place_picture(slide_obj, picture, column_left, top + index * (each + 0.2),
                               column_width, each)
    elif tables:
        needed = max(0.4, estimate_height(bullets, width, style.body_pt) + 0.15)
        body.left, body.top = Inches(left), Inches(top)
        body.width, body.height = Inches(width), Inches(needed)
        table_top = top + needed + style.table_gap
        for frame in tables:
            table_top = _place_table(slide_obj, frame, left, table_top, width, style) + 0.25
        overhang = (table_top - 0.25) - media_bottom
        if overhang > 0:
            raise DeckError(
                f"{where}: the table runs {overhang:.2f}\" off the bottom of the slide "
                f"({sum(len(f) for f in tables)} rows, plus headers). Shorten the frame in "
                "'### images' or the bullets above it, or split the slide.")
    else:
        body.left, body.top = Inches(left), Inches(top)
        body.width, body.height = Inches(width), Inches(text_bottom - top)

    _write_bullets(body, bullets, style.body_pt)
    if style.slide_numbers:
        _slide_number(slide_obj)
    notes = _notes_text(slide.cells["notes"])
    if notes:
        slide_obj.notes_slide.notes_text_frame.text = notes


def _add_section_slide(prs, layouts, title: str, style: DeckStyle) -> None:
    slide_obj = prs.slides.add_slide(layouts[style.section_layout])
    if slide_obj.shapes.title is not None:
        slide_obj.shapes.title.text_frame.paragraphs[0].add_run().text = title
        return
    placeholder = slide_obj.placeholders[style.body_idx]
    frame = placeholder.text_frame
    frame.clear()
    frame.paragraphs[0].add_run().text = title
    _suppress_bullet(frame.paragraphs[0])


def _check_overflow(prs, source: str, titles: list[str], style: DeckStyle) -> None:
    problems = []
    for index, slide_obj in enumerate(prs.slides):
        if slide_obj.slide_layout.name == style.section_layout:
            continue                       # its text is a title at the layout's own size
        for shape in slide_obj.shapes:
            if not (shape.is_placeholder and shape.placeholder_format.idx == style.body_idx):
                continue
            if not shape.has_text_frame or not shape.text_frame.text.strip():
                continue
            width = shape.width / EMU_PER_INCH
            height = shape.height / EMU_PER_INCH
            bullets = [([(para.text, False, False)], para.level, True)
                       for para in shape.text_frame.paragraphs]
            needed = estimate_height(bullets, width, style.body_pt)
            if needed > height:
                title = titles[index - 1] if index else titles[0]
                problems.append(f"  {source} / '{title}' / ### text: needs ~{needed:.2f}\" in a "
                                f"{height:.2f}\" box (over by {needed - height:.2f}\") - shorten "
                                "the bullets")
    if problems:
        raise DeckError("text overflows its placeholder:\n" + "\n".join(problems))


def build_deck(notebook, out, template=None, style: DeckStyle | None = None,
               figure_dir=None, section_slide: bool = True) -> Path:
    """Render an executed notebook as a pptx deck.

    Args:
        notebook: path to the executed notebook.
        out: path to write the deck to.
        template: a .pptx whose layouts and theme to use, or None for python-pptx's default.
        style: layout knobs; see `DeckStyle`.
        figure_dir: where to put PNGs pulled out of the notebook. A temporary directory by default,
            which is enough because python-pptx embeds a picture when it is added.
        section_slide: whether to open the deck with a slide carrying the notebook's `# ` title.

    Returns:
        The path written.

    Raises:
        DeckError: for anything that would otherwise produce a quietly wrong slide - see
            `read_notebook` and the layout rules in the module docstring.
    """
    from pptx import Presentation

    style = style or DeckStyle()
    deck_title, slides = read_notebook(notebook)
    prs = Presentation(str(template) if template else None)
    layouts = {layout.name: layout for layout in prs.slide_layouts}
    for needed in (style.content_layout, style.section_layout):
        if needed not in layouts:
            raise DeckError(f"template has no layout named '{needed}'; it has {sorted(layouts)}")
    _strip_sample_slides(prs)

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(figure_dir) if figure_dir else Path(scratch)
        directory.mkdir(parents=True, exist_ok=True)
        if section_slide:
            _add_section_slide(prs, layouts, deck_title, style)
        for index, slide in enumerate(slides):
            _build_slide(prs, layouts, slide, directory, f"slide{index:02d}",
                         Path(notebook).name, style)
        _check_overflow(prs, Path(notebook).name, [s.title for s in slides], style)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(out))
    return out
