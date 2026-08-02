"""Render assets/demo.gif — an animated terminal capture of the xpref demo.

Draws the real ``xpref eval`` + ``xpref attach --replay`` output as a typed
terminal (scrolling viewport, syntax-highlighted numbers, blinking cursor),
loops forever. Pillow is a build-time only dep (the shipped artifact is the
gif; the canonical re-render path is docs/demo.tape + demo.yml via vhs).

Run:  python scripts/gen_demo_gif.py   (needs Pillow)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("assets/demo.gif")

# --- canvas -----------------------------------------------------------------
W, H = 780, 470
LINE_H = 18
PAD_X = 16
PAD_TOP = 34  # below the title bar
BG = (13, 17, 23)          # #0d1117 github dark
TITLE_BG = (21, 27, 35)
DOT_RED = (255, 95, 86)
DOT_YEL = (254, 188, 64)
DOT_GRN = (44, 187, 99)
FG = (201, 209, 217)       # gray
GREEN = (126, 231, 135)
BLUE = (88, 166, 255)
PURPLE = (210, 168, 255)
AMBER = (227, 179, 65)
DIM = (139, 148, 158)

# --- the demo script (captured from the real CLI) ---------------------------
LINES: list[tuple[str, str]] = [
    ("$ xpref eval --trace samples/k3-q4-128tok.bin", "prompt"),
    ("xpref eval — k3-q4-128tok.bin", "header"),
    ("  tokens=128 layers=8 experts=896 active=16", "muted"),
    ("  recall@16 = 0.7405", "result"),
    ("  reactive t/s  = 4.0", "amber"),
    ("  xpref  t/s   = 12.00  (3.00x)", "green"),
    ("  per-layer recall:", "muted"),
    ("    layer 0: 0.7451", "purple"),
    ("    layer 2: 0.7367", "purple"),
    ("    layer 4: 0.7406", "purple"),
    ("    layer 6: 0.7426", "purple"),
    ("", "blank"),
    ("$ xpref attach --replay samples/k3-q4-128tok.bin --checkpoint ckpt.gguf", "prompt"),
    ("xpref attach — replay k3-q4-128tok.bin  [live madvise]", "header"),
    ("  checkpoint=k3-q4.gguf  experts=896 active=16 layers=8 tokens=128", "muted"),
    ("   tok   recall  pred/act  proj t/s   prefetch", "muted"),
    ("     0    0.758   12/16       12.00        4 KB", "data"),
    ("    16    0.742   12/16       12.00        4 KB", "data"),
    ("    32    0.734   12/16       12.00        4 KB", "data"),
    ("    48    0.727   12/16       12.00        4 KB", "data"),
    ("    64    0.773   12/16       12.00        4 KB", "data"),
    ("    80    0.750   12/16       12.00        4 KB", "data"),
    ("   104    0.773   12/16       12.00        4 KB", "data"),
    ("   120    0.734   12/16       12.00        4 KB", "data"),
    ("   126    0.773   12/16       12.00        4 KB", "data"),
    ("", "blank"),
    ("replay done: tokens=127 recall=0.741 proj t/s=12.00 prefetched=585216 bytes", "green"),
]

VIEWPORT = (H - PAD_TOP - 8) // LINE_H


def _load_font(size: int):
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/System/Library/Fonts/Courier.ttc", "/Library/Fonts/Menlo.ttc"):
        try:
            from PIL import ImageFont
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    from PIL import ImageFont
    return ImageFont.load_default()


def _token_color(tok: str, kind: str):
    if kind == "prompt":
        return GREEN if tok == "$" else FG
    if kind == "header":
        return BLUE
    if kind == "green":
        return GREEN
    if kind == "amber":
        return AMBER
    if kind == "purple":
        return PURPLE
    if kind == "result":
        return BLUE
    if kind == "muted":
        return DIM
    if kind == "data":
        # numbers green, "12/16" green, t/s value green, rest dim
        is_num = tok.replace(".", "", 1).replace("/", "", 1).isdigit() or tok.endswith("KB")
        return GREEN if is_num else DIM
    return FG


def _draw_line(draw, font, text, kind, y):
    x = PAD_X
    if kind == "prompt":
        # split "$" from the rest
        head, _, rest = text.partition(" ")
        draw.text((x, y), head, font=font, fill=GREEN)
        x += font.getlength(head) + font.getlength(" ")
        draw.text((x, y), rest, font=font, fill=FG)
        x += font.getlength(rest)
        return x
    # tokenise and colour inline (data rows get green numbers)
    for tok in text.split(" "):
        col = _token_color(tok, kind)
        draw.text((x, y), tok, font=font, fill=col)
        x += font.getlength(tok)
        x += font.getlength(" ")
    return x


def _render(revealed: int, cursor_on: bool, font) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # title bar
    draw.rectangle([0, 0, W, 26], fill=TITLE_BG)
    for i, c in enumerate((DOT_RED, DOT_YEL, DOT_GRN)):
        draw.ellipse([12 + i * 18, 8, 12 + i * 18 + 10, 18], fill=c)
    draw.text((66, 6), "xpref — predict + prefetch MoE experts (4 → 12 t/s)",
              font=font, fill=FG)
    shown = LINES[:revealed]
    if len(shown) > VIEWPORT:
        shown = shown[-VIEWPORT:]
    y = PAD_TOP
    for text, kind in shown:
        end_x = _draw_line(draw, font, text, kind, y)
        y += LINE_H
    if cursor_on and shown:
        last_text, last_kind = shown[-1]
        if last_text:
            cx = end_x + 4
            draw.rectangle([cx, y - LINE_H + 3, cx + 9, y - 3], fill=GREEN)
        else:
            draw.rectangle([PAD_X, y - LINE_H + 3, PAD_X + 9, y - 3], fill=GREEN)
    return img


def main() -> None:
    font = _load_font(14)
    frames = []
    n = len(LINES)
    # reveal one line per frame
    for i in range(1, n + 1):
        frames.append(_render(i, True, font))
    # blinking-cursor tail (loop seam)
    for k in range(14):
        frames.append(_render(n, k % 2 == 0, font))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=120, loop=0, disposal=2, optimize=True,
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes, {len(frames)} frames)")


if __name__ == "__main__":
    main()
