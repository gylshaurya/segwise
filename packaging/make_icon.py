"""Draw the segwise mark (a seismic trace on a dark rounded square) as a multi-size .ico."""
import sys

from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 10 / 32), fill=(17, 18, 20, 255))
pts = [(5, 16), (9, 16), (11, 10), (14, 23), (17, 8), (20, 20), (22, 16), (27, 16)]
pts = [(x * S / 32, y * S / 32) for x, y in pts]
d.line(pts, fill=(255, 255, 255, 255), width=int(S * 2.4 / 32), joint="curve")
r = S * 1.2 / 32
for x, y in (pts[0], pts[-1]):
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
out = sys.argv[1] if len(sys.argv) > 1 else "segwise.ico"
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.resize((256, 256), Image.LANCZOS).save(out.replace(".ico", ".png"))
