import pathlib
from PIL import Image
src = pathlib.Path("frontend/examples")
dst = src / "thumbs"; dst.mkdir(exist_ok=True)
for f in sorted(src.glob("*.png")):
    im = Image.open(f).convert("RGB")
    im.thumbnail((160, 160))
    im.save(dst / (f.stem + ".jpg"), quality=88)
    print("thumb:", f.name)
