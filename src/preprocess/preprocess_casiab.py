import cv2, numpy as np, re, shutil
from pathlib import Path

file_re = re.compile(
    r"^(\d+)-(nm|bg|cl)-(\d+)-(\d+)-(\d+)\.png$",
    re.IGNORECASE
)

def preprocess_casia_b(src_root, dst_root, out_size=64):

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    pngs = list(src_root.rglob("*.png"))
    print("PNG encontrados:", len(pngs))

    for p in pngs:
        m = file_re.match(p.name)
        if not m:
            continue

        pid, cond, seq, view, frame = m.groups()
        if pid == "005":
            continue

        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = mask_bin[y1:y2+1, x1:x2+1]
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        outdir = dst_root / pid / cond / view / seq
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{p.name}"
        cv2.imwrite(str(outpath), crop)

    print("Preprocesamiento terminado.")
