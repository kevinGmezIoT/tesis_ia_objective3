import numpy as np
import torch, cv2, glob
from pathlib import Path

def load_sequence(seq_dir, frame_num=30, img_size=112):
    paths = sorted(glob.glob(str(Path(seq_dir)/"*.png")))
    if len(paths)==0:
        raise RuntimeError("No hay PNGs en la carpeta.")

    if len(paths) >= frame_num:
        idx = np.linspace(0, len(paths)-1, frame_num, dtype=int)
    else:
        base = np.arange(len(paths))
        extra = np.random.choice(len(paths), frame_num-len(paths), True)
        idx = np.concatenate([base,extra])

    frames = []
    for i in idx:
        m = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img_size,img_size), cv2.INTER_NEAREST)
        m = (m>127).astype(np.float32)[None,:,:]
        frames.append(m)

    clip = np.stack(frames,0)          # [T,1,H,W]
    clip = clip.transpose(1,0,2,3)     # [1,T,H,W]
    clip = clip[None]                  # [1,1,T,H,W]
    return torch.from_numpy(clip).float()

def embedding(model, seq_dir, frame_num, img_size, device="cuda"):
    model.eval()
    clip = load_sequence(seq_dir, frame_num, img_size).to(device)
    with torch.no_grad():
        out = model(clip)
        if isinstance(out, tuple):
            out = out[0]
        n = out.size(0)
        emb = out.view(n,-1).mean(0)
    return emb.cpu().numpy()
