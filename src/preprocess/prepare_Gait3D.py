# Ejemplo de comando de ejecución:
# python src/preprocess/prepare_Gait3D.py --raw-root "D:/datasets/Gait3D" --out-root "D:/datasets/Gait3D_processed" --sequence-len 25 --overlap 10

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def rel_to_out_root(out_root: Path, path: Path) -> str:
    return path.resolve().relative_to(out_root.resolve()).as_posix()


def parse_gait3d_filename(fname: str) -> Optional[int]:
    """
    Parse de: human_crop_f11627.png
    Retorna: frame_int
    """
    m = re.search(r"f(\d+)\.(png|jpg)$", fname, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def build_cycles_from_sorted_frames(
    frames: List[Tuple[int, Path]],
    seq_length: int,
    overlap: int,
) -> List[List[Tuple[int, Path]]]:
    if len(frames) < seq_length:
        return []
    step = seq_length - overlap
    if step <= 0:
        step = 1
    cycles: List[List[Tuple[int, Path]]] = []
    for i in range(0, len(frames) - seq_length + 1, step):
        chunk = frames[i : i + seq_length]
        if len(chunk) == seq_length:
            cycles.append(chunk)
    return cycles


def copy_cycle_silhouettes(
    cycle: List[Tuple[int, Path]],
    dst_cycle_dir: Path,
    overwrite: bool,
) -> Tuple[List[int], List[Path]]:
    dst_cycle_dir.mkdir(parents=True, exist_ok=True)
    frame_nums: List[int] = []
    dst_paths: List[Path] = []

    for fn, src in cycle:
        dst = dst_cycle_dir / f"{fn:06d}.png"
        if overwrite or (not dst.exists()):
            shutil.copy2(src, dst)
        frame_nums.append(fn)
        dst_paths.append(dst)

    return frame_nums, dst_paths


def main():
    ap = argparse.ArgumentParser(description="Organiza Gait3D (Siluetas) a estructura de ciclos y genera CSVs.")
    ap.add_argument("--raw-root", required=True, help="Raíz Gait3D (debe contener 2D_Silhouettes).")
    ap.add_argument("--out-root", required=True, help="Raíz de salida (crea silhouettes/ y metadata/).")
    ap.add_argument("--sequence-len", type=int, default=25)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing-cycles", action="store_true")

    # Debug/test
    ap.add_argument("--debug", action="store_true", help="Procesa solo los primeros 5 IDs.")
    ap.add_argument("--limit-cycles", type=int, default=None, help="Limita total de ciclos generados.")

    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    sil_raw_root = raw_root / "2D_Silhouettes"
    if not sil_raw_root.exists():
        raise FileNotFoundError(f"No existe la carpeta 2D_Silhouettes en: {raw_root}")

    sil_out_root = out_root / "silhouettes"
    meta_out_root = out_root / "metadata"
    meta_out_root.mkdir(parents=True, exist_ok=True)

    sequences_rows: List[Dict] = []
    frames_rows: List[Dict] = []
    total_cycles = 0

    # Gait3D no suele venir separado en carpetas train/test de forma obvia en disco por estructura,
    # pero a veces sí. Si está en 2D_Silhouettes directo exploramos los IDs.
    person_dirs = sorted([p for p in sil_raw_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))
    
    if args.debug:
        person_dirs = person_dirs[:5]
        print(f"[DEBUG] Procesando solo {len(person_dirs)} IDs.")

    print(f"\nProcesando {len(person_dirs)} IDs en {sil_raw_root} ...")

    for pid_dir in tqdm(person_dirs, desc="Processing IDs", unit="id"):
        # Estructura: 2D_Silhouettes / {ID} / {cam_video} / {seq}
        person_id = pid_dir.name
        
        # Iterar por cam_video (ej: camid0_videoid2)
        video_dirs = sorted([v for v in pid_dir.iterdir() if v.is_dir()], key=lambda v: natural_sort_key(v.name))
        
        for video_dir in video_dirs:
            scene = video_dir.name # camid0_videoid2
            
            # Iterar por seq (ej: seq0)
            seq_dirs = sorted([s for s in video_dir.iterdir() if s.is_dir()], key=lambda s: natural_sort_key(s.name))
            
            for seq_dir in seq_dirs:
                seq_id_name = seq_dir.name # seq0
                
                # Obtener imágenes
                imgs = sorted(seq_dir.glob("*.png"), key=lambda p: natural_sort_key(p.name))
                if not imgs:
                    continue
                
                frames_list = []
                for img_path in imgs:
                    fn = parse_gait3d_filename(img_path.name)
                    if fn is not None:
                        frames_list.append((fn, img_path))
                
                if not frames_list:
                    continue
                
                cycles = build_cycles_from_sorted_frames(
                    frames=frames_list,
                    seq_length=args.sequence_len,
                    overlap=args.overlap,
                )
                
                if not cycles:
                    continue

                for cyc_idx, cycle in enumerate(cycles, start=1):
                    total_cycles += 1
                    cycle_dirname = f"cycle{cyc_idx:04d}"
                    dst_cycle_dir = sil_out_root / person_id / scene / seq_id_name / cycle_dirname

                    if args.skip_existing_cycles and dst_cycle_dir.exists():
                        if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                            break
                        continue

                    frame_nums, dst_paths = copy_cycle_silhouettes(
                        cycle=cycle,
                        dst_cycle_dir=dst_cycle_dir,
                        overwrite=args.overwrite,
                    )

                    sequence_id = f"{person_id}_{scene}_{seq_id_name}_cyc{cyc_idx:04d}"

                    sequences_rows.append({
                        "sequence_id": sequence_id,
                        "person_global_id": person_id,
                        "dataset": "Gait3D",
                        "scene": scene,
                        "seq_id": seq_id_name,
                        "cycle_index": cyc_idx,
                        "silhouette_dir": rel_to_out_root(out_root, dst_cycle_dir),
                        "num_frames": len(frame_nums),
                        "frame_start": int(frame_nums[0]),
                        "frame_end": int(frame_nums[-1]),
                        "frame_numbers_json": json.dumps(frame_nums, ensure_ascii=False),
                    })

                    for fn, dst in zip(frame_nums, dst_paths):
                        frames_rows.append({
                            "sequence_id": sequence_id,
                            "person_global_id": person_id,
                            "scene": scene,
                            "seq_id": seq_id_name,
                            "cycle_index": cyc_idx,
                            "frame_number": int(fn),
                            "silhouette_path": rel_to_out_root(out_root, dst),
                        })

                    if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                        break

                if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                    break
            
            if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                break
        
        if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
            break

    if not sequences_rows:
        print("\n[WARN] No se generaron secuencias. Revisa la ruta de entrada y estructura.")
        return

    seq_df = pd.DataFrame(sequences_rows)
    frm_df = pd.DataFrame(frames_rows)

    seq_csv = meta_out_root / "sequences.csv"
    frm_csv = meta_out_root / "sequence_frames.csv"

    seq_df.to_csv(seq_csv, index=False, encoding="utf-8")
    frm_df.to_csv(frm_csv, index=False, encoding="utf-8")

    print("\nOK:")
    print(f"- Silhouettes root:    {sil_out_root}")
    print(f"- sequences.csv:       {seq_csv} (secuencias: {len(seq_df)})")
    print(f"- sequence_frames.csv: {frm_csv} (frames: {len(frm_df)})")


if __name__ == "__main__":
    main()
