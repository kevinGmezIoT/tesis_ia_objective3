import yaml
import torch
import argparse
from pathlib import Path
from model.network.vgg_c3d import c3d_vgg_Fusion
from gaitgl.inference_utils import embedding

def main():
    parser = argparse.ArgumentParser(description='Inference GaitGL')
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    print("Inicializando red neuronal...")
    # Initialize the network directly, not the Model trainer wrapper
    model = c3d_vgg_Fusion(num_classes=conf["train_pid_num"])
    
    ckpt_path = conf["checkpoint"]
    if Path(ckpt_path).exists():
        print(f"Cargando checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        
        # Handle DataParallel module. prefix
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        
        model.load_state_dict(new_state)
    else:
        print(f"ADVERTENCIA: No se encontró checkpoint en {ckpt_path}. Usando pesos aleatorios.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    seq_dir = input("Ruta de carpeta con PNGs: ").strip()
    if not seq_dir:
        print("Ruta vacía.")
        return

    try:
        emb = embedding(model, seq_dir, conf["frame_num"], conf["img_size"], device=device)
        print("Vector embedding:", emb.shape)
        print(emb)
    except Exception as e:
        print(f"Error durante inferencia: {e}")

if __name__ == "__main__":
    main()
