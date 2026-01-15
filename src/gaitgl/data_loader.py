import os
import pandas as pd
from pathlib import Path
from .dataset import DataSet

def load_data(data_root, index_csv, resolution):
    """
    Carga los datos basándose en el CSV pre-split.
    Retorna train_src, test_src, val_src.
    """
    print(f"Leyendo índice de datos desde: {index_csv}")
    df = pd.read_csv(index_csv)

    def create_dataset(subset_df):
        if subset_df.empty:
            return None
        
        # El dataset espera una lista de listas para seq_dir por compatibilidad heredada
        # (originalmente podía recibir múltiples backbones/fuentes para una misma secuencia)
        seq_dirs = [[str(Path(data_root) / row['silhouette_dir'])] for _, row in subset_df.iterrows()]
        labels = subset_df['person_id'].astype(str).tolist()
        
        # Ajuste de compatibilidad refinado:
        # Priorizamos 'cam_id' como la VISTA para evaluación cross-camera (mAP)
        if 'cam_id' in subset_df.columns:
            views = subset_df['cam_id'].tolist()
            # En Gait3D refactorizado, usamos video_id o seq_id como tipo
            seq_types = subset_df['video_id'].tolist() if 'video_id' in subset_df.columns else subset_df['seq_id'].tolist()
        else:
            # Fallback para datasets antiguos o el dataset propio sin 'cam_id' (aunque suele tenerlo)
            seq_types = subset_df['seq_id'].tolist()
            views = subset_df['scene'].tolist()

        return DataSet(
            seq_dirs,
            labels,
            seq_types,
            views,
            cache=True, 
            resolution=resolution, 
            cut=False # El CSV ya apunta a ciclos recortados si fuera el caso
        )

    print("Creando split de TRAIN...")
    train_src = create_dataset(df[df['split'] == 'train'])
    
    print("Creando split de TEST...")
    test_src = create_dataset(df[df['split'] == 'test'])
    
    print("Creando split de VAL...")
    val_src = create_dataset(df[df['split'] == 'val'])

    return train_src, test_src, val_src
