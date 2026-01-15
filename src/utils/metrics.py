import torch
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_map(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, filter_same_cam=True, batch_size=512):
    """
    Calculate Mean Average Precision (mAP) and Rank-1 for Re-ID.
    Uses batching to avoid massive memory allocation for large datasets.
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)
    
    num_q = q_feat.size(0)
    num_g = g_feat.size(0)
    
    all_ap = []
    all_cmc = []
    
    # Process queries in batches to save memory
    for start in range(0, num_q, batch_size):
        end = min(start + batch_size, num_q)
        q_batch = q_feat[start:end]
        
        # Calculate distances for this batch only
        # [batch_size, num_g]
        distmat_batch = torch.cdist(q_batch, g_feat).cpu().numpy()
        
        for i in range(end - start):
            q_idx = start + i
            dist = distmat_batch[i]
            
            # Sort gallery indices for this query
            indices = np.argsort(dist)
            
            # Re-ID evaluation protocol
            if filter_same_cam:
                # Standard Re-ID: exclude same person + same camera
                keep = ~((g_pids[indices] == q_pids[q_idx]) & (g_camids[indices] == q_camids[q_idx]))
            else:
                # Gait3D context: exclude only the query itself if query is in gallery
                keep = np.ones(num_g, dtype=bool)
                if torch.equal(q_feat, g_feat):
                    keep[indices == q_idx] = False
            
            filtered_indices = indices[keep]
            # Binary mask of matches for this query
            y_true = (g_pids[filtered_indices] == q_pids[q_idx]).astype(np.int32)
            
            if not np.any(y_true):
                continue
                
            # Score is negative distance
            y_score = -dist[filtered_indices]
            
            # AP
            ap = average_precision_score(y_true, y_score)
            all_ap.append(ap)
            
            # Rank-1 (CMC)
            if y_true[0] == 1:
                all_cmc.append(1)
            else:
                all_cmc.append(0)
                
    mAP = np.mean(all_ap) if all_ap else 0.0
    rank1 = np.mean(all_cmc) if all_cmc else 0.0
    
    return mAP, rank1

def calculate_inter_intra_ratio(features, labels):
    """
    Calculate the ratio of average inter-class distance to average intra-class distance.
    """
    # Normalize features to unit sphere
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    unique_labels = torch.unique(labels)
    centroids = []
    intra_distances = []
    
    for label in unique_labels:
        mask = (labels == label)
        class_features = features[mask]
        
        # Centroid calculation
        centroid = class_features.mean(dim=0)
        centroids.append(centroid)
        
        if class_features.size(0) > 1:
            # Distance to centroid: ||x - c|| = sqrt(||x||^2 + ||c||^2 - 2*x.c)
            centroid_norm = centroid.norm().item()
            dot_product = torch.matmul(class_features, centroid.unsqueeze(1)).squeeze(1)
            dists = torch.sqrt(torch.clamp(1 + centroid_norm**2 - 2*dot_product, min=1e-8))
            intra_distances.append(dists.mean().item())
            
    if not centroids:
        return 0.0, 0.0, 0.0
        
    centroids = torch.stack(centroids)
    
    # Inter-class distances
    if centroids.size(0) > 1:
        # 4000x4000 classes is only ~64MB
        inter_distmat = torch.cdist(centroids, centroids)
        inter_dist = inter_distmat[torch.triu(torch.ones_like(inter_distmat), diagonal=1) == 1].mean().item()
    else:
        inter_dist = 0.0
        
    avg_intra_dist = np.mean(intra_distances) if intra_distances else 0.0
    ratio = inter_dist / avg_intra_dist if avg_intra_dist > 0 else 0.0
    
    return ratio, inter_dist, avg_intra_dist
