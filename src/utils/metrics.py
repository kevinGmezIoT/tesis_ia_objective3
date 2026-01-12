import torch
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_map(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, filter_same_cam=True):
    """
    Calculate Mean Average Precision (mAP) and Rank-1 for Re-ID.
    Each query feature is compared against all gallery features.
    """
    # Use torch.cdist for efficient distance calculation
    # Features are expected to be (N, D)
    distmat = torch.cdist(q_feat, g_feat).cpu().numpy()
    
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)
    
    num_q, num_g = distmat.shape
    
    # Sort gallery indices by distance (ascending)
    indices = np.argsort(distmat, axis=1)
    
    # Binary mask of matches
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_ap = []
    all_cmc = []
    
    # Check if query and gallery are likely the same set
    is_self_gallery = (num_q == num_g and torch.equal(q_feat, g_feat))

    for i in range(num_q):
        # Re-ID evaluation protocol: filter out samples 
        if filter_same_cam:
            # Standard Re-ID: exclude same person + same camera
            keep = ~((g_pids[indices[i]] == q_pids[i]) & (g_camids[indices[i]] == q_camids[i]))
        else:
            # Gait3D context: we want to match cycles of the same person
            # But we MUST exclude the query sequence itself if it's in the gallery
            keep = np.ones(num_g, dtype=bool)
            if is_self_gallery:
                # Find the position of the query index 'i' in the sorted 'indices[i]'
                # and mark it as False in 'keep'
                # The query is at indices[i] where value == i
                keep[indices[i] == i] = False
        
        y_true = matches[i][keep]
        # Score is negative distance (higher similarity = higher score)
        y_score = -distmat[i][indices[i]][keep]
        
        if not np.any(y_true):
            continue
            
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
    Higher values indicate better feature discrimination.
    """
    # Normalize features to unit sphere for distance calculations
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    unique_labels = torch.unique(labels)
    centroids = []
    intra_distances = []
    
    for label in unique_labels:
        mask = (labels == label)
        class_features = features[mask]
        
        # Calculate class centroid
        centroid = class_features.mean(dim=0)
        centroids.append(centroid)
        
        if class_features.size(0) > 1:
            # Average distance from each point in class to its centroid
            dists = torch.cdist(class_features, centroid.unsqueeze(0))
            intra_distances.append(dists.mean().item())
            
    if not centroids:
        return 0.0, 0.0, 0.0
        
    centroids = torch.stack(centroids)
    
    # Inter-class distances (Euclidean distance between class centroids)
    if centroids.size(0) > 1:
        inter_distmat = torch.cdist(centroids, centroids)
        # Take mean of upper triangle entries (excluding diagonal)
        inter_dist = inter_distmat[torch.triu(torch.ones_like(inter_distmat), diagonal=1) == 1].mean().item()
    else:
        inter_dist = 0.0
        
    avg_intra_dist = np.mean(intra_distances) if intra_distances else 0.0
    
    # Avoid division by zero
    ratio = inter_dist / avg_intra_dist if avg_intra_dist > 0 else 0.0
    
    return ratio, inter_dist, avg_intra_dist
