import numpy as np

def _to_vec(x):
    if x is None:
        return None
    try:
        v = np.asarray(x, dtype=np.float32)
        if v.ndim > 1:
            v = v.reshape(-1)
        n = np.linalg.norm(v) + 1e-8
        return v / n
    except Exception:
        return None

def cosine_sim(a, b):
    a = _to_vec(a); b = _to_vec(b)
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))

def group_unknown(unknown_db: dict, similarity_threshold: float = 0.85):
    """
    unknown_db: {unknown_id: {'embedding': vector, 'photos': [paths], ...}}
    Returns (new_db, merged_count)
    """
    ids = list(unknown_db.keys())
    visited = set()
    new_db = {}
    merged = 0

    for i, uid in enumerate(ids):
        if uid in visited:
            continue
        base = unknown_db[uid]
        base_vec = base.get('embedding')
        group = [uid]
        visited.add(uid)

        for j in range(i+1, len(ids)):
            vid = ids[j]
            if vid in visited:
                continue
            vec = unknown_db[vid].get('embedding')
            if base_vec is not None and vec is not None and cosine_sim(base_vec, vec) >= similarity_threshold:
                group.append(vid)
                visited.add(vid)

        if len(group) > 1:
            merged += len(group) - 1
        photos = []
        vecs = []
        for gid in group:
            item = unknown_db[gid]
            photos.extend(item.get('photos', []) or [])
            v = _to_vec(item.get('embedding'))
            if v is not None:
                vecs.append(v)
        avg_vec = None
        if vecs:
            avg_vec = (np.mean(np.stack(vecs, axis=0), axis=0)).astype(np.float32)

        new_db[uid] = dict(unknown_db[uid])
        if photos:
            new_db[uid]['photos'] = photos
        if avg_vec is not None:
            new_db[uid]['embedding'] = avg_vec

    return new_db, merged
