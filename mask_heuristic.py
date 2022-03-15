import torch
import numpy as np

device='cuda:0'

# return true if all neighbors of a pixel is set to 1
def is_patch(mask, i, j):
    patch = True
    for m in range(-1, 2, 1):
        for n in range(-1, 2, 1):
            x = i + m
            y = j + n
            if (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]):
                if mask[x, y] == 0:
                    patch = False
    return patch

# get greedy mask
# x: sensitivity matrix
def generate_greedy_mask(x, pixel_thresh=1000, patch_thresh=10):
    print("generate greedy mask")
    sensitivity_matrix = x.cpu().detach().numpy()
    greedy_mask = torch.zeros(sensitivity_matrix.shape, dtype=torch.float32)
    pixel_count = 0
    patch_count = 0
    while pixel_count < pixel_thresh:
        index = np.unravel_index(np.argmax(sensitivity_matrix, axis=None), sensitivity_matrix.shape)
        if not (is_patch(greedy_mask, index[0], index[1]) and patch_count >= patch_thresh):
            greedy_mask[index[0], index[1]] = 1
            pixel_count += 1
            if is_patch(greedy_mask, index[0], index[1]):
                patch_count += 1
        sensitivity_matrix[index[0], index[1]] = 0
    
    print("greedy_mask pixel count", torch.sum(greedy_mask))
    return greedy_mask

# get a mask full of 1s. i.e. equivalent to not using mask
def generate_full_mask():
    print("generate full mask")
    return torch.full((500, 500), 1, dtype=torch.float32).to(device)

# get random mask
def generate_random_mask(height, width, pixel_thresh=1000):
    print("generate random mask")
    pixel_count = height * width
    while pixel_count > pixel_thresh:
        random_mask = (torch.cuda.FloatTensor(height, width).uniform_() > (1- pixel_thresh/(height*width))).to(torch.float32)
        pixel_count = torch.sum(random_mask)
    #plt.savefig('./knn_mask_random.jpg')
    print("random mask pixel count", torch.sum(random_mask))

    return random_mask

def get_k_candidates(mask, sensitivity_matrix, score, patch_count, k, patch_thresh):
    topk_masks = mask.unsqueeze(0).repeat(k,1,1)
    topk_sensitivity_matrix = np.repeat(sensitivity_matrix[np.newaxis, :, :], k, axis=0)
    temp = np.copy(sensitivity_matrix)
    scores = [score]*k
    patch_counts = [patch_count]*k
    mask_count = 0
    while mask_count < k:
        idx = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
        if not (is_patch(mask, idx[0], idx[1]) and patch_count >= patch_thresh):
            topk_masks[mask_count, idx[0], idx[1]] = 1
            topk_sensitivity_matrix[mask_count, idx[0], idx[1]] = 0
            scores[mask_count] += temp[idx[0], idx[1]]
            mask_count += 1
            if is_patch(mask, idx[0], idx[1]):
                patch_counts[mask_count] += 1
        temp[idx[0], idx[1]] = 0
    return topk_masks, topk_sensitivity_matrix, scores, patch_counts

def get_topk_idx(scores, k):
    scores = np.array(scores)
    return scores.argsort()[-k:][::-1].tolist()

def get_new_topk(topk_masks, topk_sensitivity_matrix, scores, patch_counts, k, patch_thresh):
    candidate_masks = []
    candidate_sensitivity_matrix = []
    candidate_scores = []
    candidate_patch_counts = []
    for i in range(topk_masks.shape[0]):
        m, sm, s, pc = get_k_candidates(topk_masks[i], topk_sensitivity_matrix[i], scores[i], patch_counts[i], k, patch_thresh)
        candidate_masks.append(m)
        candidate_sensitivity_matrix.append(sm)
        candidate_scores += s
        candidate_patch_counts += pc
    topk_idx = get_topk_idx(candidate_scores, k)
    for i in range(topk_masks.shape[0]):
        x = topk_idx[i] // k
        y = topk_idx[i] % k
        topk_masks[i] = candidate_masks[x][y]
        topk_sensitivity_matrix[i] = candidate_sensitivity_matrix[x][y]
        scores[i] = candidate_scores[topk_idx[i]]
        patch_counts[i] = candidate_patch_counts[topk_idx[i]]
    return topk_masks, topk_sensitivity_matrix, scores, patch_counts

def initialzie_beam_search(sensitivity_matrix, k, patch_thresh):
  mask = torch.zeros(sensitivity_matrix.shape, dtype=torch.float32)
  return get_k_candidates(mask, sensitivity_matrix, 0, 0, k, patch_thresh)
  
# get beam mask
# x: sensitivity matrix
# k: number of beams
def generate_beam_mask(x, k=3, pixel_thresh=1000, patch_thresh=10):
    print("generate beam mask")
    sensitivity_matrix = x.cpu().detach().numpy()
    topk_masks, topk_sensitivity_matrix, scores, patch_counts = initialzie_beam_search(sensitivity_matrix, k, patch_thresh)
    for i in range(int(pixel_thresh)):
        topk_masks, topk_sensitivity_matrix, scores, patch_counts = get_new_topk(topk_masks, topk_sensitivity_matrix, scores, patch_counts, k, patch_thresh)
    return topk_masks[scores.index(max(scores))]

