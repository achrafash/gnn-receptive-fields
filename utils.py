import torch


def mean_average_distance(x, mask=None):
    '''
    Computes cosine similarity between each pair of node representations
    Args:
        x (torch.tensor): node features
        mask (torch.tensor): mask matrix composed of 0 and 1 only if the node pair (i,j) is the target one.
    '''

    MAD_num = 0
    MAD_den = 0

    for i in range(x.size(0)):
        D_avg_num = 0
        D_avg_den = 0
        for j in range(i):
            cos_sim = 0
            if mask == None or (mask != None and mask[i, j]):
                cos_sim = 1 - \
                    torch.dot(x[i, :], x[j, :]) / (torch.norm(x[i, :])
                                                   * torch.norm(x[j, :]) + 1e-16)
            D_avg_num += cos_sim
            D_avg_den += 1 if cos_sim > 0 else 0

        if D_avg_den:
            D_avg = D_avg_num / D_avg_den
            MAD_num += D_avg_num
            MAD_den += 1 if D_avg > 0 else 0

    MAD = MAD_num / MAD_den
    return MAD


def mean_average_distance_gap(x, adj_matrix):
    '''
    Computes MADGap: MAD_remote - MAD_neighbors
    Args:
        x (torch.tensor): input feature matrix
        adj_matrix (torch.tensor): adjacency matrix
    '''
    MAD_rem = mean_average_distance(x, mask=1-adj_matrix)
    MAD_neb = mean_average_distance(x, mask=adj_matrix)

    return MAD_rem - MAD_neb


def influence_distribution(x, h):
    '''
    Computes influence distribution
    ...math: I_x(y) = e^T \left[ \frac{ \partial h_x^{(k)} }{ \partial h_y^{(0)} } \right] e / \left( \sum_{z \in V} e^T \left[ \frac{ \partial h_x^{(k)} }{ \partial h_z^{(0)} } \right] e \right)

    Args:
        h (torch.tensor): hidden feature matrix
        x (torch.tensor): input feature matrix
    '''
    e = torch.ones(x.size(1), 1)

    # jacob_xy =
    # I[x, y] = torch.dot(torch.dot(torch.transpose(e, 0, 1), jacob_xy), e)
