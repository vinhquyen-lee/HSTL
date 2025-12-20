import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import CircleLoss as PMLCircleLoss
from .base import BaseLoss, gather_and_scale_wrapper

'''
This is wrapper for Circle Loss from pytorch-metric-learning library:
https://kevinmusgrave.github.io/pytorch-metric-learning

some critical points:

1. Normalization:
    - The original embeddings from the model is Batch Normalized features.
    - This is a layer (nn.BatchNorm1d) that shifts and scales the distribution of features to have mean 0 and variance 1 across the batch.
        Result: It stabilizes training, but it does not guarantee that the vector length is 1. One vector might have length 5.0, another 12.0.
    - If you feed un-normalized vectors into Circle Loss, the math breaks because the dot product can be arbitrarily large (e.g., 500.0),
      whereas cosine similarity must be between -1 and 1.
    - Circle Loss requires cosine similarity as input, so we L2-normalize the "bnft" embeddings before passing to the loss function.


2. Reshaping (ignore for HSTL partitioning on body)
    - HSTL output embeddings and labels as:
        - embeddings: [n, v, c], where n=batch size, v=number of views (sequence length), c=feature dimension.
        - labels: [n]

    - Circle Loss expects: [batch_size, feature_dimension], with one label per embedding.

    expample:
        n = 4  # 4 people in batch
        v = 3  # 3 spatial parts per person
        c = 2  # 2D embeddings (simplified)

        embeddings = [
            [[1.2, 0.5],  # Person 0, Part 0
            [0.8, 1.1],  # Person 0, Part 1
            [1.0, 0.9]], # Person 0, Part 2

            [[1.1, 0.6],  # Person 1, Part 0
            [0.9, 1.0],  # Person 1, Part 1
            [1.2, 0.8]], # Person 1, Part 2

            # ... Person 2, 3
        ]
        # Shape: [4, 3, 2]

        labels = [0, 1, 2, 3]  # 4 different people

    what we do:
        embeddings_flat = embeddings.reshape(n * v, c)
        # [4, 3, 2] -> [12, 2]

            [
                [1.2, 0.5],  # Person 0, Part 0
                [0.8, 1.1],  # Person 0, Part 1
                [1.0, 0.9],  # Person 0, Part 2
                [1.1, 0.6],  # Person 1, Part 0
                [0.9, 1.0],  # Person 1, Part 1
                [1.2, 0.8],  # Person 1, Part 2
                # ... Person 2, 3 parts
            ]
            # Now 12 vectors total (4 people x 3 parts)

        labels_flat = labels.unsqueeze(1).repeat(1, v).reshape(n * v)
        # [4] -> [4, 1] -> [4, 3] -> [12]

            [0, 0, 0,  # All 3 parts belong to Person 0
            1, 1, 1,  # All 3 parts belong to Person 1
            2, 2, 2,  # All 3 parts belong to Person 2
            3, 3, 3]  # All 3 parts belong to Person 3
'''

class CircleLoss(BaseLoss):
    def __init__(self, m=0.25, gamma=128, loss_term_weight=1.0):
        super(CircleLoss, self).__init__(loss_term_weight)
        self.m = m
        self.gamma = gamma
        self.loss_fn = PMLCircleLoss(m=m, gamma=gamma)
        # print(f"DEBUG: Initialized CircleLoss with m={m}, gamma={gamma}, loss_term_weight={loss_term_weight}")
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, v, c], labels: [n]
        n, v, c = embeddings.shape

        # CRITICAL: L2-normalize for cosine similarity
        # Circle Loss requires unit vectors (||x|| = 1)
        # bnft from BatchNorm has arbitrary magnitudes
        # L2-normalize along channel dimension (dim=2)
        embeddings_normalized = F.normalize(embeddings, p=2, dim=2)

        # Compute loss independently for each body part
        '''
        Compute loss per-part independently,
         preserving semantic distinctiveness of each body region.
          This matches TripletLoss behavior.
        '''
        total_loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        for i in range(v):
            # Extract i-th part: [n, c]
            part_embeddings = embeddings_normalized[:, i, :]

            # Compute Circle Loss for this part
            part_loss = self.loss_fn(part_embeddings, labels)
            total_loss += part_loss
        # Average across parts (like TripletLoss does)
        avg_loss = total_loss / v

        # Match BaseLoss interface
        self.info.update({'loss': avg_loss.detach().clone()})

        return avg_loss.mean(), self.info