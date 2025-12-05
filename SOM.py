import numpy as np


class SOM:
    def __init__(self, width, height, input_dim, n_iters, lr0):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.n_iters = n_iters
        self.lr0 = lr0

        # Total neurons
        self.M = width * height

        # Weights: (M, D)
        self.W = np.random.randn(self.M, self.input_dim)

        # Grid coordinates for each neuron index: (M, 2)
        self.coordinates = np.zeros((self.M, 2), dtype=int)
        idx = 0
        for r in range(self.height):
            for c in range(self.width):
                self.coordinates[idx, 0] = r
                self.coordinates[idx, 1] = c
                idx += 1

    def predict_distances(self, X):
        """
        Return distances from each row of X to each neuron weight.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape N, {self.input_dim}")

        N = X.shape[0]
        distances = np.zeros((N, self.M), dtype=float)

        for i in range(N):
            x = X[i]
            for j in range(self.M):
                diff = x - self.W[j]
                distances[i, j] = np.sqrt(np.sum(diff * diff))

        return distances

    def predict(self, X):
        """
        Return BMU indices for each row in X.
        """
        distances = self.predict_distances(X)
        return np.argmin(distances, axis=1)

    def fit(self, X):
        """
        Batch SOM training using Gaussian neighborhood and exponential decay.

        For each iteration:
          1. Find BMU for each sample
          2. Compute neighborhood influence for each neuron relative to each sample's BMU
          3. Build batch target weights as weighted averages
          4. Blend current weights toward batch target using learning rate
        """

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape N, {self.input_dim}")

        N = X.shape[0]
        sigma0 = max(self.width, self.height) / 2.0

        for t in range(self.n_iters):
            lr = self.lr0 * np.exp(-t / self.n_iters)
            sigma = sigma0 * np.exp(-t / self.n_iters)
            sigma = max(sigma, 1e-6)

            bmus = self.predict(X)                   
            bmu_coordinates = self.coordinates[bmus]       

            W_batch = np.zeros_like(self.W)           
            weight_sum = np.zeros(self.M, dtype=float)  

            for i in range(N):
                x = X[i]
                ri, ci = bmu_coordinates[i]

                for j in range(self.M):
                    rj, cj = self.coordinates[j]
                    dr = rj - ri
                    dc = cj - ci
                    d2 = dr * dr + dc * dc

                    h = np.exp(-d2 / (2.0 * sigma * sigma))

                    W_batch[j] += h * x
                    weight_sum[j] += h

            for j in range(self.M):
                if weight_sum[j] > 0:
                    W_batch[j] /= weight_sum[j]
                else:
                    W_batch[j] = self.W[j] 

            self.W = (1.0 - lr) * self.W + lr * W_batch

        return self


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 400
    cluster1 = rng.normal(loc=(0.0, 0.0), scale=0.35, size=(N // 2, 2))
    cluster2 = rng.normal(loc=(3.0, 3.0), scale=0.35, size=(N // 2, 2))
    X = np.vstack([cluster1, cluster2])

    som = SOM(width=6, height=6, input_dim=2, n_iters=20, lr0=0.5)

    # BEFORE
    bmus0 = som.predict(X)
    hits0 = np.bincount(bmus0, minlength=som.M).reshape(som.height, som.width)
    qe0 = np.mean(np.min(som.predict_distances(X), axis=1))

    som.fit(X)

    # AFTER
    bmus1 = som.predict(X)
    hits1 = np.bincount(bmus1, minlength=som.M).reshape(som.height, som.width)
    qe1 = np.mean(np.min(som.predict_distances(X), axis=1))

    print("Hit map BEFORE:")
    print(hits0)
    print("\nHit map AFTER:")
    print(hits1)

    print("\nTotal samples:", X.shape[0])
    print("Sum of hit map:", hits1.sum())
    print("Active before:", np.count_nonzero(hits0), "out of", som.M)
    print("Active after :", np.count_nonzero(hits1), "out of", som.M)
    print("Quantization error before:", qe0, "and after:", qe1)
