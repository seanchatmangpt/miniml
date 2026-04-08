# Clustering Tutorial

Unsupervised learning with micro-ml. Covers K-Means, DBSCAN, hierarchical clustering, and model evaluation.

## When to Use Clustering

- Customer segmentation
- Image compression (color quantization)
- Anomaly detection (outliers as noise)
- Exploratory data analysis
- Any problem where you want to find groups without labels

## K-Means

Partition data into k spherical clusters. Fast and simple.

```ts
import { kmeans } from 'micro-ml';

// Customer data: [annualSpend, visitFrequency, avgOrderValue]
const customers = [
  [100, 50, 2],    // low-value
  [5000, 200, 25], // high-value
  [200, 80, 3],    // low-value
  [8000, 300, 30], // high-value
  [150, 60, 2],    // low-value
  [6000, 250, 28], // high-value
];

const model = await kmeans(customers, { k: 2, maxIterations: 100 });

console.log(model.getAssignments());  // [0, 1, 0, 1, 0, 1]
console.log(model.getCentroids());    // [[150, 63, 2.3], [6333, 250, 27.7]]
console.log(model.inertia);          // Sum of squared distances to centroids

// Assign new customers
const newCustomers = [[300, 100, 5], [7000, 280, 27]];
const clusters = model.predict(newCustomers);
console.log(clusters);  // [0, 1]
```

## K-Means++

Same as K-Means but with smarter initialization (avoids poor starting centroids):

```ts
import { kmeansPlus } from 'micro-ml';

const result = kmeansPlus(
  new Float64Array(customers.flat()),
  3,   // n_features
  2,   // n_clusters
  100, // max_iter
);

// Result layout: [nClusters, assignments..., centroids_flat...]
const nClusters = result[0];
const nSamples = customers.length;
const assignments = result.slice(1, 1 + nSamples);
const centroidsStart = 1 + nSamples;
```

## DBSCAN

Density-based clustering. Finds arbitrary-shaped clusters and identifies noise points.

```ts
import { dbscan } from 'micro-ml';

const data = [
  [0, 0], [0.1, 0.1], [0, 0.2],        // cluster 1
  [10, 10], [10.1, 9.9], [9.9, 10.1],  // cluster 2
  [5, 5],                               // noise point
];

const result = await dbscan(data, { eps: 0.5, minPoints: 2 });

console.log(result.nClusters);  // 2
console.log(result.nNoise);     // 1
console.log(result.getLabels()); // [0, 0, 0, 1, 1, 1, -1]
```

### Choosing DBSCAN Parameters

- `eps`: Neighborhood radius. Too small → everything is noise. Too large → one big cluster.
  - Start with `eps` that captures ~5-10 nearest neighbors
- `minPoints`: Minimum points to form a cluster.
  - Rule of thumb: `minPoints >= dimensions + 1`
  - Typical: 3-10

## Hierarchical Clustering

Builds a tree of clusters (dendrogram). No need to specify k upfront.

### Single Linkage (minimum distance)

```ts
import { hierarchicalClustering } from 'micro-ml';

const labels = hierarchicalClustering(
  new Float64Array(data.flat()),
  3,   // n_features
  2,   // n_clusters
);
console.log(labels);  // [0, 0, 0, 1, 1, 1]
```

### Complete Linkage (maximum distance)

```ts
import { agglomerativeComplete } from 'micro-ml';

const labels = agglomerativeComplete(
  new Float64Array(data.flat()),
  3,
  2,
);
```

### When to Use Which Linkage

| Linkage | Cluster Shape | Use Case |
|---------|--------------|----------|
| Single | Elongated, chain-like | Finding paths |
| Complete | Compact, spherical | Well-separated clusters |

## Gaussian Mixture Models (GMM)

Soft clustering - each point has a probability of belonging to each cluster.

```ts
import { gmmFit } from 'micro-ml';

const model = gmmFit(
  new Float64Array(data.flat()),
  3,   // n_features
  2,   // n_clusters
  100, // max_iter
);

// Hard assignment
const clusters = model.predict(new Float64Array([0, 0]));
console.log(clusters);  // [0]

// Soft assignment (probabilities)
const probas = model.predictProba(new Float64Array([0, 0]));
console.log(probas);  // [0.9, 0.1] → 90% cluster 0, 10% cluster 1
```

## Spectral Clustering

For non-convex clusters (circles, moons, etc.):

```ts
import { spectralClustering } from 'micro-ml';

const labels = spectralClustering(
  new Float64Array(data.flat()),
  2,    // n_features
  2,    // n_clusters
  1.0,  // sigma (kernel bandwidth)
);
```

## Mini-Batch K-Means

For large datasets where full K-Means is too slow:

```ts
import { miniBatchKmeans } from 'micro-ml';

const result = miniBatchKmeans(
  new Float64Array(largeData.flat()),
  10,   // n_features
  5,    // n_clusters
  100,  // max_iter
  32,   // batch_size
);
```

## Evaluating Clusters

```ts
import { silhouetteScore } from 'micro-ml';

const data = new Float64Array(customers.flat());
const labels = new Float64Array([0, 1, 0, 1, 0, 1]);

const score = silhouetteScore(data, 3, labels);
console.log(score);  // Range: -1 to 1, higher = better clusters
```

### Interpreting Silhouette Score

| Score | Meaning |
|-------|---------|
| 0.7-1.0 | Strong, well-separated clusters |
| 0.5-0.7 | Reasonable clusters |
| 0.25-0.5 | Weak, overlapping clusters |
| < 0.25 | No meaningful structure |

## Choosing a Clustering Algorithm

| Situation | Algorithm |
|-----------|-----------|
| Know number of clusters, fast | K-Means, K-Means++ |
| Large dataset | Mini-Batch K-Means |
| Don't know number of clusters | DBSCAN, Hierarchical |
| Overlapping clusters | GMM |
| Non-convex shapes | Spectral, DBSCAN |
| Need probability per cluster | GMM |
| Identify outliers | DBSCAN (noise = -1) |

## Customer Segmentation Example

```ts
import { kmeans, silhouetteScore } from 'micro-ml';

// Try different k values
for (let k = 2; k <= 6; k++) {
  const model = await kmeans(customerData, { k, maxIterations: 200 });
  const labels = model.getAssignments();
  const score = silhouetteScore(
    new Float64Array(customerData.flat()),
    customerData[0].length,
    new Float64Array(labels),
  );
  console.log(`k=${k}: silhouette=${score.toFixed(3)}`);
}

// Pick k with highest silhouette score
```
