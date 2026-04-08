# Choose K for K-Means

Determine the optimal number of clusters for your data.

## Problem

K-Means requires you to specify `k` (the number of clusters) upfront. Too few clusters merge distinct groups; too many split natural groups into fragments. You need a principled way to pick the right value.

## Solution

Use the silhouette score to evaluate cluster quality across different values of k. The silhouette score measures how well each point fits within its assigned cluster compared to neighboring clusters.

### Step 1: Try a range of k values

```typescript
import { kmeans, silhouetteScore, standardScaler } from "@seanchatmangpt/wminml";

const nSamples = X.length / nFeatures;

// Scale features first -- K-Means is distance-based
const { scaled } = standardScaler(X, nSamples, nFeatures);

const maxK = Math.min(10, Math.floor(nSamples / 2));
const results: { k: number; labels: Float64Array; score: number }[] = [];

for (let k = 2; k <= maxK; k++) {
  const { labels } = kmeans(scaled, nSamples, nFeatures, k, 100);
  const score = silhouetteScore(scaled, labels, nSamples, nFeatures, k);
  results.push({ k, labels, score });
  console.log(`k=${k}: silhouette=${score.toFixed(4)}`);
}
```

### Step 2: Pick the best k

```typescript
// Sort by silhouette score (higher is better, range -1 to 1)
results.sort((a, b) => b.score - a.score);
const best = results[0];
console.log(`\nBest k=${best.k} (silhouette=${best.score.toFixed(4)})`);

// Apply the final clustering with the optimal k
const { labels, centroids } = kmeans(
  scaled,
  nSamples,
  nFeatures,
  best.k,
  200
);
```

### Step 3: Inspect cluster sizes

After choosing k, verify the clusters are reasonable.

```typescript
const clusterCounts = new Map<number, number>();
for (let i = 0; i < labels.length; i++) {
  const cluster = labels[i];
  clusterCounts.set(cluster, (clusterCounts.get(cluster) || 0) + 1);
}

console.log("\nCluster sizes:");
for (const [cluster, count] of clusterCounts) {
  console.log(`  Cluster ${cluster}: ${count} samples`);
}

// Flag any suspiciously small or empty clusters
for (const [cluster, count] of clusterCounts) {
  if (count < 3) {
    console.log(
      `  WARNING: Cluster ${cluster} has only ${count} samples -- consider reducing k`
    );
  }
}
```

### Step 4: Use K-Means++ for better initialization

K-Means++ chooses initial centroids more intelligently than random initialization, leading to more stable results.

```typescript
import { kmeansPlusPlus } from "@seanchatmangpt/wminml";

const { labels: labelsPP } = kmeansPlusPlus(
  scaled,
  nSamples,
  nFeatures,
  best.k,
  100
);

const scorePP = silhouetteScore(scaled, labelsPP, nSamples, nFeatures, best.k);
console.log(`K-Means++ silhouette: ${scorePP.toFixed(4)}`);

// Compare with standard K-Means
const scoreStd = best.score;
console.log(`Standard K-Means silhouette: ${scoreStd.toFixed(4)}`);
```

### Interpreting silhouette scores

| Score Range | Interpretation |
|-------------|---------------|
| 0.7 - 1.0 | Strong, well-separated clusters |
| 0.5 - 0.7 | Reasonable structure |
| 0.25 - 0.5 | Weak structure, clusters overlap |
| < 0.25 | No meaningful structure (or wrong k) |
| < 0 | Points assigned to wrong cluster |

## Tips

- Always scale features before K-Means. Unscaled features with larger ranges dominate the distance calculation.
- Run K-Means multiple times with different random seeds if results vary. K-Means++ reduces this variance.
- If all silhouette scores are low (< 0.25), the data may not have natural clusters. Consider DBSCAN instead (see [Handle Arbitrary Shapes](dbscan.md)).
- Silhouette scores above 0.5 indicate reliable clustering. Above 0.7 is excellent.

## See Also

- [Handle Arbitrary Shapes](dbscan.md) -- when clusters are not spherical
- [Hierarchical Clustering](hierarchical.md) -- alternative that does not require k upfront
- [Scale Your Features](../preprocessing/scaling.md) -- essential before K-Means
