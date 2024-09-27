using Embedding.DistanceMetric;
using Embedding.EmbeddingVector;

namespace Clustering;

public sealed class StreamingClusterClassifier<TVector, TDistanceMetric>(
    float similarityThreshold,
    float centroidDriftResistance
) : IClusterClassifier<TVector>
    where TVector : IEmbeddingVector<TVector>
    where TDistanceMetric : IDistanceMetric<TVector>
{
    private readonly Dictionary<Guid, HashSet<int>> _clusterAssignments = [];
    private readonly List<(Guid id, TVector centroid)> _clusterCentroids = [];
    private readonly float _similarityThreshold = 1 - similarityThreshold;

    public ClusteringResult Classify(IEnumerable<TVector> dataset)
    {
        List<int> vectorHashes = [];
        foreach (var vector in dataset)
        {
            vectorHashes.Add(vector.GetHashCode());
            var (index, clusterId, distance) = FindClosestCluster(vector);
            if (clusterId == Guid.Empty || distance > _similarityThreshold)
            {
                var guid = Guid.NewGuid();
                _clusterCentroids.Add((guid, vector));
                _clusterAssignments.Add(guid, [vector.GetHashCode()]);
                continue;
            }

            _ = _clusterAssignments[clusterId].Add(vector.GetHashCode());

            var (id, centroid) = _clusterCentroids[index];
            _clusterCentroids[index] = (id, centroid.GetCentroidVector(vector, centroidDriftResistance));
        }

        MergeClusters();

        List<Guid> labels = [];
        List<float> outlierScores = [];
        foreach (int hash in vectorHashes)
        {
            var (clusterId, members) = _clusterAssignments.First(kvp => kvp.Value.Contains(hash));
            labels.Add(clusterId);
            outlierScores.Add(1f / members.Count);
        }

        return new(labels, outlierScores);
    }
    private (int, Guid, float) FindClosestCluster(TVector vector)
    {
        return _clusterCentroids
            .Select((cluster, index) => (index, cluster.id, vector.DistanceTo(cluster.centroid, TDistanceMetric.CalculateUnitSphereDistance)))
            .DefaultIfEmpty()
            .MinBy(tup => tup.Item3);
    }

    private void MergeClusters()
    {
        List<List<(int index, Guid id)>> clustersToMerge;
        do
        {
            clustersToMerge = GetConnectedClusters();
            foreach (var cluster in clustersToMerge)
            {
                var (baseClusterIndex, baseClusterId) = cluster.Last();
                var newClusterCentroid = _clusterCentroids[baseClusterIndex].centroid;
                var newClusterAssignments = _clusterAssignments[baseClusterId];

                foreach (var (clusterIndex, clusterId) in cluster.SkipLast(1))
                {
                    newClusterAssignments.UnionWith(_clusterAssignments[clusterId]);
                    _ = _clusterAssignments.Remove(clusterId);

                    var centroid = _clusterCentroids[clusterIndex].centroid;
                    newClusterCentroid = newClusterCentroid.GetCentroidVector(centroid, centroidDriftResistance);
                    _clusterCentroids.RemoveAt(clusterIndex);
                }

                _ = _clusterAssignments.Remove(baseClusterId);
                _clusterCentroids.RemoveAt(baseClusterIndex);

                var newClusterId = Guid.NewGuid();
                _clusterCentroids.Add((newClusterId, newClusterCentroid));
                _clusterAssignments.Add(newClusterId, newClusterAssignments);
            }
        } while (clustersToMerge.Count > 0);
    }

    private List<List<(int, Guid)>> GetConnectedClusters()
    {
        List<List<(int, Guid)>> clusterTrees = [];
        HashSet<Guid> visited = [];
        for (int clusterIndex = 0; clusterIndex < _clusterCentroids.Count; ++clusterIndex)
        {
            var (id, centroid) = _clusterCentroids[clusterIndex];
            if (!visited.Add(id))
            {
                continue;
            }

            List<(int, Guid)> connectedClusters = [];

            for (int otherClusterIndex = 0; otherClusterIndex < _clusterCentroids.Count; ++otherClusterIndex)
            {
                var (otherClusterId, otherClusterCentroid) = _clusterCentroids[otherClusterIndex];
                if (id != otherClusterId
                    && !visited.Contains(otherClusterId)
                    && centroid.DistanceTo(otherClusterCentroid, TDistanceMetric.CalculateUnitSphereDistance) <= _similarityThreshold)
                {
                    connectedClusters.Add((otherClusterIndex, otherClusterId));
                    _ = visited.Add(otherClusterId);
                }
            }

            if (connectedClusters.Count > 0)
            {
                connectedClusters.Add((clusterIndex, id));
                clusterTrees.Add(connectedClusters);
            }
        }

        return clusterTrees;
    }
}
