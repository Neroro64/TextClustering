using Embedding;
using Embedding.DistanceMetric;

namespace Clustering;

public sealed class StreamingClusterClassifier<TVector, TDistanceMetric>(float similarityThreshold) : IClusterClassifier<TVector>
    where TVector : IEmbeddingVector<TVector>
    where TDistanceMetric : IDistanceMetric<TVector>
{
    private readonly Dictionary<Guid, HashSet<int>> _clusterAssignments = [];
    private readonly List<(Guid id, TVector centroid)> _clusterCentroids = [];

    public ClusteringResult Classify(IEnumerable<TVector> dataset)
    {
        foreach (var vector in dataset)
        {
            var (clusterId, distance) = FindClosestCluster(vector);
            if (clusterId == Guid.Empty || distance > similarityThreshold)
            {
                var guid = Guid.NewGuid();
                _clusterCentroids.Add((guid, vector));
                _clusterAssignments.Add(guid, [vector.GetHashCode()]);
                continue;
            }
            _ = _clusterAssignments[clusterId].Add(vector.GetHashCode());
        }
        return new([], []);
    }
    private (Guid, float) FindClosestCluster(TVector vector)
    {
        return _clusterCentroids
            .Select(cluster => (cluster.id, vector.DistanceTo(cluster.centroid, TDistanceMetric.CalculateDistance)))
            .DefaultIfEmpty()
            .MinBy(tup => tup.Item2);
    }
}
