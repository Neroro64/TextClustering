using Embedding;

namespace Clustering.StreamingClusteringAlgorithm;

public class StreamingClusterBuilder<TVector> : IClusterBuilder<TVector>
{
    public ClusteringResult Build(IEnumerable<IEmbeddingVector<TVector>> dataset)
    {
        return new([], []);
    }
}
