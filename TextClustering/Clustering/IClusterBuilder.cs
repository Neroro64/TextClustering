using Embedding;

namespace Clustering;

public interface IClusterBuilder<TVector>
{
    ClusteringResult Build(IEnumerable<IEmbeddingVector<TVector>> dataset);
}

