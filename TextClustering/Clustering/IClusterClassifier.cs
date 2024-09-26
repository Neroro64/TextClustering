using Embedding;

namespace Clustering;

public interface IClusterClassifier<in TVector> where TVector : IEmbeddingVector<TVector>
{
    ClusteringResult Classify(IEnumerable<TVector> dataset);
}
