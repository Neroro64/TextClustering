using System.Numerics.Tensors;

using Embedding.EmbeddingVector;

namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate cosine similarity between vectors, as distance.
/// </summary>
public abstract class CosineSimilarity : IDistanceMetric<DenseVector>, IDistanceMetric<SparseVector>
{
    /// <summary>
    /// Calculates the cosine similarity between two dense vectors and return it as a distance: (1 - cosine_similarity).
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        return 1f - TensorPrimitives.CosineSimilarity(vector1.Data, vector2.Data);
    }

    /// <summary>
    /// Calculates the cosine similarity between two sparse vectors and return it as a distance: (1 - cosine_similarity).
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        var (v1, v2) = SparseVector.ToDenseVectors(vector1, vector2);
        return CalculateDistance(v1, v2);
    }

    /// <summary>
    /// Calculates the cosine similarity distance between two vectors on the unit sphere.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity distance between the two vectors.</returns>
    public static float CalculateUnitSphereDistance(DenseVector vector1, DenseVector vector2) => CalculateDistance(vector1, vector2);

    /// <summary>
    /// Calculates the cosine similarity distance between two sparse vectors on the unit sphere.
    /// </summary>
    /// <param name="vector1">The first sparse vector.</param>
    /// <param name="vector2">The second sparse vector.</param>
    /// <returns>The cosine similarity distance between the two vectors.</returns>
    public static float CalculateUnitSphereDistance(SparseVector vector1, SparseVector vector2) => CalculateDistance(vector1, vector2);
}
