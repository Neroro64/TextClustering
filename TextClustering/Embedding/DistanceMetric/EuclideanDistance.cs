using System.Numerics.Tensors;

namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate Euclidean distance between vectors.
/// </summary>
public abstract class EuclideanDistance : IDistanceMetric<DenseVector>, IDistanceMetric<SparseVector>
{
    /// <summary>
    /// Calculates the Euclidean distance between two dense vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Euclidean distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        return TensorPrimitives.Distance(vector1.Data, vector2.Data);
    }

    /// <summary>
    /// Calculates the Euclidean distance between two sparse vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Euclidean distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        var (v1, v2) = SparseVector.ToDenseVectors(vector1, vector2);
        return CalculateDistance(v1, v2);
    }
}
