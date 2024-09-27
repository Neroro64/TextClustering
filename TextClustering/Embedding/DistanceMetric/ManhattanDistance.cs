using System.Numerics.Tensors;

namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate Manhattan distance between vectors.
/// </summary>
public abstract class ManhattanDistance : IDistanceMetric<DenseVector>, IDistanceMetric<SparseVector>
{
    /// <summary>
    /// Calculates the Manhattan distance between two dense vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Manhattan distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        float[] intermediate = new float[vector1.Data.Length];
        TensorPrimitives.Subtract(vector1.Data, vector2.Data, intermediate);
        TensorPrimitives.Abs(intermediate, intermediate);
        return TensorPrimitives.Sum(intermediate);
    }

    /// <summary>
    /// Calculates the Manhattan distance between two sparse vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Manhattan distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        var (v1, v2) = SparseVector.ToDenseVectors(vector1, vector2);
        return CalculateDistance(v1, v2);
    }
}
