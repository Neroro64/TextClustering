using System.Numerics;

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
        var v1 = vector1.Data.ToSIMDVectors();
        var v2 = vector2.Data.ToSIMDVectors();

        double distance = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            distance += Vector.Dot(v1[i] - v2[i], v1[i] - v2[i]);
        }

        return (float)Math.Sqrt(distance);
    }

    /// <summary>
    /// Calculates the Euclidean distance between two sparse vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Euclidean distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        double distance = vector1.Data.Keys
            .Intersect(vector2.Data.Keys)
            .AsParallel()
            .Sum(key => (vector1[key] - vector2[key]) * (vector1[key] - vector2[key]));

        return (float)Math.Sqrt(distance);
    }
}
