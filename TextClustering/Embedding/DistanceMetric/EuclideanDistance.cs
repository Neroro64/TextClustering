namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate Euclidean distance between vectors.
/// </summary>
public static class EuclideanDistance
{
    /// <summary>
    /// Calculates the Euclidean distance between two dense vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Euclidean distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        double distance = 0;
        for (int i = 0; i < vector1.Length; i++)
        {
            distance += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
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
            .Sum(key => (vector1[key] - vector2[key]) * (vector1[key] - vector2[key]));

        return (float)Math.Sqrt(distance);
    }
}
