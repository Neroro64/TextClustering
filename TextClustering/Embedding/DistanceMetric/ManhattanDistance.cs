namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate Manhattan distance between vectors.
/// </summary>
public static class ManhattanDistance
{
    /// <summary>
    /// Calculates the Manhattan distance between two dense vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Manhattan distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        double distance = 0;
        for (int i = 0; i < vector1.Length; i++)
        {
            distance += Math.Abs(vector1[i] - vector2[i]);
        }

        return (float)distance;
    }

    /// <summary>
    /// Calculates the Manhattan distance between two sparse vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The Manhattan distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
        => vector1.Data.Keys.Intersect(vector2.Data.Keys).Sum(key => Math.Abs(vector1[key] - vector2[key]));
}
