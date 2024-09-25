namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate cosine similarity between vectors.
/// </summary>
public static class CosineSimilarity
{
    /// <summary>
    /// A small value used to prevent division by zero in calculations.
    /// </summary>
    private const float Epsilon = 1e-10f;

    /// <summary>
    /// Calculates the cosine similarity between two dense vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        // Calculate dot product of the two vectors
        double dotProduct = 0;
        for (int i = 0; i < vector1.Length; i++)
        {
            dotProduct += vector1[i] * vector2[i];
        }

        // Calculate magnitudes
        double magnitude1 = Math.Sqrt(vector1.Data.Sum(static v => v * v));
        double magnitude2 = Math.Sqrt(vector2.Data.Sum(static v => v * v));

        // Calculate cosine similarity
        return (float)(dotProduct / Math.Max(magnitude1 * magnitude2, Epsilon));
    }

    /// <summary>
    /// Calculates the cosine similarity between two sparse vectors.
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        // Calculate dot product
        double dotProduct = vector1.Data.Keys.Intersect(vector2.Data.Keys).Sum(key => vector1[key] * vector2[key]);

        // Calculate magnitudes
        double magnitude1 = Math.Sqrt(vector1.Data.Values.Sum(v => v * v));
        double magnitude2 = Math.Sqrt(vector2.Data.Values.Sum(v => v * v));

        // Calculate cosine similarity
        return (float)(dotProduct / Math.Max(magnitude1 * magnitude2, Epsilon));
    }
}
