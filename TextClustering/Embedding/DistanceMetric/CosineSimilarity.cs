using System.Numerics;

namespace Embedding.DistanceMetric;

/// <summary>
/// Provides static methods to calculate cosine similarity between vectors, as distance.
/// </summary>
public abstract class CosineSimilarity : IDistanceMetric<DenseVector>, IDistanceMetric<SparseVector>
{
    /// <summary>
    /// A small value used to prevent division by zero in calculations.
    /// </summary>
    private const float Epsilon = 1e-10f;

    /// <summary>
    /// Calculates the cosine similarity between two dense vectors and return it as a distance: (1 - cosine_similarity).
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity distance as a float value.</returns>
    public static float CalculateDistance(DenseVector vector1, DenseVector vector2)
    {
        // Calculate dot product of the two vectors
        var v1 = vector1.Data.ToSIMDVectors();
        var v2 = vector2.Data.ToSIMDVectors();

        float dotProduct = 0;
        float v1InnerProduct = 0;
        float v2InnerProduct = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            dotProduct += Vector.Dot(v1[i], v2[i]);
            v1InnerProduct += Vector.Dot(v1[i], v1[i]);
            v2InnerProduct += Vector.Dot(v2[i], v2[i]);
        }

        // Calculate magnitudes
        double magnitude1 = Math.Sqrt(v1InnerProduct);
        double magnitude2 = Math.Sqrt(v2InnerProduct);

        // Calculate cosine similarity
        return 1f - (float)(dotProduct / Math.Max(magnitude1 * magnitude2, Epsilon));
    }

    /// <summary>
    /// Calculates the cosine similarity between two sparse vectors and return it as a distance: (1 - cosine_similarity).
    /// </summary>
    /// <param name="vector1">The first vector.</param>
    /// <param name="vector2">The second vector.</param>
    /// <returns>The cosine similarity distance as a float value.</returns>
    public static float CalculateDistance(SparseVector vector1, SparseVector vector2)
    {
        // Calculate dot product
        double dotProduct = vector1.Data.Keys.Intersect(vector2.Data.Keys).Sum(key => vector1[key] * vector2[key]);

        // Calculate magnitudes
        double magnitude1 = Math.Sqrt(vector1.Data.Values.Sum(v => v * v));
        double magnitude2 = Math.Sqrt(vector2.Data.Values.Sum(v => v * v));

        // Calculate cosine similarity
        return 1f - (float)(dotProduct / Math.Max(magnitude1 * magnitude2, Epsilon));
    }
}
