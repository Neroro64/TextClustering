namespace Embedding.EmbeddingVector;

/// <summary>
/// Represents an interface for embedding vectors, providing methods to calculate distances between vectors.
/// </summary>
/// <typeparam name="TVector">The type of the vector.</typeparam>
public interface IEmbeddingVector<TVector> : IEquatable<IEmbeddingVector<TVector>>
{
    /// <summary>
    /// Calculates the distance between this vector and another vector using a specified distance metric.
    /// </summary>
    /// <param name="other">The other vector to calculate the distance with.</param>
    /// <param name="calculateDistance">The function to calculate the distance between two vectors.</param>
    /// <returns>The calculated distance between the two vectors.</returns>
    float DistanceTo(TVector other, Func<TVector, TVector, float> calculateDistance);

    /// <summary>
    /// Gets the length of this vector.
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Calculates the centroid vector between this vector and another vector adjusted by the weight.
    /// This method effectively computes 'a + weight * (b + a) / 2'
    /// </summary>
    /// <param name="other">The other vector to calculate the centroid with.</param>
    /// <param name="weight">The weight to apply to the other vector.</param>
    /// <returns>The centroid vector between the two vectors.</returns>
    TVector GetCentroidVector(TVector other, float weight = 1f);
}
