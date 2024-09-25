namespace Embedding;

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
    double DistanceTo(TVector other, Func<TVector, TVector, float> calculateDistance);

    /// <summary>
    /// Gets the length of this vector.
    /// </summary>
    int Length { get; }
}

/// <summary>
/// Represents a dense embedding vector with a specified data type and array of values.
/// </summary>
public record DenseVector(float[] Data) : IEmbeddingVector<DenseVector>
{
    /// <inheritdoc />
    public int Length => Data.Length;
    public float this[int index] => Data[index];

    private int _hashCode = -1;

    /// <inheritdoc />
    public double DistanceTo(DenseVector other, Func<DenseVector, DenseVector, float> calculateDistance)
    {
        return calculateDistance(this, other);
    }

    public override int GetHashCode()
    {
        unchecked
        {
            if (_hashCode != -1)
            {
                return _hashCode;
            }

            const int prime = 31;
            int hash = 17;
            foreach (float value in Data)
            {
                hash = (hash * prime) + BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
            }
            _hashCode = hash;
            return hash;
        }
    }

    bool IEquatable<IEmbeddingVector<DenseVector>>.Equals(IEmbeddingVector<DenseVector>? other)
    {
        return GetHashCode() == other?.GetHashCode();
    }
}

/// <summary>
/// Represents a sparse embedding vector with a specified data type and dictionary of non-zero values.
/// </summary>
public record SparseVector(Dictionary<int, float> Data) : IEmbeddingVector<SparseVector>
{
    /// <inheritdoc />
    public int Length => Data.Count;
    public float this[int index] => Data[index];
    public bool ContainsKey(int key) => Data.ContainsKey(key);

    private int _hashCode = -1;

    /// <inheritdoc />
    public double DistanceTo(SparseVector other, Func<SparseVector, SparseVector, float> calculateDistance)
    {
        return calculateDistance(this, other);
    }

    public override int GetHashCode()
    {
        unchecked
        {
            if (_hashCode != -1)
            {
                return _hashCode;
            }

            const int prime = 31;
            int hash = 17;
            foreach (var kvp in Data)
            {
                hash = (hash * prime) + kvp.Key.GetHashCode();
                hash = (hash * prime) + kvp.Value.GetHashCode();
            }
            _hashCode = hash;
            return hash;
        }
    }

    bool IEquatable<IEmbeddingVector<SparseVector>>.Equals(IEmbeddingVector<SparseVector>? other)
    {
        return GetHashCode() == other?.GetHashCode();
    }
}
