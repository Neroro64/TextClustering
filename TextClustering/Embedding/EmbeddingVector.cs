using System.Numerics.Tensors;

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
    float DistanceTo(TVector other, Func<TVector, TVector, float> calculateDistance);

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

    /// <summary>
    /// Gets the value at the specified index in the vector.
    /// </summary>
    /// <param name="index">The index of the value to retrieve.</param>
    /// <returns>The value at the specified index.</returns>
    public float this[int index] => Data[index];

    private int _hashCode = -1;

    /// <inheritdoc />
    public float DistanceTo(DenseVector other, Func<DenseVector, DenseVector, float> calculateDistance)
    {
        return calculateDistance(this, other);
    }

    /// <summary>
    /// Gets the hash code for this vector.
    /// </summary>
    /// <returns>The hash code.</returns>
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

    /// <summary>
    /// Determines whether this vector is equal to another object.
    /// </summary>
    /// <param name="other">The other object to compare with.</param>
    /// <returns>True if the vectors are equal; otherwise, false.</returns>
    bool IEquatable<IEmbeddingVector<DenseVector>>.Equals(IEmbeddingVector<DenseVector>? other)
    {
        return GetHashCode() == other?.GetHashCode();
    }

    /// <summary>
    /// Returns the dense vector as a unit vector.
    /// </summary>
    /// <returns>A new <see cref="DenseVector"/> in the same direction but with magnitude 1.</returns>
    public DenseVector ToUnitVector()
    {
        float v1Norm = TensorPrimitives.Norm(Data);
        float[] unitVector = new float[Length];
        TensorPrimitives.Divide(Data, v1Norm, unitVector);
        return new(unitVector);
    }
}

/// <summary>
/// Represents a sparse embedding vector with a specified data type and dictionary of non-zero values.
/// </summary>
public record SparseVector(Dictionary<int, float> Data) : IEmbeddingVector<SparseVector>
{
    /// <inheritdoc />
    public int Length => Data.Count;

    /// <summary>
    /// Gets the value at the specified index in the vector.
    /// </summary>
    /// <param name="index">The index of the value to retrieve.</param>
    /// <returns>The value at the specified index.</returns>
    public float this[int index] => Data[index];

    /// <summary>
    /// Determines whether the vector contains a non-zero value at the specified key.
    /// </summary>
    /// <param name="key">The key to check.</param>
    /// <returns>True if the vector contains a non-zero value at the specified key; otherwise, false.</returns>
    public bool ContainsKey(int key) => Data.ContainsKey(key);

    private int _hashCode = -1;

    /// <inheritdoc />
    public float DistanceTo(SparseVector other, Func<SparseVector, SparseVector, float> calculateDistance)
    {
        return calculateDistance(this, other);
    }

    /// <summary>
    /// Gets the hash code for this vector.
    /// </summary>
    /// <returns>The hash code.</returns>
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

    /// <summary>
    /// Determines whether this vector is equal to another object.
    /// </summary>
    /// <param name="other">The other object to compare with.</param>
    /// <returns>True if the vectors are equal; otherwise, false.</returns>
    bool IEquatable<IEmbeddingVector<SparseVector>>.Equals(IEmbeddingVector<SparseVector>? other)
    {
        return GetHashCode() == other?.GetHashCode();
    }

    /// <summary>
    /// Converts this sparse vector to a dense vector.
    /// </summary>
    /// <param name="keys">The keys to include in the dense vector. If null, all keys from the sparse vector are used.</param>
    /// <returns>A new dense vector containing the values from this sparse vector.</returns>
    private DenseVector ToDenseVector(int[]? keys = null)
    {
        keys ??= [.. Data.Keys];
        float[] denseVector = new float[keys.Length];
        for (int i = 0; i < keys.Length; i++)
        {
            if (Data.TryGetValue(keys[i], out float value))
            {
                denseVector[i] = value;
            }
        }
        return new DenseVector(denseVector);
    }

    /// <summary>
    /// Converts a pair of sparse vectors to a pair of dense vectors with the same keys.
    /// </summary>
    /// <param name="vector1">The first sparse vector.</param>
    /// <param name="vector2">The second sparse vector.</param>
    /// <returns>A tuple containing the converted dense vectors.</returns>
    public static (DenseVector, DenseVector) ToDenseVectors(SparseVector vector1, SparseVector vector2)
    {
        // Get the union of keys from both vectors
        int[] unionKeys = vector1.Data.Keys.Union(vector2.Data.Keys).ToArray();

        // Convert the vectors to dense vectors
        var v1 = vector1.ToDenseVector(unionKeys);
        var v2 = vector2.ToDenseVector(unionKeys);

        return (v1, v2);
    }
}
