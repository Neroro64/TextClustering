using System.Numerics.Tensors;

namespace Embedding.EmbeddingVector;

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

    /// <inheritdoc />
    public DenseVector GetCentroidVector(DenseVector other, float weight = 1f)
    {
        float[] centroid = new float[Length];
        TensorPrimitives.Subtract(other.Data, Data, centroid);
        TensorPrimitives.MultiplyAdd(centroid, weight * 0.5f, Data, centroid);
        return new(centroid);
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
