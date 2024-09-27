using Embedding.EmbeddingVector;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class EmbeddingVectorTests
{
    [TestMethod]
    public void DenseVector_GetHashCode_ReturnSameHashForEqualVectors()
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3, 4, 5]);
        var vector2 = new DenseVector([1, 2, 3, 4, 5]);

        // Act
        int hash1 = vector1.GetHashCode();
        int hash2 = vector2.GetHashCode();

        // Assert
        Assert.AreEqual(hash1, hash2, "Expected hash codes to be equal for equal vectors");
    }

    [TestMethod]
    public void SparseVector_GetHashCode_ReturnSameHashForEqualVectors()
    {
        // Arrange
        var vector1 = new SparseVector(new() { [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5 });
        var vector2 = new SparseVector(new() { [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5 });

        // Act
        int hash1 = vector1.GetHashCode();
        int hash2 = vector2.GetHashCode();

        // Assert
        Assert.AreEqual(hash1, hash2, "Expected hash codes to be equal for equal vectors");
    }

    [TestMethod]
    public void DenseVector_GetHashCode_ReturnDifferentHashForDifferentVectors()
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3, 4, 5]);
        var vector2 = new DenseVector([5, 4, 3, 2, 1]);

        // Act
        int hash1 = vector1.GetHashCode();
        int hash2 = vector2.GetHashCode();

        // Assert
        Assert.AreNotEqual(hash1, hash2, "Expected hash codes to be different for different vectors");
    }

    [TestMethod]
    public void SparseVector_GetHashCode_ReturnDifferentHashForDifferentVectors()
    {
        // Arrange
        var vector1 = new SparseVector(new() { [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5 });
        var vector2 = new SparseVector(new() { [1] = 5, [2] = 4, [3] = 3, [4] = 2, [5] = 1 });
        var vector3 = new SparseVector(new() { [5] = 1, [4] = 2, [3] = 3, [2] = 4, [1] = 5 });

        // Act
        int hash1 = vector1.GetHashCode();
        int hash2 = vector2.GetHashCode();
        int hash3 = vector3.GetHashCode();

        // Assert
        Assert.AreNotEqual(hash1, hash2, "Expected hash codes to be different for different vectors");
        Assert.AreNotEqual(hash1, hash3, "Expected hash codes to be different for different vectors");
        Assert.AreNotEqual(hash2, hash3, "Expected hash codes to be different for different vectors");
    }

    [DataTestMethod]
    [DataRow(1f, new[] { 3f, 3, 3, 3, 3 })]
    [DataRow(0.1f, new[] { 1.2f, 2.1f, 3f, 3.9f, 4.8f })]
    [DataRow(1.1f, new[] { 3.2f, 3.1f, 3f, 2.9f, 2.8f })]
    public void DenseVector_GetCentroidVector_ReturnsCorrectCentroid(float weight, float[] expectedValues)
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3, 4, 5]);
        var vector2 = new DenseVector([5, 4, 3, 2, 1]);

        // Act
        var centroid = vector1.GetCentroidVector(vector2, weight);

        // Assert
        for (int i = 0; i < expectedValues.Length; ++i)
        {
            Assert.AreEqual(expectedValues[i], centroid[i], Tolerance, $"Expected centroid value at index {i} to be within tolerance");
        }
    }

    [DataTestMethod]
    [DataRow(1f, new[] { 3f, 3, 3, 3, 3 }, new[] { 0.5f, 1f, 1.5f, 2f, 2.5f, 0.5f, 1f, 1.5f, 2f, 2.5f })]
    [DataRow(0.1f, new[] { 1.2f, 2.1f, 3f, 3.9f, 4.8f }, new[] { 0.95f, 1.9f, 2.85f, 3.8f, 4.75f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f })]
    [DataRow(1.1f, new[] { 3.2f, 3.1f, 3f, 2.9f, 2.8f }, new[] { 0.45f, 0.9f, 1.35f, 1.8f, 2.25f, 0.55f, 1.1f, 1.65f, 2.2f, 2.75f })]
    public void SparseVector_GetCentroidVector_ReturnsCorrectCentroid(float weight, float[] expectedValues1, float[] expectedValues2)
    {
        // Arrange
        var vector1 = new SparseVector(new() { [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5 });
        var vector2 = new SparseVector(new() { [1] = 5, [2] = 4, [3] = 3, [4] = 2, [5] = 1 });
        var vector3 = new SparseVector(new() { [6] = 1, [7] = 2, [8] = 3, [9] = 4, [10] = 5 });

        // Act
        float[] centroid1 = [.. vector1.GetCentroidVector(vector2, weight).Data.Values];
        float[] centroid2 = [.. vector1.GetCentroidVector(vector3, weight).Data.Values];

        // Assert
        for (int i = 0; i < expectedValues1.Length; ++i)
        {
            Assert.AreEqual(expectedValues1[i], centroid1[i], Tolerance, $"Expected centroid value at index {i} to be within tolerance");
        }
        for (int i = 0; i < expectedValues2.Length; ++i)
        {
            Assert.AreEqual(expectedValues2[i], centroid2[i], Tolerance, $"Expected centroid value at index {i} to be within tolerance");
        }
    }

    private const float Tolerance = 1e-3f;
}
