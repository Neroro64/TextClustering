using Embedding;
using Embedding.DistanceMetric;

namespace TextClustering.Tests.DistanceMetric;

[TestClass]
public class CosineSimilarityTests
{
    [TestMethod]
    public void CalculateDistance_DenseVectors_SameVector_ReturnsOne()
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3]);
        var vector2 = new DenseVector([1, 2, 3]);

        // Act
        float result = CosineSimilarity.CalculateDistance(vector1, vector2);

        // Assert
        Assert.AreEqual(1f, result);
    }

    [TestMethod]
    public void CalculateDistance_DenseVectors_OppositeVectors_ReturnsMinusOne()
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3]);
        var vector2 = new DenseVector([-1, -2, -3]);

        // Act
        float result = CosineSimilarity.CalculateDistance(vector1, vector2);

        // Assert
        Assert.AreEqual(-1f, result);
    }

    [TestMethod]
    public void CalculateDistance_DenseVectors_PerpendicularVectors_ReturnsZero()
    {
        // Arrange
        var vector1 = new DenseVector([1, 0, 0]);
        var vector2 = new DenseVector([0, 1, 0]);

        // Act
        float result = CosineSimilarity.CalculateDistance(vector1, vector2);

        // Assert
        Assert.AreEqual(0f, result);
    }

    [TestMethod]
    public void CalculateDistance_SparseVectors_SameVector_ReturnsOne()
    {
        // Arrange
        var vector1 = new SparseVector(new Dictionary<int, float>
        {
            { 1, 2 },
            { 2, 3 },
            { 3, 4 }
        });
        var vector2 = new SparseVector(new Dictionary<int, float>
        {
            { 1, 2 },
            { 2, 3 },
            { 3, 4 }
        });

        // Act
        float result = CosineSimilarity.CalculateDistance(vector1, vector2);

        // Assert
        Assert.AreEqual(1f, result);
    }

    [TestMethod]
    public void CalculateDistance_SparseVectors_OppositeVectors_ReturnsMinusOne()
    {
        // Arrange
        var vector1 = new SparseVector(new Dictionary<int, float>
        {
            { 1, 2 },
            { 2, 3 },
            { 3, 4 }
        });
        var vector2 = new SparseVector(new Dictionary<int, float>
        {
            { 1, -2 },
            { 2, -3 },
            { 3, -4 }
        });

        // Act
        float result = CosineSimilarity.CalculateDistance(vector1, vector2);

        // Assert
        Assert.AreEqual(-1f, result);
    }

    [TestMethod]
    public void CalculateDistance_DenseVectors_SimilarVectors_ReturnsHigherScoreThanDifferentVectors()
    {
        // Arrange
        // Similar to vector1 but not identical
        var vector1 = new DenseVector([0.8f, 0.6f, 0.4f]);
        var vector2 = new DenseVector([0.9f, 0.7f, 0.5f]);

        var vector3 = new DenseVector([1, 2, 3]);

        // Act
        float result1 = CosineSimilarity.CalculateDistance(vector1, vector2);
        float result2 = CosineSimilarity.CalculateDistance(vector1, vector3);

        // Assert
        // Similar vectors should have higher score than different vectors
        Assert.IsTrue(result1 > result2);
    }

}
