using Embedding;
using Embedding.DistanceMetric;

namespace TextClustering.Tests.DistanceMetric;

[TestClass]
public class DistanceMetricBasicTests
{
    private const float Epsilon = 1e-3f;

    [DataTestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance, 0f)]
    [DataRow(DistanceMetricType.EuclideanDistance, 0f)]
    [DataRow(DistanceMetricType.CosineSimilarity, 0f)]
    public void CalculateDistance_DenseVectors_SameVector(DistanceMetricType distanceMetric, float expectedValue)
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3]);
        var vector2 = new DenseVector([1, 2, 3]);

        // Act
        float result = distanceMetric switch
        {
            DistanceMetricType.ManhattanDistance => ManhattanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.EuclideanDistance => EuclideanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.CosineSimilarity => CosineSimilarity.CalculateDistance(vector1, vector2),
            _ => throw new InvalidOperationException()
        };

        // Assert
        Assert.AreEqual(expectedValue, result, Epsilon);
    }

    [TestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance, 12f)]
    [DataRow(DistanceMetricType.EuclideanDistance, 7.4833f)]
    [DataRow(DistanceMetricType.CosineSimilarity, 2f)]
    public void CalculateDistance_DenseVectors_OppositeVectors(DistanceMetricType distanceMetric, float expectedValue)
    {
        // Arrange
        var vector1 = new DenseVector([1, 2, 3]);
        var vector2 = new DenseVector([-1, -2, -3]);

        // Act
        float result = distanceMetric switch
        {
            DistanceMetricType.ManhattanDistance => ManhattanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.EuclideanDistance => EuclideanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.CosineSimilarity => CosineSimilarity.CalculateDistance(vector1, vector2),
            _ => throw new InvalidOperationException()
        };

        // Assert
        Assert.AreEqual(expectedValue, result, Epsilon);
    }

    [TestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance, 2f)]
    [DataRow(DistanceMetricType.EuclideanDistance, 1.414f)]
    [DataRow(DistanceMetricType.CosineSimilarity, 1f)]
    public void CalculateDistance_DenseVectors_PerpendicularVectors(DistanceMetricType distanceMetric, float expectedValue)
    {
        // Arrange
        var vector1 = new DenseVector([1, 0, 0]);
        var vector2 = new DenseVector([0, 1, 0]);

        // Act
        float result = distanceMetric switch
        {
            DistanceMetricType.ManhattanDistance => ManhattanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.EuclideanDistance => EuclideanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.CosineSimilarity => CosineSimilarity.CalculateDistance(vector1, vector2),
            _ => throw new InvalidOperationException()
        };

        // Assert
        Assert.AreEqual(expectedValue, result, Epsilon);
    }

    [TestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance, 0f)]
    [DataRow(DistanceMetricType.EuclideanDistance, 0f)]
    [DataRow(DistanceMetricType.CosineSimilarity, 0f)]
    public void CalculateDistance_SparseVectors_SameVector(DistanceMetricType distanceMetric, float expectedValue)
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
        float result = distanceMetric switch
        {
            DistanceMetricType.ManhattanDistance => ManhattanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.EuclideanDistance => EuclideanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.CosineSimilarity => CosineSimilarity.CalculateDistance(vector1, vector2),
            _ => throw new InvalidOperationException()
        };

        // Assert
        Assert.AreEqual(expectedValue, result, Epsilon);
    }

    [TestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance, 18f)]
    [DataRow(DistanceMetricType.EuclideanDistance, 10.77f)]
    [DataRow(DistanceMetricType.CosineSimilarity, 2f)]
    public void CalculateDistance_SparseVectors_OppositeVectors(DistanceMetricType distanceMetric, float expectedValue)
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
        float result = distanceMetric switch
        {
            DistanceMetricType.ManhattanDistance => ManhattanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.EuclideanDistance => EuclideanDistance.CalculateDistance(vector1, vector2),
            DistanceMetricType.CosineSimilarity => CosineSimilarity.CalculateDistance(vector1, vector2),
            _ => throw new InvalidOperationException()
        };

        // Assert
        Assert.AreEqual(expectedValue, result, Epsilon);
    }

    [TestMethod]
    [DataRow(DistanceMetricType.ManhattanDistance)]
    [DataRow(DistanceMetricType.EuclideanDistance)]
    [DataRow(DistanceMetricType.CosineSimilarity)]
    public void CalculateDistance_DenseVectors_SimilarVectors_ReturnsHigherScore_ThanDifferentVectors(DistanceMetricType distanceMetric)
    {
        // Arrange
        // Similar to vector1 but not identical
        var vector1 = new DenseVector([0.8f, 0.6f, 0.4f]);
        var vector2 = new DenseVector([0.9f, 0.7f, 0.5f]);

        var vector3 = new DenseVector([1, 2, 3]);

        // Act
        float result1;
        float result2;
        switch (distanceMetric)
        {
            case DistanceMetricType.ManhattanDistance:
                result1 = ManhattanDistance.CalculateDistance(vector1, vector2);
                result2 = ManhattanDistance.CalculateDistance(vector1, vector3);
                break;
            case DistanceMetricType.EuclideanDistance:
                result1 = EuclideanDistance.CalculateDistance(vector1, vector2);
                result2 = EuclideanDistance.CalculateDistance(vector1, vector3);
                break;
            case DistanceMetricType.CosineSimilarity:
                result1 = CosineSimilarity.CalculateDistance(vector1, vector2);
                result2 = CosineSimilarity.CalculateDistance(vector1, vector3);
                break;
            default:
                throw new InvalidOperationException();
        }

        // Assert
        // Similar vectors should have lower distance than different vectors
        Assert.IsTrue(result1 < result2);
    }

    public enum DistanceMetricType
    {
        ManhattanDistance,
        EuclideanDistance,
        CosineSimilarity
    }
}
