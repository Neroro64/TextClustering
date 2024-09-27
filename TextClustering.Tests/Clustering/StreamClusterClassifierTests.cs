using Embedding.EmbeddingVector;
using Embedding.DistanceMetric;

using Clustering;

namespace TextClustering.Tests.Clustering;

[TestClass]
public class StreamingClusterClassifierTests
{
    [TestMethod]
    public void Classify_ShouldAssignVectorsToNewClusters_WhenNoExistingClusters()
    {
        var classifier = new StreamingClusterClassifier<DenseVector, EuclideanDistance>(0.99f, 1f);
        var dataset = new List<DenseVector>
        {
            new([1.0f, 2.0f]),
            new([3.0f, 4.0f])
        };

        var result = classifier.Classify(dataset);

        Assert.AreEqual(2, result.Labels.Count);
        Assert.AreEqual(2, result.OutlierScores.Count);
    }

    [TestMethod]
    public void Classify_ShouldMergeClusters_WhenCentroidsAreClose()
    {
        var classifier = new StreamingClusterClassifier<DenseVector<float>, EuclideanDistanceMetric>(0.9f, 0.1f);
        var dataset = new List<DenseVector<float>>
        {
            new DenseVector<float>(new float[] { 1.0f, 1.0f }),
            new DenseVector<float>(new float[] { 1.1f, 1.1f }),
            new DenseVector<float>(new float[] { 5.0f, 5.0f })
        };

        var result = classifier.Classify(dataset);

        Assert.AreEqual(2, result.Labels.Distinct().Count());
    }

    [TestMethod]
    public void Classify_ShouldHandleEmptyDataset()
    {
        var classifier = new StreamingClusterClassifier<DenseVector<float>, EuclideanDistanceMetric>(0.5f, 0.1f);
        var dataset = new List<DenseVector<float>>();

        var result = classifier.Classify(dataset);

        Assert.AreEqual(0, result.Labels.Count);
        Assert.AreEqual(0, result.OutlierScores.Count);
    }

    [TestMethod]
    public void Classify_ShouldAssignOutlierScoresCorrectly()
    {
        var classifier = new StreamingClusterClassifier<DenseVector<float>, EuclideanDistanceMetric>(0.5f, 0.1f);
        var dataset = new List<DenseVector<float>>
        {
            new DenseVector<float>(new float[] { 1.0f, 2.0f }),
            new DenseVector<float>(new float[] { 3.0f, 4.0f }),
            new DenseVector<float>(new float[] { 1.0f, 2.0f })
        };

        var result = classifier.Classify(dataset);

        Assert.AreEqual(1f, result.OutlierScores[0]);
        Assert.AreEqual(0.5f, result.OutlierScores[1]);
        Assert.AreEqual(1f, result.OutlierScores[2]);
    }
}
