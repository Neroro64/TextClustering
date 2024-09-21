using TextClustering.Embedding.BoWVectorizer;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class TFIDFVectorizerTests
{
    [TestMethod]
    public void Transform_SingleString()
    {
        // Arrange
        var vectorizer = new TFIDFVectorizer(new() { Languages = [] });

        // Act
        var embeddings = vectorizer.FitThenTransform(["hello world"]).FirstOrDefault();

        // Assert
        Assert.IsNotNull(embeddings);
        Assert.AreEqual(0, embeddings[1], "TDIDF value for 'hello' does not match expected value of 1");
        Assert.AreEqual(0, embeddings[2], "TDIDF value for 'world' does not match expected value of 1");
    }

    [TestMethod]
    public void FitThenTransform_EmptyInput_ReturnsEmptyOutput()
    {
        // Arrange
        var vectorizer = new TFIDFVectorizer(new() { Languages = [] });
        var documents = new List<string>();

        // Act
        var result = vectorizer.FitThenTransform(documents).FirstOrDefault();

        // Assert
        Assert.AreEqual(0, vectorizer.Vocabulary.Count);
        Assert.IsNull(result, "Expected empty output for empty input");
    }

    [TestMethod]
    public void Transform_AfterFit_ReturnsConsistentVectors()
    {
        // Arrange
        var vectorizer = new TFIDFVectorizer(new() { Languages = [] });
        var trainingDocs = new List<string> { "apple banana", "banana cherry", "cherry date" };
        var testDocs = new List<string> { "apple banana", "cherry date" };

        // Act
        vectorizer.Fit(trainingDocs);
        var result = vectorizer.Transform(testDocs).ToList();

        // Assert
        Assert.AreEqual(2, result.Count, "Expected 2 vectors in the output");

        Assert.IsTrue(result[0].ContainsKey(1) && result[0].ContainsKey(2), "'First vector should contain indices for 'apple' and 'banana'");
        Assert.AreEqual(TFIDFVectorizer.ComputeTFIDF(1, trainingDocs.Count, 1), result[0][1], 1e-6, "TDIDF value for 'apple' does not match expected value.");
        Assert.AreEqual(TFIDFVectorizer.ComputeTFIDF(1, trainingDocs.Count, 2), result[0][2], 1e-6, "TDIDF value for 'banana' does not match expected value.");

        Assert.IsTrue(result[1].ContainsKey(3) && result[1].ContainsKey(4), "'First vector should contain indices for 'cherry' and 'date'");
        Assert.AreEqual(TFIDFVectorizer.ComputeTFIDF(1, trainingDocs.Count, 2), result[1][3], "TDIDF value for 'cherry' does not match expected value.");
        Assert.AreEqual(TFIDFVectorizer.ComputeTFIDF(1, trainingDocs.Count, 1), result[1][4], "TDIDF value for 'date' does not match expected value.");
    }

    [TestMethod]
    public void Fit_MultipleCalls_UpdatesVocabulary()
    {
        // Arrange
        var vectorizer = new TFIDFVectorizer(new() { Languages = [] });
        var docs1 = new List<string> { "apple banana", "date" };
        var docs2 = new List<string> { "banana cherry" };

        // Act
        vectorizer.Fit(docs1);
        var result1 = vectorizer.Transform(new List<string> { "apple banana" }).FirstOrDefault();

        vectorizer.Fit(docs2);
        var result2 = vectorizer.Transform(new List<string> { "banana cherry" }).FirstOrDefault();

        // Assert
        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);
        Assert.AreEqual(2, result1.Count);
        Assert.IsTrue(result2.ContainsKey(2) && result2.ContainsKey(4), "Both 'banana' and 'cherry' should be in vocabulary after second fit");
        Assert.IsTrue(result2[2] < result1[2], "TDIDF value for 'banana' decreased after seeing it in another document.");
    }
}
