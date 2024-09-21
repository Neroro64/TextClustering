using TextClustering.Embedding;
using TextClustering.Embedding.CountVectorizer;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class CountVectorizerTests
{
    [TestMethod]
    public void Fit_SingleString_CorrectlyBuildVocabulary()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });

        // Act
        vectorizer.Fit(["hello world"]);

        // Assert
        Assert.AreEqual(2, vectorizer.Vocabulary.Count, "Expected vocabulary size of 2");
        Assert.AreEqual(new TermStats(1, 1), vectorizer.Vocabulary["hello"], "Term 'hello' stats do not match expected values");
        Assert.AreEqual(new TermStats(2, 1), vectorizer.Vocabulary["world"], "Term 'world' stats do not match expected values");
    }

    [TestMethod]
    public void Transform_SingleString()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });

        // Act
        var embeddings = vectorizer.FitThenTransform(["hello world"]).FirstOrDefault();

        // Assert
        Assert.IsNotNull(embeddings);
        Assert.AreEqual(1, embeddings[1], "Expected count for 'hello' to be 1");
        Assert.AreEqual(1, embeddings[2], "Expected count of 'world' to be 1");
    }

    [TestMethod]
    public void FitThenTransform_EmptyInput_ReturnsEmptyOutput()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });
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
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var trainingDocs = new List<string> { "apple banana", "banana cherry", "cherry date" };
        var testDocs = new List<string> { "apple banana", "cherry date" };

        vectorizer.Fit(trainingDocs);
        var result = vectorizer.Transform(testDocs).ToList();

        Assert.AreEqual(2, result.Count, "Expected 2 vectors in the output");

        Assert.IsTrue(result[0].ContainsKey(1) && result[0].ContainsKey(2), "First vector should contain indices for 'apple' and 'banana'");
        Assert.AreEqual(1, result[0][1]);
        Assert.AreEqual(1, result[0][2]);

        Assert.IsTrue(result[1].ContainsKey(3) && result[1].ContainsKey(4), "Second vector should contain indices for 'cherry' and 'date'");
        Assert.AreEqual(1, result[1][1]);
        Assert.AreEqual(1, result[1][2]);
    }

    [TestMethod]
    public void Transform_BeforeFit_ReturnsEmptyVector()
    {
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var documents = new List<string> { "test document" };

        var result = vectorizer.Transform(documents).FirstOrDefault();

        Assert.IsNotNull(result);
        Assert.AreEqual(0, result.Count);
    }

    [TestMethod]
    public void FitThenTransform_HandlesUnseenWords()
    {
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var documents = new List<string> { "apple banana", "cherry date" };

        // Act
        vectorizer.Fit(documents);
        var vector = vectorizer.Transform(["apple unseen"]).FirstOrDefault();

        Assert.AreEqual(4, vectorizer.Vocabulary.Count);
        Assert.IsNotNull(vector);
        Assert.AreEqual(1, vector.Count);
        Assert.IsTrue(vector.ContainsKey(1));
        Assert.AreEqual(1, vector[1]);
    }

    [TestMethod]
    public void Fit_MultipleCalls_UpdatesVocabulary()
    {
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var docs1 = new List<string> { "apple banana" };
        var docs2 = new List<string> { "cherry date" };

        vectorizer.Fit(docs1);
        var result1 = vectorizer.Transform(new List<string> { "apple cherry" }).FirstOrDefault();

        vectorizer.Fit(docs2);
        var result2 = vectorizer.Transform(new List<string> { "apple cherry" }).FirstOrDefault();

        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);
        Assert.IsTrue(result2.ContainsKey(1) && result2.ContainsKey(3), "Both 'apple' and 'cherry' should be in vocabulary after second fit");
    }
}
