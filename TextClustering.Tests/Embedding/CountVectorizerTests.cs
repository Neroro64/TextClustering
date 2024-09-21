using TextClustering.Embedding;

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
        Assert.AreEqual(new TermStats { Id = 1, NumberOfDocumentsWhereTheTermAppears = 1 }, vectorizer.Vocabulary["hello"], "'hello' stats do not match expected values.");
        Assert.AreEqual(new TermStats { Id = 2, NumberOfDocumentsWhereTheTermAppears = 1 }, vectorizer.Vocabulary["world"], "'world' stats do not match expected values.");

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
        Assert.AreEqual(1, embeddings[1], "Count for 'hello' does not match expected value of 1");
        Assert.AreEqual(1, embeddings[2], "Count for 'world' does not match expected value of 1");
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
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var trainingDocs = new List<string> { "apple banana", "banana cherry", "cherry date" };
        var testDocs = new List<string> { "apple banana", "cherry date" };

        // Act
        vectorizer.Fit(trainingDocs);
        var result = vectorizer.Transform(testDocs).ToList();

        // Assert
        Assert.AreEqual(2, result.Count, "Expected 2 vectors in the output");

        Assert.IsTrue(result[0].ContainsKey(1) && result[0].ContainsKey(2), "'First vector should contain indices for 'apple' and 'banana'");
        Assert.AreEqual(1, result[0][1], "'Count for 'apple' does not match expected value of 1.");
        Assert.AreEqual(1, result[0][2], "'Count for 'banana' does not match expected value of 1.");

        Assert.IsTrue(result[1].ContainsKey(3) && result[1].ContainsKey(4), "'First vector should contain indices for 'cherry' and 'date'");
        Assert.AreEqual(1, result[1][3], "'Count for 'cherry' does not match expected value of 1.");
        Assert.AreEqual(1, result[1][4], "'Count for 'date' does not match expected value of 1.");
    }

    [TestMethod]
    public void Transform_BeforeFit_ReturnsEmptyVector()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var documents = new List<string> { "test document" };

        // Act
        var result = vectorizer.Transform(documents).FirstOrDefault();

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual(0, result.Count, "'result' should be an empty vector when Transform is called before Fit.");
    }

    [TestMethod]
    public void FitThenTransform_HandlesUnseenWords()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var documents = new List<string> { "apple banana", "cherry date" };

        // Act
        vectorizer.Fit(documents);
        var vector = vectorizer.Transform(["apple unseen"]).FirstOrDefault();

        // Assert
        Assert.AreEqual(4, vectorizer.Vocabulary.Count, "Expected vocabulary size to increase after handling unseen words");
        Assert.IsNotNull(vector);
        Assert.AreEqual(1, vector.Count, "'vector' should have one entry for 'apple'");
        Assert.IsTrue(vector.ContainsKey(1), "'vector' does not contain index for 'apple'");
        Assert.AreEqual(1, vector[1], "'Count for 'apple' does not match expected value of 1.");
    }

    [TestMethod]
    public void Fit_MultipleCalls_UpdatesVocabulary()
    {
        // Arrange
        var vectorizer = new CountVectorizer(new() { Languages = [] });
        var docs1 = new List<string> { "apple banana" };
        var docs2 = new List<string> { "cherry date" };

        // Act
        vectorizer.Fit(docs1);
        var result1 = vectorizer.Transform(new List<string> { "apple cherry" }).FirstOrDefault();

        vectorizer.Fit(docs2);
        var result2 = vectorizer.Transform(new List<string> { "apple cherry" }).FirstOrDefault();

        // Assert
        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);
        Assert.AreEqual(1, result1.Count);
        Assert.IsTrue(result2.ContainsKey(1) && result2.ContainsKey(3), "Both 'apple' and 'cherry' should be in vocabulary after second fit");
    }
}
