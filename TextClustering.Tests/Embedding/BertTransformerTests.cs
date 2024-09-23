using System.Text;

using Embedding.Transformer;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class BertTransformerTests
{
    [TestMethod]
    public void Transform_SingleString()
    {
        // Arrange
        using var transformer = new BertTransformer();

        // Act
        float[]? embedding = transformer.Transform(["hello world"]).FirstOrDefault();

        // Assert
        Assert.IsNotNull(embedding);
        Assert.AreEqual(384, embedding.Length);
    }

    [TestMethod]
    public void Transform_StringsLongerThanInputDimension()
    {
        // Arrange
        using var transformer = new BertTransformer();
        var defaultSettings = new BertTransformerSettings();
        int strLength = defaultSettings.InputDimension * 3;
        var random = new Random(Seed: 42);

        var builder = new StringBuilder();
        for (int i = 0; i < strLength; i++)
        {
            char c = (char)random.Next(32, 127); // ASCII printable characters
            _ = builder.Append(c);
        }
        string randomString = builder.ToString();
        string[] inputStrings = [
            randomString[..(defaultSettings.InputDimension+1)],
            randomString
        ];

        // Act
        var embeddings = transformer.Transform(inputStrings);

        // Assert
        Assert.AreEqual(2, embeddings.Count);
        Assert.AreEqual(384, embeddings[0].Length);
        Assert.AreEqual(384, embeddings[1].Length);
    }

    [TestMethod]
    public void Transform_MultipleStrings()
    {
        // Arrange
        using var transformer = new BertTransformer();

        // Act
        var embeddings = transformer.Transform(TestDocuments);

        // Assert
        Assert.AreEqual(TestDocuments.Length, embeddings.Count);
        float similarityBetweenDoc1And2 = ComputeCosineSimilary(embeddings[0], embeddings[1]);
        float similarityBetweenDoc1And3 = ComputeCosineSimilary(embeddings[0], embeddings[2]);
        Assert.IsTrue(similarityBetweenDoc1And2 < similarityBetweenDoc1And3);
    }

    [TestMethod]
    public void Transform_MultipleStrings_WithCountLargerThanOneBatch()
    {
        // Arrange
        int documentCount = (new BertTransformerSettings().BatchSize * 2) + 1;
        using var transformer = new BertTransformer();
        var testDocuments = new List<string>();
        for (int i = 0; i < documentCount; ++i)
        {
            testDocuments.Add(TestDocuments[i % 3]);
        }

        // Act
        var embeddings = transformer.Transform(testDocuments);

        // Assert
        Assert.AreEqual(documentCount, embeddings.Count);
        Assert.AreEqual(embeddings[0].Sum(), embeddings[3].Sum(), 1e-6);
        Assert.AreEqual(embeddings[1].Sum(), embeddings[4].Sum(), 1e-6);
        Assert.AreEqual(embeddings[2].Sum(), embeddings[5].Sum(), 1e-6);
    }

    private static readonly string[] TestDocuments = [
        "This is a sample document.",
        "The quick brown fox jumps over the lazy dog.",
        "Another sample document for testing purposes.",
    ];

    private static float ComputeCosineSimilary(float[] v1, float[] v2)
    {
        float dotProduct = v1.Zip(v2, static (x, y) => x * y).Sum();
        float magnitude1 = (float)Math.Sqrt(v1.Sum(static x => x * x));
        float magnitude2 = (float)Math.Sqrt(v2.Sum(static x => x * x));
        return dotProduct / (magnitude1 * magnitude2);
    }
}
