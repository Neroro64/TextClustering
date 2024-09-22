using TextClustering.Embedding.Transformer;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class BertTransformerTests
{
    [TestMethod]
    public void Constructor_LoadBertTokenizer()
    {
        // Act and Assert
        var transformer = new BertTransformer();
        Assert.IsNotNull(transformer.Tokenizer);
    }

    [TestMethod]
    public void Transform_SingleString()
    {
        // Arrange
        var transformer = new BertTransformer();

        // Act
        var embedding = transformer.Transform(["hello world"]).FirstOrDefault();

        // Assert
        Assert.IsNotNull(embedding);
        Assert.AreEqual(384, embedding.Length);
    }

    [TestMethod]
    public void Transform_MultipleStrings()
    {
        // Arrange
        var transformer = new BertTransformer();

        // Act
        var embeddings = transformer.Transform(_testDocuments);

        // Assert
        Assert.AreEqual(_testDocuments.Length, embeddings.Count);
        var similarityBetweenDoc1And2 = ComputeCosineSimilary(embeddings[0], embeddings[1]);
        var similarityBetweenDoc1And3 = ComputeCosineSimilary(embeddings[0], embeddings[2]);
        Assert.IsTrue(similarityBetweenDoc1And2 < similarityBetweenDoc1And3);
    }

    [TestMethod]
    public void Transform_MultipleStrings_WithCountLargerThanOneBatch()
    {
        // Arrange
        var transformer = new BertTransformer();
        var testDocuments = new List<string>();
        for (int i = 0; i < 65; ++i)
        {
            testDocuments.Add(_testDocuments[i % 3]);
        }

        // Act
        var embeddings = transformer.Transform(testDocuments);

        // Assert
        Assert.AreEqual(65, embeddings.Count);
        Assert.AreEqual(embeddings[0].Sum(), embeddings[3].Sum(), 1e-6);
        Assert.AreEqual(embeddings[1].Sum(), embeddings[4].Sum(), 1e-6);
        Assert.AreEqual(embeddings[2].Sum(), embeddings[5].Sum(), 1e-6);
    }

    private static readonly string[] _testDocuments = new[]
    {
        "This is a sample document.",
        "The quick brown fox jumps over the lazy dog.",
        "Another sample document for testing purposes.",
    };

    private static float ComputeCosineSimilary(float[] v1, float[] v2)
    {
        float dotProduct = v1.Zip(v2, (x, y) => x * y).Sum();
        float magnitude1 = (float)Math.Sqrt(v1.Sum(x => x * x));
        float magnitude2 = (float)Math.Sqrt(v2.Sum(x => x * x));
        return dotProduct / (magnitude1 * magnitude2);
    }
}
