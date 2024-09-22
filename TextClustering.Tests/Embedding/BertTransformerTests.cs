using TextClustering.Embedding.Transformer;

namespace TextClustering.Tests.Embedding;

[TestClass]
public class TransformerTests
{
    [TestMethod]
    public void Ctor_LoadBertTokenizer()
    {
        var transformer = new BertTransformer();
        Assert.IsNotNull(transformer);
    }
}
