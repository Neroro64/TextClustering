using Embedding.BowVectorizer;

namespace TextClustering.Benchmark;

internal sealed class SingleThreaddedCountVectorizer(BoWVectorizerConfig config) : CountVectorizer(config)
{
    protected override List<Dictionary<string, int>> ExtractTermFrequency(IEnumerable<string> documents)
    {
        return documents
            .Select(ExtractTermFrequency)
            .Where(static tf => tf.Keys.Count > 0)
            .ToList();
    }

    protected override List<SparseVector> ToSparseVector(IEnumerable<Dictionary<string, int>> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);
        return documentTermFrequency
            .Select(termFrequency =>
            {
                Dictionary<int, float> sparseVector = [];
                foreach (var tf in termFrequency)
                {
                    if (!Vocabulary.TryGetValue(tf.Key, out var value)
                        || value.NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                    {
                        continue;
                    }
                    sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
                }
                return new SparseVector(sparseVector);
            })
            .ToList();
    }
}
