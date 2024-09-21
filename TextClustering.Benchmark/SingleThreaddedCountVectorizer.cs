using System.Text.RegularExpressions;


using TextClustering.Embedding;

namespace TextClustering.Benchmark;

using TermFrequency = Dictionary<string, int>;

public sealed class SingleThreaddedCountVectorizer(BoWVectorizerConfig config) : CountVectorizer(config)
{
    protected override List<TermFrequency> ExtractTermFrequency(IEnumerable<string> documents)
    {
        return documents
            .Select(ExtractTermFrequency)
            .Where(static tf => tf.Keys.Count > 0)
            .ToList();
    }

    protected override Dictionary<int, float>[] ToSparseVector(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);
        return documentTermFrequency
            .Select(termFrequency =>
            {
                Dictionary<int, float> sparseVector = [];
                foreach (var tf in termFrequency)
                {
                    if (!Vocabulary.TryGetValue(tf.Key, out var value)
                        || value?.NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                    {
                        continue;
                    }
                    sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
                }
                return sparseVector;
            })
            .ToArray();
    }
}
