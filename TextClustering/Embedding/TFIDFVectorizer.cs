namespace TextClustering.Embedding;

using TermFrequency = Dictionary<string, int>;

public sealed class TFIDFVectorizer(BoWVectorizerConfig config) : CountVectorizer(config)
{
    protected override Dictionary<int, float>[] ToSparseVector(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);
        return documentTermFrequency
            .AsParallel()
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

                    sparseVector.Add(
                            Vocabulary[tf.Key].Id,
                            ComputeTFIDF(tf.Value, TotalDocumentCount, value!.NumberOfDocumentsWhereTheTermAppears));
                }
                return sparseVector;
            })
            .ToArray();
    }

    // tfidf(t, d, D) = tf(t,d) * idf(t,D)
    // where tf(t,d) = count of term 't' in document 'd'
    // idf(t,D) = logarithmically scaled inverse fraction
    // of total number of documents by number of document containing 't'.
    public static float ComputeTFIDF(int termFrequency, long totalDocumentCount, int numberOfDocumentsWhereTheTermAppears)
        => (float)((1 + Math.Log(termFrequency)) * Math.Log((double)totalDocumentCount / numberOfDocumentsWhereTheTermAppears));
}
