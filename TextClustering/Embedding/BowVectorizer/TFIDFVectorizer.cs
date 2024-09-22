namespace TextClustering.Embedding.BoWVectorizer;

/// <summary>
///     TF-IDF Vectorizer class that inherits from CountVectorizer, responsible for transforming term frequencies into TF-IDF vectors.
/// </summary>
/// <param name="config">BoWVectorizerConfig instance to configure the vectorizer.</param>
public sealed class TFIDFVectorizer(BoWVectorizerConfig config) : CountVectorizer(config)
{
    /// <inheritdoc/>
    protected override IList<Dictionary<int, float>> ToSparseVector(IEnumerable<Dictionary<string, int>> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);
        return documentTermFrequency
            .AsParallel()
            .Select(termFrequency =>
            {
                Dictionary<int, float> sparseVector = [];
                foreach (var tf in termFrequency)
                {
                    if (!Vocabulary.TryGetValue(tf.Key, out var value) || value?.NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                    {
                        continue;
                    }

                    sparseVector.Add(
                        Vocabulary[tf.Key].Id,
                        ComputeTFIDF(tf.Value, TotalDocumentCount, value!.NumberOfDocumentsWhereTheTermAppears));
                }
                return sparseVector;
            })
            .ToList();
    }

    /// <summary>
    ///     Computes the TF-IDF value for a given term frequency using the formula: tf(t,d) * idf(t,D).
    /// </summary>
    /// <param name="termFrequency">The frequency of the term in the current document.</param>
    /// <param name="totalDocumentCount">The total number of documents in the corpus.</param>
    /// <param name="numberOfDocumentsWhereTheTermAppears">The number of documents containing the term.</param>
    /// <returns>The computed TF-IDF value for the given term frequency.</returns>
    public static float ComputeTFIDF(int termFrequency, long totalDocumentCount, int numberOfDocumentsWhereTheTermAppears)
        => (float)((1 + Math.Log(termFrequency)) * Math.Log((double)totalDocumentCount / numberOfDocumentsWhereTheTermAppears));
}
