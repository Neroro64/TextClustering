using System.Text.RegularExpressions;

using StopWord;

/// <summary>
///     A vectorizer that converts text documents into numerical vectors by counting the occurrences of each word.
/// </summary>
/// <remarks>
///     This class is designed to be used in natural language processing (NLP) tasks, such as clustering or classification,
///     where text data needs to be transformed into a format that can be processed by machine learning algorithms.
/// </remarks>
namespace TextClustering.Embedding.BoWVectorizer;

using TermFrequency = Dictionary<string, int>;

public class CountVectorizer(BoWVectorizerConfig config) : IVectorizer
{
    /// <summary>
    ///     Gets the configuration for this vectorizer.
    /// </summary>
    protected BoWVectorizerConfig Config => config;

    /// <summary>
    ///     Regular expression used to split text documents into words.
    /// </summary>
    protected readonly Regex WordSeparatorRegex = BuildSeparatorRegexPattern(config.WordSeparator);

    /// <summary>
    ///     Set of stop words that will be ignored during the vocabulary building.
    /// </summary>
    protected readonly HashSet<string> StopWord = GetStopWordSet(config.Languages, config.Lowercase);

    /// <summary>
    ///     Vocabulary containing all unique terms found in the documents along with their term statistics.
    /// </summary>
    public Dictionary<string, TermStats> Vocabulary { get; } = new();

    /// <summary>
    ///     Total number of documents.
    /// </summary>
    public long TotalDocumentCount { get; protected set; }

    /// <inheritdoc/>
    public void Reset()
    {
        Vocabulary.Clear();
        TotalDocumentCount = 0;
    }

    /// <inheritdoc/>
    public void Fit(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
    }

    /// <inheritdoc/>
    public Dictionary<int, float>[] Transform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        return ToSparseVector(documentTermFrequency);
    }

    /// <inheritdoc/>
    public Dictionary<int, float>[] FitThenTransform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
        return ToSparseVector(documentTermFrequency);
    }

    /// <summary>
    ///     Extracts term frequencies from a collection of text documents in parallel.
    /// </summary>
    /// <param name="documents">Collection of text documents to extract term frequencies from.</param>
    /// <returns>List of term frequency dictionaries, where each dictionary represents a document and contains terms as keys and their frequencies as values.</returns>
    protected virtual List<TermFrequency> ExtractTermFrequency(IEnumerable<string> documents)
    {
        return documents
            .AsParallel()
            .Select(ExtractTermFrequency)
            .Where(static tf => tf.Keys.Count > 0)
            .ToList();
    }

    /// <summary>
    ///     Extracts term frequencies from a single text document.
    /// </summary>
    /// <param name="document">Text document to extract term frequencies from.</param>
    /// <returns>Term frequency dictionary containing terms as keys and their frequencies as values.</returns>
    protected TermFrequency ExtractTermFrequency(string document)
    {
        var termFrequency = new TermFrequency();
        var words = WordSeparatorRegex.Split(document).Where(word => !string.IsNullOrWhiteSpace(word));

        foreach (string word in words)
        {
            if (word.Length < Config.MinWordLength || StopWord.Contains(word))
            {
                continue;
            }

            string term = Config.Lowercase ? word.ToLowerInvariant() : word;

            if (!termFrequency.TryAdd(term, 1))
            {
                termFrequency[term]++;
            }
        }

        return termFrequency;
    }

    /// <summary>
    ///     Updates the vocabulary by incorporating new terms and incrementing the document counts for existing terms.
    /// </summary>
    /// <param name="documentTermFrequency">Collection of term frequencies from documents to update the vocabulary with.</param>
    protected void UpdateVocabulary(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int newTermId = Vocabulary.Count + 1;

        foreach (var termFrequency in documentTermFrequency)
        {
            foreach (var term in termFrequency)
            {
                if (Vocabulary.TryGetValue(term.Key, out var termStats))
                {
                    termStats.NumberOfDocumentsWhereTheTermAppears += 1;
                }
                else
                {
                    Vocabulary.Add(term.Key, new TermStats { Id = newTermId++, NumberOfDocumentsWhereTheTermAppears = 1 });
                }
            }
        }
    }

    /// <summary>
    ///     Transforms a collection of term frequencies into sparse vectors by filtering out terms that are either not in the vocabulary
    ///     or appear too frequently.
    /// </summary>
    /// <param name="documentTermFrequency">Collection of term frequencies to transform.</param>
    /// <returns>Array of sparse vectors, where each vector represents a document and contains term IDs as keys and term frequencies as values.</returns>
    protected virtual Dictionary<int, float>[] ToSparseVector(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);

        return documentTermFrequency
            .AsParallel()
            .Select(termFrequency =>
            {
                Dictionary<int, float> sparseVector = new();

                foreach (var tf in termFrequency)
                {
                    if (!Vocabulary.TryGetValue(tf.Key, out var value) || value?.NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                    {
                        continue;
                    }

                    sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
                }

                return sparseVector;
            })
            .ToArray();
    }

    /// <summary>
    ///     Retrieves a set of stop words for the specified languages, optionally converting them to lowercase.
    /// </summary>
    /// <param name="languages">Collection of languages to retrieve stop words for.</param>
    /// <param name="toLowercase">Flag indicating whether stop words should be converted to lowercase.</param>
    /// <returns>Set of stop words for the specified languages, possibly in lowercase.</returns>
    protected static HashSet<string> GetStopWordSet(Language[] languages, bool toLowercase)
    {
        return languages is []
            ? new HashSet<string>()
            : languages.SelectMany(
                    language => StopWords.GetStopWords(language.GetShortCode())
                    .Select(x => toLowercase ? x.ToLowerInvariant() : x))
                .ToHashSet();
    }

    /// <summary>
    ///     Builds a regular expression pattern for splitting text documents into words based on the specified separators.
    /// </summary>
    /// <param name="separators">Collection of word separators.</param>
    /// <returns>Regular expression pattern that matches word separators.</returns>
    protected static Regex BuildSeparatorRegexPattern(char[] separators)
    {
        return separators is []
            ? new Regex(@"[^a-zA-Z0-9]", RegexOptions.CultureInvariant)
            : new Regex($"[{Regex.Escape(new string(separators))}]", RegexOptions.CultureInvariant);
    }
}
