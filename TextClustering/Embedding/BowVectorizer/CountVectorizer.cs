using System.Collections.ObjectModel;
using System.Text.RegularExpressions;

using StopWord;

namespace Embedding.BowVectorizer;

/// <summary>
///     A vectorizer that converts text documents into sparse vectors by counting the occurrences of each word.
/// </summary>
/// <remarks>
///     This class is designed to be used in natural language processing (NLP) tasks, such as clustering or classification,
///     where text data needs to be transformed into a format that can be processed by machine learning algorithms.
/// </remarks>
public class CountVectorizer(BoWVectorizerConfig config) : IVectorizer<SparseVector>
{
    /// <summary>
    ///     Gets the configuration for this vectorizer.
    /// </summary>
    protected BoWVectorizerConfig Config => config;

    /// <summary>
    ///     Regular expression used to split text documents into words.
    /// </summary>
    private Regex WordSeparatorRegex { get; } = BuildSeparatorRegexPattern(config.WordSeparator);

    /// <summary>
    ///     Set of stop words that will be ignored during the vocabulary building.
    /// </summary>
    private HashSet<string> StopWord { get; } = GetStopWordSet(config.Languages, config.Lowercase);

    /// <summary>
    ///     Vocabulary containing all unique terms found in the documents along with their term statistics.
    /// </summary>
    public Dictionary<string, TermStats> Vocabulary { get; } = [];

    /// <summary>
    ///     Total number of documents.
    /// </summary>
    protected long TotalDocumentCount { get; private set; }

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
    public ReadOnlyCollection<SparseVector> Transform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        return new(ToSparseVector(documentTermFrequency));
    }

    /// <inheritdoc/>
    public ReadOnlyCollection<SparseVector> FitThenTransform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
        return new(ToSparseVector(documentTermFrequency));
    }

    /// <summary>
    ///     Extracts term frequencies from a collection of text documents in parallel.
    /// </summary>
    /// <param name="documents">Collection of text documents to extract term frequencies from.</param>
    /// <returns>List of term frequency dictionaries, where each dictionary represents a document and contains terms as keys and their frequencies as values.</returns>
    protected virtual IList<Dictionary<string, int>> ExtractTermFrequency(IEnumerable<string> documents)
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
    protected Dictionary<string, int> ExtractTermFrequency(string document)
    {
        var termFrequency = new Dictionary<string, int>();
        var words = WordSeparatorRegex.Split(document).Where(static word => !string.IsNullOrWhiteSpace(word));

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
    private void UpdateVocabulary(IEnumerable<Dictionary<string, int>> documentTermFrequency)
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
    protected virtual IList<SparseVector> ToSparseVector(IEnumerable<Dictionary<string, int>> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * Config.MaxDocumentPresence);

        return documentTermFrequency
            .AsParallel()
            .Select(termFrequency =>
            {
                Dictionary<int, float> sparseVector = [];

                foreach (var tf in termFrequency)
                {
                    if (!Vocabulary.TryGetValue(tf.Key, out var value) || value.NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                    {
                        continue;
                    }

                    sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
                }

                return new SparseVector(sparseVector);
            })
            .ToList();
    }

    /// <summary>
    ///     Retrieves a set of stop words for the specified languages, optionally converting them to lowercase.
    /// </summary>
    /// <param name="languages">Collection of languages to retrieve stop words for.</param>
    /// <param name="toLowercase">Flag indicating whether stop words should be converted to lowercase.</param>
    /// <returns>Set of stop words for the specified languages, possibly in lowercase.</returns>
    private static HashSet<string> GetStopWordSet(Language[] languages, bool toLowercase)
    {
        return languages is []
            ? []
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
    private static Regex BuildSeparatorRegexPattern(char[] separators)
    {
        return separators is []
            ? new Regex(@"[^a-zA-Z0-9]", RegexOptions.CultureInvariant)
            : new Regex($"[{Regex.Escape(new string(separators))}]", RegexOptions.CultureInvariant);
    }
}
