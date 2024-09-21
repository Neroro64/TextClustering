using System.Text.RegularExpressions;

using StopWord;

namespace TextClustering.Embedding;

using TermFrequency = Dictionary<string, int>;

public class CountVectorizer(BoWVectorizerConfig config) : IVectorizer
{
    private readonly Regex WordSeparatorRegex = BuildSeparatorRegexPattern(config.WordSeparator);
    private readonly HashSet<string> StopWord = GetStopWordSet(config.Languages, config.Lowercase);
    public Dictionary<string, TermStats> Vocabulary { get; } = [];
    public long TotalDocumentCount { get; private set; }

    public void Reset()
    {
        Vocabulary.Clear();
    }

    public void Fit(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
    }

    public Dictionary<int, float>[] Transform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        return ToSparseVector(documentTermFrequency);
    }

    public Dictionary<int, float>[] FitThenTransform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
        return ToSparseVector(documentTermFrequency);
    }

    private List<TermFrequency> ExtractTermFrequency(IEnumerable<string> documents)
    {
        return documents
            .AsParallel()
            .Select(ExtractTermFrequency)
            .Where(static tf => tf.Keys.Count > 0)
            .ToList();
    }

    private TermFrequency ExtractTermFrequency(string document)
    {
        var termFrequency = new TermFrequency();
        var words = WordSeparatorRegex
            .Split(document)
            .Where(word => !string.IsNullOrWhiteSpace(word));

        foreach (string word in words)
        {
            if (word.Length < config.MinWordLength
                || StopWord.Contains(word))
            {
                continue;
            }

            string term = config.Lowercase ? word.ToLowerInvariant() : word;

            if (!termFrequency.TryAdd(term, 1))
            {
                termFrequency[term]++;
            }
        }

        return termFrequency;
    }

    private void UpdateVocabulary(IEnumerable<TermFrequency> documentTermFrequency)
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
                    Vocabulary.Add(term.Key, new TermStats(id: newTermId++, documentFrequency: 1));
                }
            }
        }
    }

    private Dictionary<int, float>[] ToSparseVector(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * config.MaxDocumentPresence);
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
                    sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
                }
                return sparseVector;
            })
            .ToArray();
    }

    private static HashSet<string> GetStopWordSet(Language[] languages, bool toLowercase)
    {
        return languages is []
            ? []
            : languages.SelectMany(
                    language => StopWords.GetStopWords(language.GetShortCode())
                    .Select(x => toLowercase ? x.ToLowerInvariant() : x))
                .ToHashSet();
    }

    private static Regex BuildSeparatorRegexPattern(char[] separators)
    {
        return separators is []
            ? new Regex(@"[^a-zA-Z0-9]", RegexOptions.CultureInvariant)
            : new Regex($"[{Regex.Escape(new string(separators))}]", RegexOptions.CultureInvariant);
    }
}
