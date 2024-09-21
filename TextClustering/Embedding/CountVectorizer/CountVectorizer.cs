using System.Text.RegularExpressions;

using StopWord;

namespace TextClustering.Embedding.CountVectorizer;

using TermFrequency = Dictionary<string, int>;

public class CountVectorizer(BoWVectorizerConfig config) : IVectorizer
{
    private readonly Regex WordSeparatorRegex = BuildSeparatorRegexPattern(config.WordSeparator);
    private readonly HashSet<string> StopWord = GetStopWordSet(config.Languages, config.Lowercase);
    public Dictionary<string, TermStats> Vocabulary { get; } = [];
    public long TotalDocumentCount { get; private set; } = 0;

    public void Reset() => Vocabulary.Clear();

    public void Fit(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
    }

    public IEnumerable<Dictionary<int, float>> Transform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        return ToSparseVector(documentTermFrequency);
    }

    public IEnumerable<Dictionary<int, float>> FitThenTransform(IEnumerable<string> documents)
    {
        var documentTermFrequency = ExtractTermFrequency(documents);
        UpdateVocabulary(documentTermFrequency);
        TotalDocumentCount += documentTermFrequency.Count;
        return ToSparseVector(documentTermFrequency);
    }

    private List<TermFrequency> ExtractTermFrequency(IEnumerable<string> documents) => documents
            .Select(document => ExtractTermFrequency(document))
            .Where(tf => tf.Keys.Count > 0)
            .ToList();

    private TermFrequency ExtractTermFrequency(string document)
    {
        var termFrequency = new TermFrequency();
        var words = WordSeparatorRegex
            .Split(document)
            .Where(word => !String.IsNullOrWhiteSpace(word));

        foreach (var word in words)
        {
            var l = word.Length;
            if (l < config.MinWordLength
                || StopWord.Contains(word))
                continue;

            var term = config.Lowercase ? word.ToLowerInvariant() : word;

            if (termFrequency.ContainsKey(term))
                termFrequency[term]++;
            else
                termFrequency.Add(term, 1);
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
                if (Vocabulary.ContainsKey(term.Key))
                {
                    Vocabulary[term.Key].NumberOfDocumentsWhereTheTermAppears += 1;
                }
                else
                {
                    Vocabulary.Add(term.Key, new TermStats(id: newTermId++, documentFrequency: 1));
                }
            }
        }
    }

    private IEnumerable<Dictionary<int, float>> ToSparseVector(IEnumerable<TermFrequency> documentTermFrequency)
    {
        int maxDocumentFrequency = (int)(TotalDocumentCount * config.MaxDocumentPresence);
        foreach (var termFrequency in documentTermFrequency)
        {
            Dictionary<int, float> sparseVector = new();
            foreach (var tf in termFrequency)
            {
                if (!Vocabulary.ContainsKey(tf.Key)
                    || Vocabulary[tf.Key].NumberOfDocumentsWhereTheTermAppears <= maxDocumentFrequency)
                {
                    continue;
                }
                sparseVector.Add(Vocabulary[tf.Key].Id, tf.Value);
            }
            yield return sparseVector;
        }
    }

    private static HashSet<string> GetStopWordSet(Language[] languages, bool toLowercase)
    {
        if (languages is [])
            return new HashSet<string>();

        return languages
            .SelectMany(language => StopWords.GetStopWords(language.GetShortCode()).Select(x => toLowercase ? x.ToLower() : x))
            .ToHashSet();
    }

    private static Regex BuildSeparatorRegexPattern(char[] separators)
    {
        if (separators is [])
            return new Regex(@"[^a-zA-Z0-9]", RegexOptions.CultureInvariant);

        return new Regex($"[{Regex.Escape(new string(separators))}]", RegexOptions.CultureInvariant);
    }

}
