namespace Embedding.BowVectorizer;

/// <summary>
///     Configuration for a Bag-of-Words vectorizer.
/// </summary>
public record BoWVectorizerConfig
{
    /// <summary>
    ///     Separator characters to split words.
    /// </summary>
    public char[] WordSeparator { get; init; } = [];

    /// <summary>
    ///     Minimum word length to include in the vocabulary.
    /// </summary>
    public int MinWordLength { get; init; } = 2;

    /// <summary>
    ///     Maximum allowed proportion of documents a word can be present.
    ///     Words that exceed this percentage of precense will be excluded in vectorization.
    /// </summary>
    public float MaxDocumentPresence { get; init; } = 0.25f;

    /// <summary>
    ///     Stop words in the specified languages will be excluded in the documents
    /// </summary>
    public Language[] Languages { get; init; } = [Language.English];

    /// <summary>
    ///     Whether to lowercase document text.
    /// </summary>
    public bool Lowercase { get; init; } = true;
}
