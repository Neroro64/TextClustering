namespace TextClustering.Embedding;

public record BoWVectorizerConfig
{
    public char[] WordSeparator { get; init; } = [];

    public int MinWordLength { get; init; } = 2;
    public float MaxDocumentPresence { get; init; } = 0.25f;
    public Language[] Languages { get; init; } = [Language.English];
    public bool Lowercase { get; init; } = true;
}
