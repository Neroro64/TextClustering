namespace TextClustering.Embedding.Transformer;

public record BertTransformerSettings(
    bool useCuda = false,
    bool convertToLowercase = true,
    int sequenceLength = 384
)
{
    public (string name, int[] dimension)[] ModelInputSpec { get; init; } = [("input_ids", [1, 256]), ("attention_mask", [1, 256])];
    public (string name, int[] dimension)[] ModelOutputSpec { get; init; } = [("sentence_embedding", [1, 384])];
}
