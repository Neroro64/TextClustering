namespace TextClustering.Embedding.Transformer;

public record BertTransformerSettings(
    bool UseCuda = false,
    bool ConvertToLowercase = true,
    int InputDimension = 256,
    int EmbeddingDimension = 384,
    int BatchSize = 32,
    int StrideSize = 25
)
{
    public string[] ModelInputLayerNames { get; init; } = ["input_ids", "attention_mask"];
    public string[] ModelOutputLayerNames { get; init; } = ["sentence_embedding"];
}
