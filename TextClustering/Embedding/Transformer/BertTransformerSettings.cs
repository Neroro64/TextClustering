namespace TextClustering.Embedding.Transformer;

/// <summary>
///     Settings for a BERT transformer model.
/// </summary>
/// <param name="UseCuda">Whether to use CUDA for acceleration.</param>
/// <param name="ConvertToLowercase">Whether to convert input text to lowercase.</param>
/// <param name="InputDimension">The dimension of the model's input embeddings.</param>
/// <param name="EmbeddingDimension">The dimension of the model's embedding layer.</param>
/// <param name="BatchSize">The batch size for inference.</param>
/// <param name="StrideSize">The stride size for inference batching.</param>
public record BertTransformerSettings(
    bool UseCuda = false,
    bool ConvertToLowercase = true,
    int InputDimension = 256,
    int EmbeddingDimension = 384,
    int BatchSize = 32,
    int StrideSize = 25
)
{
    /// <summary>
    /// Gets the input layer names of the BERT model.
    /// </summary>
    public string[] ModelInputLayerNames { get; init; } = ["input_ids", "attention_mask"];

    /// <summary>
    /// Gets the output layer name of the BERT model.
    /// </summary>
    public string[] ModelOutputLayerNames { get; init; } = ["sentence_embedding"];
}
