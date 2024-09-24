namespace Embedding.Transformer;

/// <summary>
///     Settings for a BERT transformer model.
/// </summary>
/// <param name="RuntimeExecutionProvider">Specifies the execution provider for ONNX Runtime.</param>
/// <param name="InputDimension">The dimension of the model's input embeddings.</param>
/// <param name="EmbeddingDimension">The dimension of the model's embedding layer.</param>
/// <param name="BatchSize">The batch size for inference.</param>
/// <param name="StrideSize">The stride size for inference batching.</param>
public record BertTransformerSettings(
    OnnxRuntimeExecutionProvider RuntimeExecutionProvider = OnnxRuntimeExecutionProvider.CPU,
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

public enum OnnxRuntimeExecutionProvider
{
    CPU,
    CUDA,
    DirectML
}
