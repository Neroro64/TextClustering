#pragma warning disable FBERTTOK001

using System.Collections.ObjectModel;

using FastBertTokenizer;

using Microsoft.ML.OnnxRuntime;

namespace TextClustering.Embedding.Transformer;

/// <summary>
///   A BERT-based transformer that converts input documents into embeddings using pre-trained models.
///   By default, it uses the pretrained 'sentence-transformers/all-MiniLM-L6-v2'.
/// </summary>
public sealed class BertTransformer : IVectorizer<float[]>, IDisposable
{
    /// <summary>
    ///   The BertTokenizer is used to tokenize input texts into token IDs and attention masks that can be processed by the transformer model.
    /// </summary>
    public BertTokenizer Tokenizer { get; init; } = new BertTokenizer();

    /// <summary>
    /// A buffer to store embedding outputs.
    /// </summary>
    private readonly float[] _outputBuffer;

    private const string _defaultTokenizerJsonPath = "Transformer/PretrainedModel/tokenizer.json";
    private const string _defaultOnnxModelPath = "Transformer/PretrainedModel/all-MiniLM-L6-v2.onnx";

    private readonly BertTransformerSettings _settings;
    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;


    /// <summary>
    ///   Initializes a new instance of the BertTransformer class with optional parameters for a different tokenizer, ONNX model path, and settings.
    /// </summary>
    /// <param name="tokenizerJsonPath">Optional path to tokenizer json file.</param>
    /// <param name="onnxModelPath">Optional path to ONNX model file.</param>
    /// <param name="settings">Optional BertTransformerSettings.</param>
    public BertTransformer(
        string? tokenizerJsonPath = null,
        string? onnxModelPath = null,
        BertTransformerSettings? settings = null
    )
    {
        _settings = settings ?? new BertTransformerSettings();

        // Load the tokenizer
        using var tokenizerConfigStream = new FileStream(tokenizerJsonPath ?? _defaultTokenizerJsonPath, FileMode.Open, FileAccess.Read);
        Tokenizer.LoadTokenizerJson(tokenizerConfigStream);

        // Intialize the InferenceSession and set logging level to fatal.
        _sessionOptions = _settings.UseCuda
            ? SessionOptions.MakeSessionOptionWithCudaProvider(0)
            : new SessionOptions();
        _sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        _session = new InferenceSession(onnxModelPath ?? _defaultOnnxModelPath, _sessionOptions);

        // Initialize the output buffer for storing embedding outputs.
        _outputBuffer = new float[_settings.BatchSize * _settings.EmbeddingDimension];
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    public void Reset() { }

    /// <summary>
    ///   Not used.
    /// </summary>
    public void Fit(IEnumerable<string> documents) { }

    /// <summary>
    ///   Transforms the provided documents into embeddings.
    /// </summary>
    /// <param name="documents">A sequence of document strings to be vectorized.</param>
    /// <returns>A sequence of document vectors as float arrays.</returns>
    public ReadOnlyCollection<float[]> Transform(IEnumerable<string> documents)
    {
        // Get the model parameters. 
        string[] inputLayerNames = _settings.ModelInputLayerNames;
        int batchSize = _settings.BatchSize;
        int inputDimention = _settings.InputDimension;
        string[] outputLayerNames = _settings.ModelOutputLayerNames;
        int outputDimention = _settings.EmbeddingDimension;

        using var runOptions = new RunOptions();
        using var output = OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, _outputBuffer, [batchSize, outputDimention]);

        int documentCount = 0;
        List<float[]> embeddings = [];
        foreach (var batch in Tokenizer.CreateBatchEnumerator(
            sourceEnumerable: documents.Select(doc => (documentCount++, doc)),
            tokensPerInput: _settings.InputDimension,
            batchSize: batchSize,
            stride: _settings.StrideSize))
        {
            var inputIds = batch.InputIds;
            var attentionMask = batch.AttentionMask;

            using var inputIdTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputIds, [batchSize, inputDimention]);
            using var attentionMaskTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, attentionMask, [batchSize, inputDimention]);

            // Inference
            _session.Run(
                runOptions,
                inputNames: inputLayerNames,
                inputValues: [inputIdTensor, attentionMaskTensor],
                outputNames: outputLayerNames,
                outputValues: [output]
            );

            // Collect the outputs.
            int actualBatchSize = Math.Min(documentCount, batchSize);
            for (int i = 0; i < actualBatchSize; i++)
            {
                embeddings.Add(output
                    .GetTensorDataAsSpan<float>()
                    .Slice(i * outputDimention, outputDimention)
                    .ToArray());
            }
        }

        return new(embeddings[..documentCount]);
    }

    /// <summary>
    ///   Transforms the provided documents into embeddings.
    /// </summary>
    /// <param name="documents">A sequence of document strings to be vectorized.</param>
    /// <returns>A sequence of document vectors as float arrays.</returns>
    public ReadOnlyCollection<float[]> FitThenTransform(IEnumerable<string> documents) => Transform(documents);
}
