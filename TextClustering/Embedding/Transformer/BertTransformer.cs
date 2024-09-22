#pragma warning disable FBERTTOK001

using System.Collections.ObjectModel;

using FastBertTokenizer;

using Microsoft.ML.OnnxRuntime;

namespace TextClustering.Embedding.Transformer;

public sealed class BertTransformer : IVectorizer<float[]>, IDisposable
{
    public BertTokenizer Tokenizer { get; init; } = new BertTokenizer();

    private const string _tokenizerJsonPath = "Transformer/PretrainedModel/tokenizer.json";
    private const string _onnxModelPath = "Transformer/PretrainedModel/all-MiniLM-L6-v2.onnx";
    private readonly BertTransformerSettings _settings;

    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;

    private readonly float[] _outputBuffer;

    public BertTransformer(
            string? tokenizerJsonPath = null,
            string? onnxModelPath = null,
            BertTransformerSettings? settings = null
    )
    {
        _settings = settings ?? new BertTransformerSettings();

        // Load the tokenizer
        using var tokenizerConfigStream = new FileStream(tokenizerJsonPath ?? _tokenizerJsonPath, FileMode.Open, FileAccess.Read);
        Tokenizer.LoadTokenizerJson(tokenizerConfigStream);

        // Intialize the InferenceSession and set logging level to fatal.
        _sessionOptions = _settings.UseCuda
            ? SessionOptions.MakeSessionOptionWithCudaProvider(0)
            : new SessionOptions();
        _sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        _session = new InferenceSession(onnxModelPath ?? _onnxModelPath, _sessionOptions);

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

    public void Fit(IEnumerable<string> documents) { }

    public ReadOnlyCollection<float[]> Transform(IEnumerable<string> documents)
    {
        var batchSize = _settings.BatchSize;
        var inputLayerNames = _settings.ModelInputLayerNames;
        var inputDimention = _settings.InputDimension;
        var outputLayerNames = _settings.ModelOutputLayerNames;
        var outputDimention = _settings.EmbeddingDimension;

        using var runOptions = new RunOptions();
        using var output = OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, _outputBuffer, [batchSize, outputDimention]);

        int documentCount = 0;
        List<float[]> embeddings = [];
        foreach (var batch in Tokenizer.CreateBatchEnumerator(
                    documents.Select(doc => (documentCount++, doc)),
                    _settings.InputDimension,
                    batchSize,
                    _settings.StrideSize))
        {
            var inputIds = batch.InputIds;
            var attentionMask = batch.AttentionMask;

            using var inputIdTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputIds, [batchSize, inputDimention]);
            using var attentionMaskTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, attentionMask, [batchSize, inputDimention]);

            _session.Run(
                    runOptions,
                    inputNames: inputLayerNames,
                    inputValues: [inputIdTensor, attentionMaskTensor],
                    outputNames: outputLayerNames,
                    outputValues: [output]
                    );

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

    public ReadOnlyCollection<float[]> FitThenTransform(IEnumerable<string> documents) => Transform(documents);
}
