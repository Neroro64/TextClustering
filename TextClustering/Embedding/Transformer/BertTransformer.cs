using FastBertTokenizer;

using Microsoft.ML.OnnxRuntime;

namespace TextClustering.Embedding.Transformer;

public sealed class BertTransformer : IVectorizer, IDisposable
{
    public BertTokenizer Tokenizer { get; init; } = new BertTokenizer();

    private const string _tokenizerFilePath = "Transformer/PretrainedModel/tokenizer.json";
    private const string _modelFilePath = "Transformer/PretrainedModel/all-MiniLM-L6-v2.onnx";
    private readonly BertTransformerSettings _settings;

    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;

    private readonly float[] _outputBuffer;

    public BertTransformer(BertTransformerSettings? settings = null)
    {
        _settings = settings ?? new BertTransformerSettings();

        // Load the tokenizer
        using var tokenizerConfigStream = new FileStream(_tokenizerFilePath, FileMode.Open, FileAccess.Read);
        Tokenizer.LoadTokenizerJson(tokenizerConfigStream);

        _sessionOptions = _settings.useCuda
            ? SessionOptions.MakeSessionOptionWithCudaProvider(0)
            : new SessionOptions();
        _sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        _session = new InferenceSession(_modelFilePath, _sessionOptions);

        _outputBuffer = new float[_settings.sequenceLength];

        // using var output = OrtValue.CreateTensorValueFromMemory(new float[384], [1, 384]);
        // using var runOptions = new RunOptions();
        // var (_inputIds, _attentionMask, _) = Tokenizer.Encode("HELLo world bruh", padTo: 256);
        // using var inputIds = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _inputIds, [1, 256]);
        // using var attentionMask = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _attentionMask, [1, 256]);
        // session.Run(
        //         runOptions,
        //         inputNames: ["input_ids", "attention_mask"],
        //         inputValues: [inputIds, attentionMask],
        //         outputNames: ["sentence_embedding"],
        //         outputValues: [output]
        //         );
        //
        // float[] embedding = output.GetTensorDataAsSpan<float>().ToArray();
        // string s = string.Join(", ", embedding);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    public void Reset() { }

    public void Fit(IEnumerable<string> documents) { }

    public Dictionary<int, float>[] Transform(IEnumerable<string> documents)
    {
        throw new NotImplementedException();
    }

    public Dictionary<int, float>[] FitThenTransform(IEnumerable<string> documents)
    {
        throw new NotImplementedException();
    }
}
