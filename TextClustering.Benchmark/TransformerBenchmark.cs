using System.Text;

using BenchmarkDotNet.Attributes;

using Embedding.Transformer;

namespace TextClustering.Benchmark;

[RPlotExporter]
public class TransformerBenchmark : IDisposable
{
    private List<string> Documents { get; init; } = [];

    private readonly BertTransformer _transformer = new();

    [Params(10, 100, 1_000)]
    public int DatasetSize;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random();
        for (int i = 0; i < DatasetSize; ++i)
        {
            Documents.Add(GenerateRandomDocument(random, 2, 512));
        }
    }

    private static string GenerateRandomDocument(Random random, int minLength, int maxLength)
    {
        int length = random.Next(minLength, maxLength + 1);
        var sb = new StringBuilder(length);

        for (int i = 0; i < length; i++)
        {
            char c = (char)random.Next(32, 127); // ASCII printable characters
            _ = sb.Append(c);
        }

        return sb.ToString();
    }

    [Benchmark]
    public void Transform()
    {
        _ = _transformer.Transform(Documents);
    }

    public void Dispose()
    {
        _transformer.Dispose();
        GC.SuppressFinalize(this);
    }
}
