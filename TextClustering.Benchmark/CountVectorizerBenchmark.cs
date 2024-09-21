using System.Text;

using BenchmarkDotNet.Attributes;

using TextClustering.Embedding.BoWVectorizer;

namespace TextClustering.Benchmark;

[RPlotExporter]
public class CountVectorizerBenchmark
{
    public List<string> Documents { get; init; } = new();

    private readonly CountVectorizer multiThreaddedVectorizer = new(new() { Languages = [] });
    private readonly SingleThreaddedCountVectorizer singleThreaddedVectorizer = new(new() { Languages = [] });

    [Params(10, 100, 1_000, 10_000, 100_000)]
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
    public void SingleThreaddedVectorize()
    {
        singleThreaddedVectorizer.FitThenTransform(Documents).ToList();
    }

    [Benchmark]
    public void MultiThreaddedVectorize()
    {
        multiThreaddedVectorizer.FitThenTransform(Documents).ToList();
    }
}
