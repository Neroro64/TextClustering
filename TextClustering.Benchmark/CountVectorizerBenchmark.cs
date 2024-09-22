#pragma warning disable CS0649, CA1812
using System.Text;

using BenchmarkDotNet.Attributes;

using TextClustering.Embedding.BoWVectorizer;

namespace TextClustering.Benchmark;

[RPlotExporter]
internal sealed class CountVectorizerBenchmark
{
    internal List<string> Documents { get; init; } = [];

    private readonly CountVectorizer multiThreaddedVectorizer = new(new() { Languages = [] });
    private readonly SingleThreaddedCountVectorizer singleThreaddedVectorizer = new(new() { Languages = [] });

    [Params(10, 100, 1_000, 10_000, 100_000)]
    internal int DatasetSize;

    [GlobalSetup]
    internal void Setup()
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
    internal void SingleThreaddedVectorize()
    {
        _ = singleThreaddedVectorizer.FitThenTransform(Documents).ToList();
    }

    [Benchmark]
    internal void MultiThreaddedVectorize()
    {
        _ = multiThreaddedVectorizer.FitThenTransform(Documents).ToList();
    }
}
