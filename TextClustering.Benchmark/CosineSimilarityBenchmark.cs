using BenchmarkDotNet.Attributes;

using Embedding;
using Embedding.DistanceMetric;

using NumpyDotNet;

namespace TextClustering.Benchmark;

[RPlotExporter]
public class CosineSimilarityBenchmark
{
    private const int VectorSize = 128;
    private List<(DenseVector, DenseVector)> DataSet { get; init; } = [];
    private List<(ndarray, ndarray)> DataSet2 { get; init; } = [];
    private List<(SparseVector, SparseVector)> DataSet3 { get; init; } = [];

    [Params(100_000)]
    public int DatasetSize;

    [GlobalSetup]
    public void Setup()
    {
        np.tuning.EnableTryCatchOnCalculations = false;
        var random = new Random();
        for (int i = 0; i < DatasetSize; ++i)
        {
            DataSet.Add(GenerateRandomFloatVectors(random, -100f, 100f));
            DataSet2.Add(GenerateRandomFloatNdArray(random, -100f, 100f));
            DataSet3.Add(GenerateRandomFloatVectors3(random, -100f, 100f));
        }
    }

    private static (DenseVector, DenseVector) GenerateRandomFloatVectors(Random random, float minValue, float maxValue)
    {
        float[] vector1 = new float[VectorSize];
        float[] vector2 = new float[VectorSize];
        for (int i = 0; i < VectorSize; ++i)
        {
            vector1[i] = ((float)random.NextDouble() * (maxValue - minValue)) + minValue;
            vector2[i] = ((float)random.NextDouble() * (maxValue - minValue)) + minValue;
        }
        return (new(vector1), new(vector2));
    }

    private static (ndarray, ndarray) GenerateRandomFloatNdArray(Random random, float minValue, float maxValue)
    {
        float[] vector1 = new float[VectorSize];
        float[] vector2 = new float[VectorSize];
        for (int i = 0; i < VectorSize; ++i)
        {
            vector1[i] = ((float)random.NextDouble() * (maxValue - minValue)) + minValue;
            vector2[i] = ((float)random.NextDouble() * (maxValue - minValue)) + minValue;
        }
        return (np.array(vector1), np.array(vector2));
    }

    private static (SparseVector, SparseVector) GenerateRandomFloatVectors3(Random random, float minValue, float maxValue)
    {
        Dictionary<int, float> vector1 = [];
        Dictionary<int, float> vector2 = [];
        for (int i = 0; i < VectorSize; ++i)
        {
            vector1.Add(i, ((float)random.NextDouble() * (maxValue - minValue)) + minValue);
            vector2.Add(i, ((float)random.NextDouble() * (maxValue - minValue)) + minValue);
        }
        return (new(vector1), new(vector2));
    }

    [Benchmark]
    public void CompulteCosineSimilarityOnFloatArray()
    {
        foreach (var (vector1, vector2) in DataSet)
        {
            _ = CosineSimilarity.CalculateDistance(vector1, vector2);
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarityOnNdarray()
    {
        foreach (var (vector1, vector2) in DataSet2)
        {
            _ = np.dot(vector1, vector2) / np.sqrt(np.dot(vector1, vector1) + np.dot(vector2, vector2));
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarityOnDictionary()
    {
        foreach (var (vector1, vector2) in DataSet3)
        {
            _ = CosineSimilarity.CalculateDistance(vector1, vector2);
        }
    }
}
