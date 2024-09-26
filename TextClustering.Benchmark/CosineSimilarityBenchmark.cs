using System.Numerics;

using BenchmarkDotNet.Attributes;

using Embedding;
using Embedding.DistanceMetric;

using NumpyDotNet;

namespace TextClustering.Benchmark;

[CsvExporter]
public class CosineSimilarityBenchmark
{
    private List<(DenseVector, DenseVector)> DataSet1 { get; init; } = [];
    private List<(ndarray, ndarray)> DataSet2 { get; init; } = [];
    private List<(SparseVector, SparseVector)> DataSet3 { get; init; } = [];
    private List<(Vector<float>, Vector<float>)> DataSet4 { get; init; } = [];

    [Params(128, 256, 512, 1024)]
    public int VectorSize;

    [Params(10, 100, 1_000, 10_000, 100_000, 1_000_000)]
    public int DatasetSize;

    private readonly int _simdVectorLength = Vector<float>.Count;

    [GlobalSetup]
    public void Setup()
    {
        np.tuning.EnableTryCatchOnCalculations = false;
        var random = new Random();
        for (int i = 0; i < DatasetSize; ++i)
        {
            DataSet1.Add(GenerateRandomFloatArrays(random, VectorSize, -100f, 100f));
            DataSet2.Add(GenerateRandomFloatNdArray(random, VectorSize, -100f, 100f));
            DataSet3.Add(GenerateRandomFloatSparseVectors(random, VectorSize, -100f, 100f));
            DataSet4.Add(GenerateRandomFloatVectors(random, VectorSize, -100f, 100f));
        }
    }

    private static float GenerateRandomFloat(Random random, float minValue, float maxValue) => ((float)random.NextDouble() * (maxValue - minValue)) + minValue;

    private static (DenseVector, DenseVector) GenerateRandomFloatArrays(Random random, int vectorSize, float minValue, float maxValue)
    {
        float[] vector1 = new float[vectorSize];
        float[] vector2 = new float[vectorSize];
        for (int i = 0; i < vectorSize; ++i)
        {
            vector1[i] = GenerateRandomFloat(random, minValue, maxValue);
            vector2[i] = GenerateRandomFloat(random, minValue, maxValue);
        }
        return (new(vector1), new(vector2));
    }

    private static (ndarray, ndarray) GenerateRandomFloatNdArray(Random random, int vectorSize, float minValue, float maxValue)
    {
        float[] vector1 = new float[vectorSize];
        float[] vector2 = new float[vectorSize];
        for (int i = 0; i < vectorSize; ++i)
        {
            vector1[i] = GenerateRandomFloat(random, minValue, maxValue);
            vector2[i] = GenerateRandomFloat(random, minValue, maxValue);
        }
        return (np.array(vector1), np.array(vector2));
    }

    private static (SparseVector, SparseVector) GenerateRandomFloatSparseVectors(Random random, int vectorSize, float minValue, float maxValue)
    {
        Dictionary<int, float> vector1 = [];
        Dictionary<int, float> vector2 = [];
        for (int i = 0; i < vectorSize; ++i)
        {
            vector1.Add(i, ((float)random.NextDouble() * (maxValue - minValue)) + minValue);
            vector2.Add(i, ((float)random.NextDouble() * (maxValue - minValue)) + minValue);
        }
        return (new(vector1), new(vector2));
    }

    private static (Vector<float>, Vector<float>) GenerateRandomFloatVectors(Random random, int vectorSize, float minValue, float maxValue)
    {
        float[] vector1 = new float[vectorSize];
        float[] vector2 = new float[vectorSize];
        for (int i = 0; i < vectorSize; ++i)
        {
            vector1[i] = GenerateRandomFloat(random, minValue, maxValue);
            vector2[i] = GenerateRandomFloat(random, minValue, maxValue);
        }
        return (new(vector1), new(vector2));
    }

    [Benchmark]
    public void CompulteCosineSimilarityOnFloatArray()
    {
        foreach (var (vector1, vector2) in DataSet1)
        {
            _ = CosineSimilarity.CalculateDistance(vector1, vector2);
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarityOnNdarray()
    {
        foreach (var (vector1, vector2) in DataSet2)
        {
            _ = np.dot(vector1, vector2) / (np.sqrt(np.dot(vector1, vector1)) * np.sqrt(np.dot(vector2, vector2)));
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

    [Benchmark]
    public void ComputeCosineSimilarityOnVector()
    {
        // Since the System.Numerics.Vector<T> has fixed size, we need to vectorize the calculation manually.
        const int numberOfDotProducts = 3;
        int numberOfRepetitions = Math.Max(1, VectorSize / _simdVectorLength) * numberOfDotProducts;
        for (int i = 0; i < numberOfRepetitions; ++i)
        {
            foreach (var (vector1, vector2) in DataSet4)
            {
                _ = Vector.Dot(vector1, vector2) / (Math.Sqrt(Vector.Dot(vector1, vector1)) * Math.Sqrt(Vector.Dot(vector2, vector2)));
            }
        }
    }
}
