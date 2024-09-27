using System.Numerics;
using System.Numerics.Tensors;

using BenchmarkDotNet.Attributes;

using Embedding.DistanceMetric;

using NumpyDotNet;

namespace TextClustering.Benchmark;

[JsonExporterAttribute.Brief]
public class CosineSimilarityBenchmark
{
    private List<(DenseVector, DenseVector)> DataSet1 { get; } = [];
    private List<(ndarray, ndarray)> DataSet2 { get; } = [];
    private List<(SparseVector, SparseVector)> DataSet3 { get; } = [];

    [Params(64, 256, 1024, 4096)]
    public int VectorSize;

    [Params(100, 10_000, 1_000_000)]
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

    [Benchmark]
    public void ComputeCosineSimilarity_OnFloatArray()
    {
        foreach (var (vector1, vector2) in DataSet1)
        {
            double distance = 0;
            double v1Sum = 0;
            double v2Sum = 0;
            for (int i = 0; i < vector1.Length; ++i)
            {
                distance += vector1[i] * vector2[i];
                v1Sum += vector1[i] * vector1[i];
                v2Sum += vector2[i] * vector2[i];
            }
            _ = 1f - (float)(distance / Math.Max(Math.Sqrt(v1Sum) * Math.Sqrt(v2Sum), 1e-10));
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarity_OnNdarray()
    {
        foreach (var (vector1, vector2) in DataSet2)
        {
            _ = np.dot(vector1, vector2) / (np.sqrt(np.dot(vector1, vector1)) * np.sqrt(np.dot(vector2, vector2)));
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarity_OnDictionary()
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
        // This benchmark is only for comparing what the performance would look like if we are able to use Vectors only.
        const int numberOfDotProducts = 3;
        int numberOfRepetitions = Math.Max(1, VectorSize / _simdVectorLength) * numberOfDotProducts;
        foreach (var (vector1, vector2) in DataSet1)
        {
            var v1 = new Vector<float>(vector1.Data);
            var v2 = new Vector<float>(vector2.Data);

            for (int i = 0; i < numberOfRepetitions; ++i)
            {
                _ = Vector.Dot(v1, v2) / (Math.Sqrt(Vector.Dot(v1, v1)) * Math.Sqrt(Vector.Dot(v2, v2)));
            }
        }
    }

    [Benchmark]
    public void ComputeCosineSimilarityOnTensor()
    {
        foreach (var (vector1, vector2) in DataSet1)
        {
            _ = TensorPrimitives.CosineSimilarity(vector1.Data, vector2.Data);
        }
    }
}
