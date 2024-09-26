using System.Numerics;

namespace Embedding.DistanceMetric;

public static class NumericExtensions
{
    public static Vector<float>[] ToSIMDVectors(this float[] data)
    {
        int vectorLength = Vector<float>.Count;
        int segmentCount = (data.Length + vectorLength - 1) / vectorLength;
        var vectorSegments = new Vector<float>[segmentCount];
        for (int i = 0; i < segmentCount; ++i)
        {
            int start = i * vectorLength;
            int length = Math.Min(vectorLength, data.Length - start);
            vectorSegments[i] = new(data.AsSpan<float>(start, length));
        }
        return vectorSegments;
    }
}
