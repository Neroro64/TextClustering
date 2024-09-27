using System.Numerics;

namespace Embedding.DistanceMetric;

public static class NumericExtensions
{
    public static Vector<float>[] ToSIMDVectors(this float[] data)
    {
        int vectorLength = Vector<float>.Count;
        int segmentCount = data.Length / vectorLength;
        int totalSegmentCount = (data.Length + vectorLength - 1) / vectorLength;

        var vectorSegments = new Vector<float>[totalSegmentCount];

        // Initialize the last segment data and shift count to handle cases where the data length is not a multiple of the vector length.
        // This ensures that the last segment is correctly aligned and padded with zeros if necessary.
        int remainingValueCount = data.Length % vectorLength;
        float[] lastSegment = new float[vectorLength];
        data.AsSpan(^remainingValueCount..).CopyTo(lastSegment);
        vectorSegments[^1] = new Vector<float>(lastSegment);

        for (int i = 0; i < segmentCount; ++i)
        {
            vectorSegments[i] = new(data, i * vectorLength);
        }

        return vectorSegments;
    }
}
