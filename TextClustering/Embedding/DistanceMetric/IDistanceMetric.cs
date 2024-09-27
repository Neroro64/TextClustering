#pragma warning disable CA1000
namespace Embedding.DistanceMetric;

/// <summary>
/// Represents a distance metric for vectors.
/// </summary>
public interface IDistanceMetric<in TVector>
{
    static abstract float CalculateDistance(TVector vector1, TVector vector2);
    static abstract float CalculateUnitSphereDistance(TVector vector1, TVector vector2);
}
