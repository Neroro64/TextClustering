namespace Embedding.DistanceMetric;

public interface IDistanceMetric<TVector>
{
    static abstract float CalculateDistance(TVector vector1, TVector vector2);
}
