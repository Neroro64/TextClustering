namespace Clustering;

public record ClusteringResult(IReadOnlyCollection<Guid> Labels, IReadOnlyCollection<float> OutlierScores);
