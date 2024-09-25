namespace Clustering;

public record ClusteringResult(IReadOnlyCollection<int> Labels, IReadOnlyCollection<float> OutliersScores);
