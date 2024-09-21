namespace TextClustering.Embedding;

public record TermStats(int id, int documentFrequency)
{
    public int Id { get; set; } = id;
    public int NumberOfDocumentsWhereTheTermAppears { get; set; } = documentFrequency;
}
