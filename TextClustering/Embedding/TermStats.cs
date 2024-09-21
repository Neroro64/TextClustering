namespace TextClustering.Embedding;

public record TermStats
{
    public int Id { get; init; }
    public int NumberOfDocumentsWhereTheTermAppears { get; set; }
}
