namespace TextClustering.Embedding;

public interface IVectorizer
{
    void Reset();
    void Fit(IEnumerable<string> documents);
    Dictionary<int, float>[] Transform(IEnumerable<string> documents);
    Dictionary<int, float>[] FitThenTransform(IEnumerable<string> documents);
}
