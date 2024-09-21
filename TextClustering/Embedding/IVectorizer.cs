namespace TextClustering.Embedding;

public interface IVectorizer
{
    void Reset();
    void Fit(IEnumerable<string> documents);
    IEnumerable<Dictionary<int, float>> Transform(IEnumerable<string> documents);
    IEnumerable<Dictionary<int, float>> FitThenTransform(IEnumerable<string> documents);
}
