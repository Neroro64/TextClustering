using System.Collections.ObjectModel;

namespace Embedding;

/// <summary>
///     Interface for Vectorizer, providing methods to fit and transform documents into vectors of type <typeparamref name="TVector"/>
/// </summary>
/// <typeparam name="TVector">The type of vector used to represent documents.</typeparam>
public interface IVectorizer<TVector>
{
    /// <summary>
    ///     Resets the vectorizer to its initial state, clearing the vocabulary and total document count.
    /// </summary>
    void Reset();

    /// <summary>
    ///     Fits the vectorizer to a collection of documents by extracting term frequencies from each document
    ///     and updating the vocabulary accordingly.
    /// </summary>
    /// <param name="documents">Collection of text documents to fit the vectorizer to.</param>
    void Fit(IEnumerable<string> documents);

    /// <summary>
    ///     Transforms a collection of text documents into vectors of type <typeparamref name="TVector"/>
    /// </summary>
    /// <param name="documents">Collection of text documents to transform.</param>
    /// <returns>Array of sparse vectors, where each vector represents a document and contains term IDs as keys and term frequencies as values.</returns>
    ReadOnlyCollection<TVector> Transform(IEnumerable<string> documents);

    /// <summary>
    ///     Fits the vectorizer to a collection of documents by extracting term frequencies from each document
    ///     and updating the vocabulary accordingly, then transforms the same documents into vectors of type <typeparamref name="TVector"/>
    /// </summary>
    /// <param name="documents">Collection of text documents to fit and transform.</param>
    /// <returns>Array of sparse vectors, where each vector represents a document and contains term IDs as keys and term frequencies as values.</returns>
    ReadOnlyCollection<TVector> FitThenTransform(IEnumerable<string> documents);
}
