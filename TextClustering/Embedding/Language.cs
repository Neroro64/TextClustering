namespace TextClustering.Embedding;

public enum Language
{
    Arabic,
    Bulgarian,
    Catalan,
    Czech,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Gujarati,
    Hebrew,
    Hindi,
    Hungarian,
    Indonesian,
    Malaysian,
    Italian,
    Norwegian,
    Polish,
    Portuguese,
    Romanian,
    Russian,
    Slovak,
    Spanish,
    Swedish,
    Turkish,
    Ukrainian,
    Vietnamese
}

internal static class LanguageExtensions
{
    internal static string GetShortCode(this Language language)
    {
        return language switch
        {
            Language.Arabic => "ar",
            Language.Bulgarian => "bg",
            Language.Catalan => "ca",
            Language.Czech => "cs",
            Language.Danish => "da",
            Language.Dutch => "nl",
            Language.English => "en",
            Language.Finnish => "fi",
            Language.French => "fr",
            Language.German => "de",
            Language.Gujarati => "gu",
            Language.Hebrew => "he",
            Language.Hindi => "hi",
            Language.Hungarian => "hu",
            Language.Indonesian => "id",
            Language.Malaysian => "ms",
            Language.Italian => "it",
            Language.Norwegian => "nb",
            Language.Polish => "pl",
            Language.Portuguese => "pt",
            Language.Romanian => "ro",
            Language.Russian => "ru",
            Language.Slovak => "sk",
            Language.Spanish => "es",
            Language.Swedish => "sv",
            Language.Turkish => "tr",
            Language.Ukrainian => "uk",
            Language.Vietnamese => "vi",
            _ => throw new ArgumentOutOfRangeException(nameof(language), language, null),
        };
    }
}
