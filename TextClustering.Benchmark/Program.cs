using BenchmarkDotNet.Running;

namespace TextClustering.Benchmark;

internal sealed class Program
{
    internal static void Main(string[] args) => BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}

