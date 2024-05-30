using NeuralQuantumStates
using Documenter

DocMeta.setdocmeta!(NeuralQuantumStates, :DocTestSetup, :(using NeuralQuantumStates); recursive=true)

makedocs(;
    modules=[NeuralQuantumStates],
    authors="Alberto Mercurio",
    sitename="NeuralQuantumStates.jl",
    format=Documenter.HTML(;
        canonical="https://albertomercurio.github.io/NeuralQuantumStates.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/albertomercurio/NeuralQuantumStates.jl",
    devbranch="main",
)
