using Documenter
using DocumenterVitepress
using SpeedyCalibration

makedocs(
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/SpeedyWeather/SpeedyCalibration.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    sitename  = "SpeedyCalibration.jl",
    authors   = "Niklas Viebig and SpeedyWeather contributors",
    modules   = [SpeedyCalibration],
    checkdocs = :exports,
    pages = [
        "Home"         => "index.md",
        "Getting Started" => [
            "Installation"  => "installation.md",
            "Quick start"   => "quickstart.md",
        ],
        "User Guide" => [
            "Defining parameters" => "parameters.md",
            "Loss configuration"  => "loss.md",
            "Training"            => "training.md",
            "Validation"          => "validation.md",
            "Diagnostics"        => "diagnostics.md",
            "Reference data"     => "reference_data.md",
        ],
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(
    repo       = "github.com/SpeedyWeather/SpeedyCalibration.jl",
    devbranch  = "main",
    push_preview = true,
)
