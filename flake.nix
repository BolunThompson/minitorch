{
  description = "Python envirnoment for small projects (ie: no poetry)";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  # Flake outputs
  outputs = { self, nixpkgs }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      # Development environment output
      devShells = forAllSystems ({ pkgs }: {
        default =
          let
            python = pkgs.python311;
          in
          pkgs.mkShell {
            # The Nix packages provided in the environment
            packages = with pkgs; [
              nodejs_22
              ruff
              pyright # I dislike the maintainer but it's fast and up-to-date
              # Python plus helper tools
              (python.withPackages (ps: with ps; [
              # dev
                setuptools
                pip
                virtualenv
              # tooling -- could be moved to global
                ipython
                flit
                jupyter

                # should I include typing stuff?
                typing-extensions

                # Essential libraries
                more-itertools

                # Commonly used libraries. If I'm using this as a play shell,
                # these are essential. If I'm using this as a template for
                # medium-sized unpublished projects, the ones I don't have to
                # use would have to be deleted
                # but, unless it gets annoying, I don't care enough to make
                # a new template.
                colorama
                hypothesis
                mypy
                numba
                numpy
                pre-commit
                pytest
                pytest-env
                pytest-runner
                streamlit
                datasets
                # embeddings
                networkx
                plotly
                pydot
                python-mnist
                torch
                watchdog
              ]))
            ];
          };
      });
    };
}
