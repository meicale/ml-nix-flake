{
  description = "dev shell with uv and cuda out of the box";

  # nixConfig = {
    # extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    # extra-trusted-public-keys = [
      # "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    # ];
  # };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default = with pkgs; mkShell rec {

        packages = [
          cmake
          ninja
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          cudaPackages.cuda_cupti
          cudaPackages.cuda_nvrtc
          cudaPackages.cuda_nvtx
          cudaPackages.cudnn
          cudaPackages.libcublas
          cudaPackages.libcufft
          cudaPackages.libcurand
          cudaPackages.libcusolver
          cudaPackages.libcusparse
          cudaPackages.libnvjitlink
          cudaPackages.nccl
          uv
          python312
          zlib
        ];

        shellHook = ''
          if [ -f pyproject.toml ]; then
            uv sync
            . .venv/bin/activate
          fi
          export LD_LIBRARY_PATH="${lib.makeLibraryPath packages}:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
        '';
      };
    };
}
