{
  description = "dev shell with uv and cuda out of the box";

  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

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
          # 1. 定义 WSL 的驱动路径
          WSL_LIB_PATH="/usr/lib/wsl/lib"

          # 2. 检查该路径是否存在
          if [ -d "$WSL_LIB_PATH" ]; then
            echo "✅ Found WSL GPU driver path: $WSL_LIB_PATH"
            
            # 3. 将其加入 LD_LIBRARY_PATH 的最前面
            # 注意：必须放在最前面，确保优先于 Nix Store 的路径搜索
            export LD_LIBRARY_PATH="$WSL_LIB_PATH:${lib.makeLibraryPath packages}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            
            # 4. (可选) 有时需要显式指定 CUDA_HOME
            export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
          else
            echo "❌ ERROR: WSL GPU driver path not found at $WSL_LIB_PATH"
            echo "Please ensure your Windows NVIDIA driver is installed and WSL2 is working."
          fi

          echo "Environment configured. Try running 'python -c \"import torch; print(torch.cuda.is_available())\"'"
        
        '';
      };
    };
}
