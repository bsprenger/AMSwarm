# CMake Presets Guide

This project uses CMake Presets (CMakePresets.json) to provide convenient build configurations.

## Prerequisites

- CMake 3.14 or higher
- Ninja build system
- Appropriate compilers (GCC and/or Clang)
- OpenMP support (libomp-dev)

## Available Presets

### Basic Build Configurations

- **debug**: Debug build with symbols and no optimization
  ```bash
  cmake --preset debug
  cmake --build --preset debug
  ctest --preset debug
  ```

- **release**: Release build with O3 optimization and native tuning
  ```bash
  cmake --preset release
  cmake --build --preset release
  ctest --preset release
  ```

- **relwithdebinfo**: Release build with debug symbols for profiling
  ```bash
  cmake --preset relwithdebinfo
  cmake --build --preset relwithdebinfo
  ctest --preset relwithdebinfo
  ```

### Compiler-Specific Presets

- **gcc-release**: Release build explicitly using GCC
  ```bash
  cmake --preset gcc-release
  cmake --build --preset gcc-release
  ctest --preset gcc-release
  ```

- **clang-release**: Release build explicitly using Clang
  ```bash
  cmake --preset clang-release
  cmake --build --preset clang-release
  ctest --preset clang-release
  ```

### Sanitizer Presets (for bug detection)

- **asan**: AddressSanitizer for memory error detection
  ```bash
  cmake --preset asan
  cmake --build --preset asan
  ctest --preset asan
  ```

- **tsan**: ThreadSanitizer for thread error detection
  ```bash
  cmake --preset tsan
  cmake --build --preset tsan
  ctest --preset tsan
  ```

- **ubsan**: UndefinedBehaviorSanitizer for undefined behavior detection
  ```bash
  cmake --preset ubsan
  cmake --build --preset ubsan
  ctest --preset ubsan
  ```

### Analysis and Coverage Presets

- **clang-tidy**: Debug build with compile_commands.json for static analysis
  ```bash
  cmake --preset clang-tidy
  cmake --build --preset clang-tidy
  # Run clang-tidy on source files
  cd build/clang-tidy
  find ../../amswarm -name "*.cpp" -exec clang-tidy -p . {} \;
  ```

- **coverage**: Debug build with code coverage instrumentation
  ```bash
  cmake --preset coverage
  cmake --build --preset coverage
  ctest --preset coverage
  # Generate coverage report (requires lcov or similar)
  ```

## Key Features

All presets automatically:
- Use Ninja as the build system for fast, parallel builds
- Generate `compile_commands.json` for IDE/clangd support
- Create build directories in `build/<preset-name>` format
- Disable Python bindings (can be overridden with `-DBUILD_PYTHON_BINDINGS=ON`)
- Enable tests by default

## Listing Available Presets

To see all available presets:
```bash
cmake --list-presets
```

## IDE Integration

All presets generate `compile_commands.json` in their respective build directories. To use with your IDE or language server:

- **VSCode with clangd**: Create a symlink in the project root:
  ```bash
  ln -s build/debug/compile_commands.json compile_commands.json
  ```

- **CLion**: CLion automatically detects CMakePresets.json and shows all presets in the configuration dropdown.

## CI/CD Usage

The CI workflow has been updated to use these presets. See `.github/workflows/ci.yml` for examples.
