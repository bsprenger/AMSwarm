# AMSwarm Tests

This directory contains unit tests for the AMSwarm library using Google Test.

## Building Tests

To build the tests, enable the `BUILD_TESTS` option when configuring with CMake:

```bash
mkdir build
cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=OFF
make
```

If you want to build both tests and Python bindings:

```bash
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
make
```

## Running Tests

Run all tests using CTest:

```bash
cd build
ctest --output-on-failure
```

Or run the test executable directly:

```bash
./cpp/tests/amswarm_tests
```

## Adding New Tests

To add new tests, create a new `.cpp` file in this directory and add it to the test executable in `CMakeLists.txt`.

Example test structure:

```cpp
#include <gtest/gtest.h>
#include "your_header.h"

TEST(TestSuiteName, TestName) {
    // Your test code
    EXPECT_EQ(expected, actual);
}
```

For more information on writing tests, see the [Google Test documentation](https://google.github.io/googletest/).
