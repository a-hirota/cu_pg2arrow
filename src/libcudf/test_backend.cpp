#include <stdio.h>
#include <stdlib.h>
#include <optional>
#include <string>
#include <algorithm>

std::optional<bool> use_hw_decompression()
{
  auto const env = getenv("LIBCUDF_HW_DECOMPRESSION");
  if (env == nullptr) { return std::nullopt; }
  std::string val{env};
  std::transform(
    val.begin(), val.end(), val.begin(), [](unsigned char c) { return std::toupper(c); });
  return val == "ON";
}

int main() {
    // Test without env var
    printf("Test 1: No environment variable\n");
    auto result1 = use_hw_decompression();
    if (result1.has_value()) {
        printf("  Result: %s\n", *result1 ? "Hardware" : "CUDA");
    } else {
        printf("  Result: nullopt (DEFAULT backend)\n");
    }
    
    // Test with OFF
    setenv("LIBCUDF_HW_DECOMPRESSION", "OFF", 1);
    printf("\nTest 2: LIBCUDF_HW_DECOMPRESSION=OFF\n");
    auto result2 = use_hw_decompression();
    if (result2.has_value()) {
        printf("  Result: %s\n", *result2 ? "Hardware" : "CUDA");
    } else {
        printf("  Result: nullopt (DEFAULT backend)\n");
    }
    
    // Test with ON
    setenv("LIBCUDF_HW_DECOMPRESSION", "ON", 1);
    printf("\nTest 3: LIBCUDF_HW_DECOMPRESSION=ON\n");
    auto result3 = use_hw_decompression();
    if (result3.has_value()) {
        printf("  Result: %s\n", *result3 ? "Hardware" : "CUDA");
    } else {
        printf("  Result: nullopt (DEFAULT backend)\n");
    }
    
    return 0;
}
