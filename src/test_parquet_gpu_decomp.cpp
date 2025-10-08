/*
 * test_parquet_gpu_decomp.cpp
 *
 * Test program for GPU-based Parquet decompression
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "parquet_gpu_decomp.h"

int main(int argc, char *argv[])
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <parquet_file>\n", argv[0]);
		return 1;
	}

	const char *filename = argv[1];
	const char *error_msg = NULL;

	printf("Testing GPU Parquet decompression on: %s\n", filename);

	/* Test metadata extraction */
	int column_indices[] = {0}; /* First column */
	ParquetRowGroupInfo *info = parquetGetRowGroupInfo(
		filename, 0, column_indices, 1, &error_msg);

	if (!info) {
		fprintf(stderr, "ERROR: Failed to get row group info: %s\n",
				error_msg ? error_msg : "unknown");
		return 1;
	}

	printf("Row Group 0:\n");
	printf("  Num rows: %ld\n", info->num_rows);
	printf("  Num columns: %d\n", info->num_columns);

	for (int i = 0; i < info->num_columns; i++) {
		ParquetColumnChunkInfo *col = &info->columns[i];
		printf("  Column %d:\n", i);
		printf("    Compression: %d\n", col->compression_type);
		printf("    Compressed size: %ld\n", col->total_compressed_size);
		printf("    Uncompressed size: %ld\n", col->total_uncompressed_size);
		printf("    Data offset: %ld\n", col->data_page_offset);
		printf("    Num values: %ld\n", col->num_values);
	}

	/* Test GPU decompression (if column is compressed) */
	if (info->columns[0].compression_type != PARQUET_COMP_UNCOMPRESSED) {
		printf("\nTesting GPU decompression...\n");

		void *d_compressed_data[1];
		size_t compressed_sizes[1];
		void *d_uncompressed_data[1];
		size_t uncompressed_sizes[1];

		/* Read compressed data to GPU */
		int rc = parquetReadCompressedToGPU(filename, info,
											d_compressed_data, compressed_sizes,
											&error_msg);
		if (rc < 0) {
			fprintf(stderr, "ERROR: Failed to read compressed data: %s\n",
					error_msg ? error_msg : "unknown");
			parquetFreeRowGroupInfo(info);
			return 1;
		}

		printf("Compressed data read to GPU: %zu bytes\n", compressed_sizes[0]);

		/* Decompress on GPU */
		rc = parquetDecompressOnGPU(info,
									d_compressed_data, compressed_sizes,
									d_uncompressed_data, uncompressed_sizes,
									NULL, /* default stream */
									&error_msg);
		if (rc < 0) {
			fprintf(stderr, "ERROR: GPU decompression failed: %s\n",
					error_msg ? error_msg : "unknown");
			parquetFreeRowGroupInfo(info);
			return 1;
		}

		printf("GPU decompression successful: %zu bytes\n", uncompressed_sizes[0]);
		printf("Compression ratio: %.2fx\n",
			   (double)uncompressed_sizes[0] / compressed_sizes[0]);

		/* Cleanup GPU memory */
		cudaFree(d_compressed_data[0]);
		cudaFree(d_uncompressed_data[0]);
	} else {
		printf("\nColumn is uncompressed, skipping decompression test.\n");
	}

	parquetFreeRowGroupInfo(info);

	printf("\nTest completed successfully!\n");
	return 0;
}
